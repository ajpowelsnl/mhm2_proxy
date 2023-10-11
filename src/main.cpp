/*
 HipMer v 2.0, Copyright (c) 2020, The Regents of the University of California,
 through Lawrence Berkeley National Laboratory (subject to receipt of any required
 approvals from the U.S. Dept. of Energy).  All rights reserved."

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 (1) Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 (2) Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 (3) Neither the name of the University of California, Lawrence Berkeley National
 Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to
 endorse or promote products derived from this software without specific prior
 written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 DAMAGE.

 You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades
 to the features, functionality or performance of the source code ("Enhancements") to
 anyone; however, if you choose to make your Enhancements available either publicly,
 or directly to Lawrence Berkeley National Laboratory, without imposing a separate
 written license agreement for such Enhancements, then you hereby grant the following
 license: a  non-exclusive, royalty-free perpetual license to install, use, modify,
 prepare derivative works, incorporate into other computer software, distribute, and
 sublicense such enhancements or derivative works thereof, in binary and source code
 form.
*/

#include <sys/resource.h>

#include "contigging.hpp"
#include "klign.hpp"
#include "fastq.hpp"
#include "stage_timers.hpp"
#include "upcxx_utils.hpp"
#include "upcxx_utils/thread_pool.hpp"
#include "utils.hpp"

#include "kmer.hpp"

using std::fixed;
using std::setprecision;

using namespace upcxx_utils;

void init_devices();
void done_init_devices();

void merge_reads(vector<string> reads_fname_list, int qual_offset, double &elapsed_write_io_t,
                 vector<PackedReads *> &packed_reads_list, bool checkpoint,  int min_kmer_len);

int main(int argc, char **argv) {
  BaseTimer init_timer("upcxx::init");
  BaseTimer first_barrier("FirstBarrier");
  init_timer.start();
  upcxx::init();
  auto init_entry_msm_fut = init_timer.reduce_start();
  init_timer.stop();
  auto init_timings_fut = init_timer.reduce_timings();
  upcxx::promise<> report_init_timings(1);


  // we wish to have all ranks start at the same time to determine actual timing
  first_barrier.start();
  barrier();
  first_barrier.stop();
  when_all(report_init_timings.get_future(), init_entry_msm_fut, init_timings_fut, first_barrier.reduce_timings())
      .then([](upcxx_utils::MinSumMax<double> entry_msm, upcxx_utils::ShTimings sh_timings,
               upcxx_utils::ShTimings sh_first_barrier_timings) {
        SLOG_VERBOSE("upcxx::init Before=", entry_msm.to_string(), "\n");
        SLOG_VERBOSE("upcxx::init After=", sh_timings->to_string(), "\n");
        SLOG_VERBOSE("upcxx::init FirstBarrier=", sh_first_barrier_timings->to_string(), "\n");
      });
  auto start_t = std::chrono::high_resolution_clock::now();
  auto init_start_t = start_t;

  // keep the exact command line arguments before options may have modified anything
  string executed = argv[0];
  executed += ".py";  // assume the python wrapper was actually called
  for (int i = 1; i < argc; i++) executed = executed + " " + argv[i];
  auto options = make_shared<Options>();
  // if we don't load, return "command not found"
  if (!options->load(argc, argv)) return 127;
  SLOG_VERBOSE("Executed as: ", executed, "\n");
  report_init_timings.fulfill_anonymous(1);

  SLOG_VERBOSE(KLCYAN, "Timing reported as min/my/average/max, balance", KNORM, "\n");

  ProgressBar::SHOW_PROGRESS = options->show_progress;
  auto max_kmer_store = options->max_kmer_store_mb * ONE_MB;

  SLOG_VERBOSE("Process 0 on node 0 is initially pinned to ", get_proc_pin(), "\n");
  // pin ranks only in production
  if (options->pin_by == "cpu")
    pin_cpu();
  else if (options->pin_by == "core")
    pin_core();
  else if (options->pin_by == "numa")
    pin_numa();

  // update rlimits on RLIMIT_NOFILE files if necessary
  auto num_input_files = options->reads_fnames.size();
  if (num_input_files > 1) {
    struct rlimit limits;
    int status = getrlimit(RLIMIT_NOFILE, &limits);
    if (status == 0) {
      limits.rlim_cur = std::min(limits.rlim_cur + num_input_files * 8, limits.rlim_max);
      status = setrlimit(RLIMIT_NOFILE, &limits);
      SLOG_VERBOSE("Set RLIMIT_NOFILE to ", limits.rlim_cur, "\n");
    }
    if (status != 0) SWARN("Could not get/set rlimits for NOFILE\n");
  }
  const int num_threads = options->max_worker_threads;  // reserve up to threads in the singleton thread pool.
  upcxx_utils::ThreadPool::get_single_pool(num_threads);
  // FIXME if (!options->max_worker_threads) upcxx_utils::FASRPCCounts::use_worker_thread() = false;
  SLOG_VERBOSE("Allowing up to ", num_threads, " extra threads in the thread pool\n");

  if (!upcxx::rank_me()) {
    // get total file size across all libraries
    double tot_file_size = 0;
    for (auto const &reads_fname : options->reads_fnames) {
      auto spos = reads_fname.find_first_of(':');  // support paired reads
      if (spos == string::npos) {
        auto sz = get_file_size(reads_fname);
        SLOG("Reads file ", reads_fname, " is ", get_size_str(sz), "\n");
        tot_file_size += sz;
      } else {
        // paired files
        auto r1 = reads_fname.substr(0, spos);
        auto s1 = get_file_size(r1);
        auto r2 = reads_fname.substr(spos + 1);
        auto s2 = get_file_size(r2);
        SLOG("Paired files ", r1, " and ", r2, " are ", get_size_str(s1), " and ", get_size_str(s2), "\n");
        tot_file_size += s1 + s2;
      }
    }
    SOUT("Total size of ", options->reads_fnames.size(), " input file", (options->reads_fnames.size() > 1 ? "s" : ""), " is ",
         get_size_str(tot_file_size), "\n");
    auto nodes = upcxx::rank_n() / upcxx::local_team().rank_n();
    auto total_free_mem = get_free_mem() * nodes;
    if (total_free_mem < 3 * tot_file_size)
      SWARN("There may not be enough memory in this job of ", nodes,
            " nodes for this amount of data.\n\tTotal free memory is approx ", get_size_str(total_free_mem),
            " and should be at least 3x the data size of ", get_size_str(tot_file_size), "\n");
  }

  init_devices();

  Contigs ctgs;
  int max_kmer_len = 0;
  int max_expected_ins_size = 0;

    MemoryTrackerThread memory_tracker;  // write only to mhm2.log file(s), not a separate one too
    memory_tracker.start();
    SLOG(KBLUE, "Starting with ", get_size_str(get_free_mem()), " free on node 0", KNORM, "\n");
    PackedReads::PackedReadsList packed_reads_list;
    for (auto const &reads_fname : options->reads_fnames) {
      packed_reads_list.push_back(new PackedReads(options->qual_offset, get_merged_reads_fname(reads_fname)));
    }
    double elapsed_write_io_t = 0;
    if ((!options->restart || !options->checkpoint_merged) && !options->kmer_lens.empty()) {
      // merge the reads and insert into the packed reads memory cache
      
      stage_timers.merge_reads->start();
      merge_reads(options->reads_fnames, options->qual_offset, elapsed_write_io_t, packed_reads_list, options->checkpoint_merged,
                  options->kmer_lens[0]);
      stage_timers.merge_reads->stop();
      
    } else {
      // since this is a restart with checkpoint_merged true, the merged reads should be on disk already
      // load the merged reads instead of merge the original ones again
      stage_timers.cache_reads->start();
      double free_mem = (!rank_me() ? get_free_mem() : 0);
      upcxx::barrier();
      PackedReads::load_reads(packed_reads_list);
      stage_timers.cache_reads->stop();
      SLOG_VERBOSE(KBLUE, "Cache used ", setprecision(2), fixed, get_size_str(free_mem - get_free_mem()), " memory on node 0",
                   KNORM, "\n");
    }
    unsigned rlen_limit = 0;
    for (auto packed_reads : packed_reads_list) {
      rlen_limit = max(rlen_limit, packed_reads->get_max_read_len());
      packed_reads->report_size();
    }

    if (!options->ctgs_fname.empty()) {
      stage_timers.load_ctgs->start();
      ctgs.load_contigs(options->ctgs_fname);
      stage_timers.load_ctgs->stop();
    }
    std::chrono::duration<double> init_t_elapsed = std::chrono::high_resolution_clock::now() - init_start_t;
    SLOG("\n");
    SLOG(KBLUE, "Completed initialization in ", setprecision(2), fixed, init_t_elapsed.count(), " s at ", get_current_time(), " (",
         get_size_str(get_free_mem()), " free memory on node 0)", KNORM, "\n");
    int prev_kmer_len = options->prev_kmer_len;
    int ins_avg = 0;
    int ins_stddev = 0;

    done_init_devices();

    // contigging loops
    if (options->kmer_lens.size()) {
      max_kmer_len = options->kmer_lens.back();
      for (auto kmer_len : options->kmer_lens) {
        auto max_k = (kmer_len / 32 + 1) * 32;
        LOG(upcxx_utils::GasNetVars::getUsedShmMsg(), "\n");

    #define CONTIG_K(KMER_LEN)                                                                                                         \
    case KMER_LEN:                                                                                                                   \
    contigging<KMER_LEN>(kmer_len, prev_kmer_len, rlen_limit, packed_reads_list, ctgs, max_expected_ins_size, ins_avg, ins_stddev, \
                         options);                                                                                                 \
    break

        switch (max_k) {
          CONTIG_K(32);
    #if MAX_BUILD_KMER >= 64
          CONTIG_K(64);
    #endif
    #if MAX_BUILD_KMER >= 96
          CONTIG_K(96);
    #endif
    #if MAX_BUILD_KMER >= 128
          CONTIG_K(128);
    #endif
    #if MAX_BUILD_KMER >= 160
          CONTIG_K(160);
    #endif
          default: DIE("Built for max k = ", MAX_BUILD_KMER, " not k = ", max_k);
        }
    #undef CONTIG_K

        prev_kmer_len = kmer_len;
      }
    }


    // cleanup
    FastqReaders::close_all();  // needed to cleanup any open files in this singleton
    auto fin_start_t = std::chrono::high_resolution_clock::now();
    for (auto packed_reads : packed_reads_list) {
      delete packed_reads;
    }
    packed_reads_list.clear();

    // output final assembly
    SLOG(KBLUE "_________________________", KNORM, "\n");
    stage_timers.dump_ctgs->start();
    ctgs.dump_contigs("final_assembly.fasta", options->min_ctg_print_len);
    stage_timers.dump_ctgs->stop();

    SLOG(KBLUE "_________________________", KNORM, "\n");
    ctgs.print_stats(options->min_ctg_print_len);
    std::chrono::duration<double> fin_t_elapsed = std::chrono::high_resolution_clock::now() - fin_start_t;
    SLOG("\n");
    SLOG(KBLUE, "Completed finalization in ", setprecision(2), fixed, fin_t_elapsed.count(), " s at ", get_current_time(), " (",
         get_size_str(get_free_mem()), " free memory on node 0)", KNORM, "\n");

    SLOG(KBLUE "_________________________", KNORM, "\n");
    SLOG("Stage timing:\n");
    if (!options->restart)
      SLOG("    ", stage_timers.merge_reads->get_final(), "\n");
    else
      SLOG("    ", stage_timers.cache_reads->get_final(), "\n");
    SLOG("    ", stage_timers.analyze_kmers->get_final(), "\n");
    SLOG("      -> ", stage_timers.kernel_kmer_analysis->get_final(), "\n");
    SLOG("    ", stage_timers.dbjg_traversal->get_final(), "\n");
    SLOG("    ", stage_timers.alignments->get_final(), "\n");
    SLOG("      -> ", stage_timers.kernel_alns->get_final(), "\n");
   
    if (options->shuffle_reads) SLOG("    ", stage_timers.shuffle_reads->get_final(), "\n");
    SLOG("    ", stage_timers.cgraph->get_final(), "\n");
    SLOG("    FASTQ total read time: ", FastqReader::get_io_time(), "\n");
    SLOG("    merged FASTQ write time: ", elapsed_write_io_t, "\n");
    SLOG("    Contigs write time: ", stage_timers.dump_ctgs->get_elapsed(), "\n");
    SLOG(KBLUE "_________________________", KNORM, "\n");
    memory_tracker.stop();
    std::chrono::duration<double> t_elapsed = std::chrono::high_resolution_clock::now() - start_t;
    SLOG("Finished in ", setprecision(2), fixed, t_elapsed.count(), " s at ", get_current_time(), " for ", MHM2_VERSION, "\n");

  FastqReaders::close_all();


  upcxx_utils::ThreadPool::join_single_pool();  // cleanup singleton thread pool
  upcxx_utils::Timings::wait_pending();         // ensure all outstanding timing summaries have printed
  barrier();

#ifdef DEBUG
  _dbgstream.flush();
  while (close_dbg())
    ;
#endif
  upcxx::finalize();
  return 0;
}

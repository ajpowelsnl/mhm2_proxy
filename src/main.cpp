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
#include "fastq.hpp"
#include "upcxx_utils.hpp"
#include "upcxx_utils/thread_pool.hpp"
#include "utils.hpp"

#include "kmer.hpp"

using std::fixed;
using std::setprecision;

using namespace upcxx_utils;

void init_devices();
void done_init_devices();

void merge_reads(vector<string> reads_fname_list, int qual_offset,
                 vector<PackedReads *> &packed_reads_list, bool checkpoint,  int min_kmer_len);

int main(int argc, char **argv) {
  
  upcxx::init();
  
  barrier();
  
  // keep the exact command line arguments before options may have modified anything
  string executed = argv[0];
  executed += ".py";  // assume the python wrapper was actually called
  for (int i = 1; i < argc; i++) executed = executed + " " + argv[i];
  auto options = make_shared<Options>();
  // if we don't load, return "command not found"
  if (!options->load(argc, argv)) return 127;
  SLOG_VERBOSE("Executed as: ", executed, "\n");
  
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
      
        // paired files
        auto r1 = reads_fname.substr(0, spos);
        auto s1 = get_file_size(r1);
        auto r2 = reads_fname.substr(spos + 1);
        auto s2 = get_file_size(r2);
        SLOG("Paired files ", r1, " and ", r2, " are ", get_size_str(s1), " and ", get_size_str(s2), "\n");
        tot_file_size += s1 + s2;
      
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
    
    // merge the reads and insert into the packed reads memory cache
    merge_reads(options->reads_fnames, options->qual_offset,  packed_reads_list, options->checkpoint_merged,
                  options->kmer_lens[0]);
      
    
    unsigned rlen_limit = 0;
    for (auto packed_reads : packed_reads_list) {
      rlen_limit = max(rlen_limit, packed_reads->get_max_read_len());
      packed_reads->report_size();
    }
   
    SLOG("\n");
    SLOG(KBLUE, "Completed initialization  at ", get_current_time(), " (",
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
        //LOG(upcxx_utils::GasNetVars::getUsedShmMsg(), "\n");

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
    
    for (auto packed_reads : packed_reads_list) {
      delete packed_reads;
    }
    packed_reads_list.clear();

    // output final assembly
    SLOG(KBLUE "_________________________", KNORM, "\n");
    
    ctgs.dump_contigs("final_assembly.fasta", options->min_ctg_print_len);
   
    SLOG(KBLUE "_________________________", KNORM, "\n");
    ctgs.print_stats(options->min_ctg_print_len);
    
    SLOG("\n");
    SLOG(KBLUE, "Completed finalization at ", get_current_time(), " (",
         get_size_str(get_free_mem()), " free memory on node 0)", KNORM, "\n");

    SLOG(KBLUE "_________________________", KNORM, "\n");
    memory_tracker.stop();
    
    SLOG("Finished at ", get_current_time(), " for ", MHM2_VERSION, "\n");

  FastqReaders::close_all();


  upcxx_utils::ThreadPool::join_single_pool();  // cleanup singleton thread pool
  //upcxx_utils::Timings::wait_pending();         // ensure all outstanding timing summaries have printed
  barrier();

#ifdef DEBUG
  _dbgstream.flush();
  while (close_dbg())
    ;
#endif
  upcxx::finalize();
  return 0;
}

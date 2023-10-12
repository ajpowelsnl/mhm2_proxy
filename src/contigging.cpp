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

#include "contigging.hpp"
#include "kcount/kcount.hpp"
#include "kmer_dht.hpp"
#include "upcxx_utils.hpp"

using namespace upcxx;
using namespace upcxx_utils;

using std::fixed;
using std::setprecision;
using std::shared_ptr;
using std::string;
using std::tie;
using std::vector;

template <int MAX_K>
void traverse_debruijn_graph(unsigned kmer_len, dist_object<KmerDHT<MAX_K>> &kmer_dht, Contigs &my_uutigs);

static uint64_t estimate_num_kmers(unsigned kmer_len, vector<PackedReads *> &packed_reads_list) {
  BarrierTimer timer(__FILEFUNC__);
  int64_t num_kmers = 0;
  int64_t num_reads = 0;
  int64_t tot_num_reads = 0;
  for (auto packed_reads : packed_reads_list) {
    tot_num_reads += packed_reads->get_local_num_reads();
    packed_reads->reset();
    string id, seq, quals;
    ProgressBar progbar(packed_reads->get_local_num_reads(), "Scanning reads to estimate number of kmers");

    for (int i = 0; i < 100000; i++) {
      if (!packed_reads->get_next_read(id, seq, quals)) break;
      progbar.update();
      // do not read the entire data set for just an estimate
      if (seq.length() < kmer_len) continue;
      num_kmers += seq.length() - kmer_len + 1;
      num_reads++;
    }
    progbar.done();
    barrier();
  }
  DBG("This rank processed ", num_reads, " reads, and found ", num_kmers, " kmers\n");
  auto all_num_reads = reduce_one(num_reads, op_fast_add, 0).wait();
  auto all_tot_num_reads = reduce_one(tot_num_reads, op_fast_add, 0).wait();
  auto all_num_kmers = reduce_all(num_kmers, op_fast_add).wait();

  SLOG_VERBOSE("Processed ", perc_str(all_num_reads, all_tot_num_reads), " reads, and estimated a maximum of ",
               (all_num_reads > 0 ? all_num_kmers * (all_tot_num_reads / all_num_reads) : 0), " kmers\n");
  return num_reads > 0 ? num_kmers * tot_num_reads / num_reads : 0;
}

template <int MAX_K>
void contigging(int kmer_len, int prev_kmer_len, int rlen_limit, vector<PackedReads *> &packed_reads_list, Contigs &ctgs,
                int &max_expected_ins_size, int &ins_avg, int &ins_stddev, shared_ptr<Options> options) {
  auto loop_start_t = std::chrono::high_resolution_clock::now();
  SLOG(KBLUE, "_________________________", KNORM, "\n");
  SLOG(KBLUE, "Contig generation k = ", kmer_len, KNORM, "\n");
  SLOG("\n");
  bool is_debug = false;
#ifdef DEBUG
  is_debug = true;
#endif

  auto max_kmer_store = options->max_kmer_store_mb * ONE_MB;

  string uutigs_fname("uutigs-" + to_string(kmer_len) + ".fasta");
  if (options->ctgs_fname != uutigs_fname) {
    Kmer<MAX_K>::set_k(kmer_len);
    // duration of kmer_dht
    
    int64_t my_num_kmers = estimate_num_kmers(kmer_len, packed_reads_list);
    // use the max among all ranks
    my_num_kmers = reduce_all(my_num_kmers, op_fast_max).wait();
    dist_object<KmerDHT<MAX_K>> kmer_dht(world(), my_num_kmers, max_kmer_store, options->max_rpcs_in_flight,
                                         options->use_heavy_hitters, options->use_qf);
    barrier();
    analyze_kmers(kmer_len, prev_kmer_len, options->qual_offset, packed_reads_list, options->dmin_thres, ctgs, kmer_dht,
                  options->dump_kmers);
    
    barrier();
    
    traverse_debruijn_graph(kmer_len, kmer_dht, ctgs);
    
    if (is_debug) {
      ctgs.dump_contigs(uutigs_fname, 0);
    }
  }

  if (kmer_len < options->kmer_lens.back()) {
    if (kmer_len == options->kmer_lens.front()) {
      size_t num_reads = 0;
      for (auto packed_reads : packed_reads_list) {
        num_reads += packed_reads->get_local_num_reads();
      }
      auto avg_num_reads = reduce_one(num_reads, op_fast_add, 0).wait() / rank_n();
      auto max_num_reads = reduce_one(num_reads, op_fast_max, 0).wait();
      SLOG_VERBOSE("Avg reads per rank ", avg_num_reads, " max ", max_num_reads, " (balance ",
                   (double)avg_num_reads / max_num_reads, ")\n");
     
    }
    barrier();
      

  }
  barrier();
  if (is_debug || options->checkpoint) {
    string contigs_fname("contigs-" + to_string(kmer_len) + ".fasta");
    ctgs.dump_contigs(contigs_fname, 0);
  }
  SLOG(KBLUE "_________________________", KNORM, "\n");
  ctgs.print_stats(500);
  std::chrono::duration<double> loop_t_elapsed = std::chrono::high_resolution_clock::now() - loop_start_t;
  SLOG("\n");
  SLOG(KBLUE, "Completed contig round k = ", kmer_len, " in ", setprecision(2), fixed, loop_t_elapsed.count(), " s at ",
       get_current_time(), " (", get_size_str(get_free_mem()), " free memory on node 0)", KNORM, "\n");
  barrier();
}

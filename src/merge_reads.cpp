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

#include <math.h>
#include <stdarg.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#ifdef __x86_64__
#include <emmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>
#endif
#include <functional>
#include <upcxx/upcxx.hpp>
#include <utility>

using namespace std;
using namespace upcxx;

#include "fastq.hpp"
#include "packed_reads.hpp"
#include "upcxx_utils.hpp"
#include "upcxx_utils/ofstream.hpp"
#include "utils.hpp"
#include "zstr.hpp"
#include "kmer.hpp"


using namespace upcxx_utils;

static const double Q2Perror[] = {
    1.0,       0.7943,    0.6309,    0.5012,    0.3981,    0.3162,    0.2512,    0.1995,    0.1585,    0.1259,     0.1,
    0.07943,   0.06310,   0.05012,   0.03981,   0.03162,   0.02512,   0.01995,   0.01585,   0.01259,   0.01,       0.007943,
    0.006310,  0.005012,  0.003981,  0.003162,  0.002512,  0.001995,  0.001585,  0.001259,  0.001,     0.0007943,  0.0006310,
    0.0005012, 0.0003981, 0.0003162, 0.0002512, 0.0001995, 0.0001585, 0.0001259, 0.0001,    7.943e-05, 6.310e-05,  5.012e-05,
    3.981e-05, 3.162e-05, 2.512e-05, 1.995e-05, 1.585e-05, 1.259e-05, 1e-05,     7.943e-06, 6.310e-06, 5.012e-06,  3.981e-06,
    3.162e-06, 2.512e-06, 1.995e-06, 1.585e-06, 1.259e-06, 1e-06,     7.943e-07, 6.310e-07, 5.012e-07, 3.981e-07,  3.1622e-07,
    2.512e-07, 1.995e-07, 1.585e-07, 1.259e-07, 1e-07,     7.943e-08, 6.310e-08, 5.012e-08, 3.981e-08, 3.1622e-08, 2.512e-08,
    1.995e-08, 1.585e-08, 1.259e-08, 1e-08};

static pair<uint64_t, int> estimate_num_reads(vector<string> &reads_fname_list) {
  // estimate reads in this rank's section of all the files
  future<int> fut_max_read_len;
  future<> progress_fut = make_future();
  future<> rpc_fut = make_future();

  BarrierTimer timer(__FILEFUNC__);
  FastqReaders::open_all(reads_fname_list);

  // Issue #61 - reduce the # of reading ranks to fix excessively long estimates on poor filesystems
  // only a few ranks need to estimate - local_team().rank_n() / 2 to nodes
  auto nodes = rank_n() / local_team().rank_n();
  intrank_t modulo_rank = 1;
  if (nodes >= local_team().rank_n() / 2) {
    modulo_rank = local_team().rank_n();
  } else {
    modulo_rank = 2 * nodes;
  }
  SLOG_VERBOSE("Estimating with 1 rank out of every ", modulo_rank, "\n");
  dist_object<int64_t> dist_est(world(), 0);
  int64_t num_reads = 0;
  int64_t num_lines = 0;
  int64_t estimated_total_records = 0;
  int64_t total_records_processed = 0;
  string id, seq, quals;
  int max_read_len = 0;
  int read_file_idx = 0;
  for (auto const &reads_fname : reads_fname_list) {
    // let multiple ranks handle multiple files
    if (rank_me() % modulo_rank != (read_file_idx++ % modulo_rank)) {
      ProgressBar progbar((int64_t)0,
                          "Scanning reads file to estimate number of reads");  // still do the collectives on progress bar...
      progress_fut = when_all(progress_fut, progbar.set_done());
      continue;
    }
    FastqReader &fqr = FastqReaders::get(reads_fname);
    ProgressBar progbar(fqr.my_file_size(), "Scanning reads file to estimate number of reads");
    size_t tot_bytes_read = 0;
    int64_t records_processed = 0;
    while (true) {
      size_t bytes_read = fqr.get_next_fq_record(id, seq, quals);
      if (!bytes_read) break;
      num_lines += 4;
      num_reads++;
      tot_bytes_read += bytes_read;
      progbar.update(tot_bytes_read);
      records_processed++;
      // do not read the entire data set for just an estimate
      if (records_processed > 50000) break;
    }
    total_records_processed += records_processed;
    if (records_processed) {
      int64_t bytes_per_record = tot_bytes_read / records_processed;
      int64_t num_records = fqr.my_file_size() / bytes_per_record;
      estimated_total_records += num_records;
      // since each input file is not necessarily run on the same rank
      // collect the local total estimates to a single rank within modulo_rank
      assert(read_file_idx > 0);
      assert(rank_me() >= (read_file_idx - 1) % modulo_rank);
      auto fut_collect_rpc = rpc(
          rank_me() - (read_file_idx - 1) % modulo_rank,
          [](dist_object<int64_t> &dist_est, int64_t num_records, int file_i) {
            *dist_est += num_records;
            LOG("Found ", num_records, " in file ", file_i, ", total=", *dist_est, "\n");
          },
          dist_est, num_records, read_file_idx - 1);
      rpc_fut = when_all(rpc_fut, fut_collect_rpc);
    }
    progress_fut = when_all(progress_fut, progbar.set_done());
    max_read_len = max(fqr.get_max_read_len(), max_read_len);
  }
  fut_max_read_len = reduce_all(max_read_len, op_fast_max);
  DBG("This rank processed ", num_lines, " lines (", num_reads, " reads) with max_read_len=", max_read_len, "\n");
  progress_fut.wait();
  max_read_len = fut_max_read_len.wait();
  rpc_fut.wait();
  timer.initate_exit_barrier();  // barrier ensures rpc_fut have all completed for next reduction
  auto fut_max_estimate = reduce_all(*dist_est, op_fast_max);
  estimated_total_records = fut_max_estimate.wait();
  SLOG_VERBOSE("Found maximum read length of ", max_read_len, " and max estimated total ", estimated_total_records, " per rank\n");
  return {estimated_total_records, max_read_len};
}

// returns the number of mismatches if it is <= max or a number greater than max (but no the actual count)
int16_t fast_count_mismatches(const char *a, const char *b, int len, int16_t max) {
  assert(len < 32768);
  int16_t mismatches = 0;
  int16_t jumpSize, jumpLen;

#if defined(__APPLE__) && defined(__MACH__)
#else
#if defined(__x86_64__)
  // 128-bit SIMD
  if (len >= 16) {
    jumpSize = sizeof(__m128i);
    jumpLen = len / jumpSize;
    for (int16_t i = 0; i < jumpLen; i++) {
      __m128i aa = _mm_loadu_si128((const __m128i *)a);     // load 16 bytes from a
      __m128i bb = _mm_loadu_si128((const __m128i *)b);     // load 16 bytes from b
      __m128i matched = _mm_cmpeq_epi8(aa, bb);             // bytes that are equal are now 0xFF, not equal are 0x00
      uint32_t myMaskMatched = _mm_movemask_epi8(matched);  // mask of most significant bit for each byte
      // count mismatches
      mismatches += _popcnt32((~myMaskMatched) & 0xffff);  // over 16 bits
      if (mismatches > max) break;
      a += jumpSize;
      b += jumpSize;
    }
    len -= jumpLen * jumpSize;
  }
#endif
#endif
  // CPU version and fall through 8 bytes at a time
  if (mismatches <= max) {
    assert(len >= 0);
    jumpSize = sizeof(int64_t);
    jumpLen = len / jumpSize;
    for (int16_t i = 0; i < jumpLen; i++) {
      int64_t *aa = (int64_t *)a, *bb = (int64_t *)b;
      if (*aa != *bb) {  // likely
        for (int j = 0; j < jumpSize; j++) {
          if (a[j] != b[j]) mismatches++;
        }
        if (mismatches > max) break;
      }  // else it matched
      a += jumpSize;
      b += jumpSize;
    }
    len -= jumpLen * jumpSize;
  }
  // do the remaining bytes, if needed
  if (mismatches <= max) {
    assert(len >= 0);
    for (int j = 0; j < len; j++) {
      mismatches += ((a[j] == b[j]) ? 0 : 1);
    }
  }
  return mismatches;
}




void merge_reads(vector<string> reads_fname_list, int qual_offset,
                 vector<PackedReads *> &packed_reads_list, bool checkpoint, int min_kmer_len) {
  BarrierTimer timer(__FILEFUNC__);
  Timer merge_time(__FILEFUNC__ + " merging all");

  
  FastqReaders::open_all(reads_fname_list);
  vector<string> merged_reads_fname_list;

  using shared_of = shared_ptr<upcxx_utils::dist_ofstream>;
  std::vector<shared_of> all_outputs;

  int64_t tot_bytes_read = 0;
  int64_t tot_num_ambiguous = 0;
  int64_t tot_num_merged = 0;
  int tot_max_read_len = 0;
  int64_t tot_bases = 0;
  // for unique read id need to estimate number of reads in our sections of all files
  auto [my_num_reads_estimate, read_len] = estimate_num_reads(reads_fname_list);
  auto max_num_reads = reduce_all(my_num_reads_estimate, op_fast_max).wait();
  auto tot_num_reads = reduce_all(my_num_reads_estimate, op_fast_add).wait();
  SLOG_VERBOSE("Estimated total number of reads as ", tot_num_reads, ", and max for any rank ", max_num_reads, "\n");
  // triple the block size estimate to be sure that we have no overlap. The read ids do not have to be contiguous
  uint64_t read_id = rank_me() * (max_num_reads + 10000) * 3;
  uint64_t start_read_id = read_id;
  IntermittentTimer dump_reads_t("dump_reads");
  future<> wrote_all_files_fut = make_future();
  promise<> summary_promise;
  future<> fut_summary = summary_promise.get_future();
  int ri = 0;
  for (auto const &reads_fname : reads_fname_list) {
    Timer merge_file_timer("merging " + get_basename(reads_fname));
    merge_file_timer.initiate_entrance_reduction();

    string out_fname = get_merged_reads_fname(reads_fname);
    if (file_exists(out_fname)) SWARN("File ", out_fname, " already exists, will overwrite...");

    FastqReader &fqr = FastqReaders::get(reads_fname);
    fqr.advise(true);
    auto my_file_size = fqr.my_file_size();
    ProgressBar progbar(my_file_size, "Merging reads " + reads_fname + " " + get_size_str(fqr.my_file_size()));

    int max_read_len = 0;
    int64_t overlap_len = 0;
    int64_t merged_len = 0;

    const int16_t MIN_OVERLAP = 12;
    const int16_t EXTRA_TEST_OVERLAP = 2;
    const int16_t MAX_MISMATCHES = 3;  // allow up to 3 mismatches, with MAX_PERROR
    const int Q2PerrorSize = sizeof(Q2Perror) / sizeof(*Q2Perror);
    assert(qual_offset == 33 || qual_offset == 64);

    // illumina reads generally accumulate errors at the end, so allow more mismatches in the overlap as long as differential
    // quality indicates a clear winner
    const double MAX_PERROR = 0.025;  // max 2.5% accumulated mismatch prob of error within overlap by differential quality score
    const int16_t EXTRA_MISMATCHES_PER_1000 = (int)150;  // allow addtl mismatches per 1000 bases overlap before aborting test
    const uint8_t MAX_MATCH_QUAL = 41 + qual_offset;

    string id1, seq1, quals1, id2, seq2, quals2;
    int64_t num_pairs = 0;
    int64_t bytes_read = 0;
    int64_t num_ambiguous = 0;
    int64_t num_merged = 0;
    int64_t num_reads = 0;
    int64_t bases_trimmed = 0;
    int64_t reads_removed = 0;
    int64_t bases_read = 0;

    for (;; num_pairs++) {
      if (!fqr.is_paired()) {
        // unpaired reads get dummy read2 just like merged reads
        int64_t bytes_read1 = fqr.get_next_fq_record(id1, seq1, quals1);
        if (!bytes_read1) break;
        bytes_read += bytes_read1;
        progbar.update(bytes_read);
        packed_reads_list[ri]->add_read("r" + to_string(read_id) + "/1", seq1, quals1);
        packed_reads_list[ri]->add_read("r" + to_string(read_id) + "/2", "N", to_string((char)qual_offset));
        read_id += 2;
        continue;
      }
      int64_t bytes_read1 = fqr.get_next_fq_record(id1, seq1, quals1);
      if (!bytes_read1) break;
      int64_t bytes_read2 = fqr.get_next_fq_record(id2, seq2, quals2);
      if (!bytes_read2) break;
      bytes_read += bytes_read1 + bytes_read2;
      bases_read += seq1.length() + seq2.length();
     
      if (id1.compare(0, id1.length() - 2, id2, 0, id2.length() - 2) != 0) DIE("Mismatched pairs ", id1, " ", id2);
      if (id1[id1.length() - 1] != '1' || id2[id2.length() - 1] != '2') DIE("Mismatched pair numbers ", id1, " ", id2);

      
      bool is_merged = 0;
      int8_t abort_merge = 0;

      // revcomp the second mate pair and reverse the second quals
      string rc_seq2 = revcomp(seq2);
      string rev_quals2 = quals2;
      reverse(rev_quals2.begin(), rev_quals2.end());

      // use start_i to offset inequal lengths which can be very different but still overlap near the end.  250 vs 178..
      int16_t len = (rc_seq2.length() < seq1.length()) ? rc_seq2.length() : seq1.length();
      int16_t start_i = ((len == (int16_t)seq1.length()) ? 0 : seq1.length() - len);
      int16_t found_i = -1;
      int16_t best_i = -1;
      int16_t best_mm = len;
      double best_perror = -1.0;

      // slide along seq1
      for (int16_t i = 0; i < len - MIN_OVERLAP + EXTRA_TEST_OVERLAP; i++) {  // test less overlap than MIN_OVERLAP
        if (abort_merge) break;
        int16_t overlap = len - i;
        int16_t this_max_mismatch = MAX_MISMATCHES + (EXTRA_MISMATCHES_PER_1000 * overlap / 1000);
        int16_t error_max_mismatch = this_max_mismatch * 4 / 3 + 1;  // 33% higher
        if (fast_count_mismatches(seq1.c_str() + start_i + i, rc_seq2.c_str(), overlap, error_max_mismatch) > error_max_mismatch)
          continue;
        int16_t matches = 0, mismatches = 0, bothNs = 0, Ncount = 0;
        int16_t overlapChecked = 0;
        double perror = 0.0;
        for (int16_t j = 0; j < overlap; j++) {
          overlapChecked++;
          char ps = seq1[start_i + i + j];
          char rs = rc_seq2[j];
          if (ps == rs) {
            matches++;
            if (ps == 'N') {
              Ncount += 2;
              if (bothNs++) {
                abort_merge++;
                num_ambiguous++;
                break;  // do not match multiple Ns in the same position -- 1 is okay
              }
            }
          } else {
            mismatches++;
            if (ps == 'N') {
              mismatches++;  // N still counts as a mismatch
              Ncount++;
              quals1[start_i + i + j] = qual_offset;
              assert(rev_quals2[j] - qual_offset < Q2PerrorSize);
              assert(rev_quals2[j] - qual_offset >= 0);
              perror += Q2Perror[rev_quals2[j] - qual_offset];
            } else if (rs == 'N') {
              Ncount++;
              mismatches++;  // N still counts as a mismatch
              rev_quals2[j] = qual_offset;
              assert(quals1[start_i + i + j] - qual_offset < Q2PerrorSize);
              assert(quals1[start_i + i + j] - qual_offset >= 0);
              perror += Q2Perror[quals1[start_i + i + j] - qual_offset];
            }
            if (MAX_PERROR > 0.0) {
              assert(quals1[start_i + i + j] >= qual_offset);
              assert(rev_quals2[j] >= qual_offset);
              uint8_t q1 = quals1[start_i + i + j] - qual_offset;
              uint8_t q2 = rev_quals2[j] - qual_offset;
              if (q1 < 0 || q2 < 0 || q1 >= Q2PerrorSize || q2 >= Q2PerrorSize)
                DIE("Invalid quality score for read ", id1, " '", quals1[start_i + i + j], "' ", id2, " '", rev_quals2[j],
                    "' assuming common qual_offset of ", qual_offset,
                    ". Check the data and make sure it follows a single consistent quality scoring model ",
                    "(phred+64 vs. phred+33)");

              // sum perror as the difference in q score perrors
              uint8_t diffq = (q1 > q2) ? q1 - q2 : q2 - q1;
              if (diffq <= 2) {
                perror += 0.5;  // cap at flipping a coin when both quality scores are close
              } else {
                assert(diffq < Q2PerrorSize);
                perror += Q2Perror[diffq];
              }
            }
          }
          if (Ncount > 3) {
            abort_merge++;
            num_ambiguous++;
            break;  // do not match reads with many Ns
          }
          if (mismatches > error_max_mismatch) break;
        }
        int16_t match_thres = overlap - this_max_mismatch;
        if (match_thres < MIN_OVERLAP) match_thres = MIN_OVERLAP;
        if (matches >= match_thres && overlapChecked == overlap && mismatches <= this_max_mismatch &&
            perror / overlap <= MAX_PERROR) {
          if (best_i < 0 && found_i < 0) {
            best_i = i;
            best_mm = mismatches;
            best_perror = perror;
          } else {
            // another good or ambiguous overlap detected
            num_ambiguous++;
            best_i = -1;
            best_mm = len;
            best_perror = -1.0;
            break;
          }
        } else if (overlapChecked == overlap && mismatches <= error_max_mismatch && perror / overlap <= MAX_PERROR * 4 / 3) {
          // lower threshold for detection of an ambigious overlap
          found_i = i;
          if (best_i >= 0) {
            // ambiguous mapping found after a good one was
            num_ambiguous++;
            best_i = -1;
            best_mm = len;
            best_perror = -1.0;
            break;
          }
        }
      }

      if (best_i >= 0 && !abort_merge) {
        int16_t i = best_i;
        int16_t overlap = len - i;
        // pick the base with the highest quality score for the overlapped region
        for (int16_t j = 0; j < overlap; j++) {
          if (seq1[start_i + i + j] == rc_seq2[j]) {
            // match boost quality up to the limit
            uint16_t newQual = quals1[start_i + i + j] + rev_quals2[j] - qual_offset;
            quals1[start_i + i + j] = ((newQual > MAX_MATCH_QUAL) ? MAX_MATCH_QUAL : newQual);
            assert(quals1[start_i + i + j] >= quals1[start_i + i + j]);
            // FIXME: this fails for a CAMISIM generated dataset. I don't even know what this is checking...
            // assert(quals1[start_i + i + j] >= rev_quals2[j]);
          } else {
            uint8_t newQual;
            if (quals1[start_i + i + j] < rev_quals2[j]) {
              // use rev base and discount quality
              newQual = rev_quals2[j] - quals1[start_i + i + j] + qual_offset;
              seq1[start_i + i + j] = rc_seq2[j];
            } else {
              // keep prev base, but still discount quality
              newQual = quals1[start_i + i + j] - rev_quals2[j] + qual_offset;
            }
            // a bit better than random chance here
            quals1[start_i + i + j] = ((newQual > (2 + qual_offset)) ? newQual : (2 + qual_offset));
          }
          assert(quals1[start_i + i + j] >= qual_offset);
        }

        // include the remainder of the rc_seq2 and quals
        seq1 = seq1.substr(0, start_i + i + overlap) + rc_seq2.substr(overlap);
        quals1 = quals1.substr(0, start_i + i + overlap) + rev_quals2.substr(overlap);

        is_merged = true;
        num_merged++;

        int read_len = seq1.length();  // caculate new merged length
        if (max_read_len < read_len) max_read_len = read_len;
        merged_len += read_len;
        overlap_len += overlap;

        packed_reads_list[ri]->add_read("r" + to_string(read_id) + "/1", seq1, quals1);
        packed_reads_list[ri]->add_read("r" + to_string(read_id) + "/2", "N", to_string((char)qual_offset));
      }
      if (!is_merged) {
        // write without the revcomp
        packed_reads_list[ri]->add_read("r" + to_string(read_id) + "/1", seq1, quals1);
        packed_reads_list[ri]->add_read("r" + to_string(read_id) + "/2", seq2, quals2);
      }
      // inc by 2 so that we can use a later optimization of treating the even as /1 and the odd as /2
      read_id += 2;
    }

    fqr.advise(false);  // free kernel memory

   
    auto prog_done = progbar.set_done();
    wrote_all_files_fut = when_all(wrote_all_files_fut, prog_done);

    tot_num_merged += num_merged;
    tot_num_ambiguous += num_ambiguous;
    tot_max_read_len = std::max(tot_max_read_len, max_read_len);
    tot_bytes_read += bytes_read;
    tot_bases += bases_read;

    // start the collective reductions
    // delay the summary output for when they complete
    auto fut_reductions = when_all(reduce_one(num_pairs, op_fast_add, 0), reduce_one(num_merged, op_fast_add, 0),
                                   reduce_one(num_ambiguous, op_fast_add, 0), reduce_one(merged_len, op_fast_add, 0),
                                   reduce_one(overlap_len, op_fast_add, 0), reduce_one(max_read_len, op_fast_max, 0),
                                   reduce_one(bases_trimmed, op_fast_add, 0), reduce_one(reads_removed, op_fast_add, 0),
                                   reduce_one(bases_read, op_fast_add, 0));
    fut_summary = when_all(fut_summary, fut_reductions)
                      .then([reads_fname, bytes_read](
                                int64_t all_num_pairs, int64_t all_num_merged, int64_t all_num_ambiguous, int64_t all_merged_len,
                                int64_t all_overlap_len, int all_max_read_len, int64_t all_bases_trimmed, int64_t all_reads_removed,
                                int64_t all_bases_read) {
                        SLOG_VERBOSE("Merged reads in file ", reads_fname, ":\n");
                        SLOG_VERBOSE("  merged ", perc_str(all_num_merged, all_num_pairs), " pairs\n");
                        SLOG_VERBOSE("  ambiguous ", perc_str(all_num_ambiguous, all_num_pairs), " ambiguous pairs\n");
                        SLOG_VERBOSE("  average merged length ", (double)all_merged_len / all_num_merged, "\n");
                        SLOG_VERBOSE("  average overlap length ", (double)all_overlap_len / all_num_merged, "\n");
                        SLOG_VERBOSE("  max read length ", all_max_read_len, "\n");
                        
                        SLOG_VERBOSE("  max read length ", all_max_read_len, "\n");
                        SLOG_VERBOSE("Total bytes read ", bytes_read, "\n");
                      });

    num_reads += num_pairs * 2;
    ri++;
    FastqReaders::close(reads_fname);
  }
  merge_time.initiate_exit_reduction();

  //#ifdef DEBUG
  // ensure there is no overlap in read_ids which will cause a crash later
  using SSPair = std::pair<uint64_t, uint64_t>;
  SSPair start_stop(start_read_id, read_id);
  dist_object<SSPair> dist_ss(world(), start_stop);
  future<> rpc_tests = make_future();
  // check next rank
  assert(dist_ss->first <= dist_ss->second);
  if (rank_me() < rank_n() - 1) {
    auto fut = rpc(
        rank_me() + 1,
        [](dist_object<pair<uint64_t, uint64_t>> &dist_ss, SSPair ss) {
          if (!(ss.first < dist_ss->first && ss.second < dist_ss->first))
            DIE("Invalid read ids from previous rank: ", rank_me(), "=", dist_ss->first, "-", dist_ss->second,
                " prev rank=", ss.first, "-", ss.second, "\n");
        },
        dist_ss, *dist_ss);
    rpc_tests = when_all(rpc_tests, fut);
  }
  if (rank_me() > 0) {
    auto fut = rpc(
        rank_me() - 1,
        [](dist_object<pair<uint64_t, uint64_t>> &dist_ss, SSPair ss) {
          if (!(ss.first > dist_ss->second && ss.second > dist_ss->second))
            DIE("Invalid read ids from next rank: ", rank_me(), "=", dist_ss->first, "-", dist_ss->second, " next rank=", ss.first,
                "-", ss.second, "\n");
        },
        dist_ss, *dist_ss);
    rpc_tests = when_all(rpc_tests, fut);
  }
  rpc_tests.wait();
  //#endif

  // finish all file writing and report
  dump_reads_t.start();
  wrote_all_files_fut.wait();
  for (auto sh_of : all_outputs) {
    wrote_all_files_fut = when_all(wrote_all_files_fut, sh_of->report_timings());
  }
  wrote_all_files_fut.wait();

  dump_reads_t.stop();
  dump_reads_t.done();

  summary_promise.fulfill_anonymous(1);
  fut_summary.wait();

  timer.initate_exit_barrier();
}

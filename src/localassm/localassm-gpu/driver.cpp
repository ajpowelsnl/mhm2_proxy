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

#include <numeric>
#include <cassert>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "driver.hpp"
#include "kernel.hpp"
#include "gpu-utils/gpu_utils.hpp"
#include "gpu-utils/gpu_common.hpp"

using namespace std;

static void revcomp(char *str, char *str_rc, int size) {
  int size_rc = 0;
  for (int i = size - 1; i >= 0; i--) {
    switch (str[i]) {
      case 'A': str_rc[size_rc] = 'T'; break;
      case 'C': str_rc[size_rc] = 'G'; break;
      case 'G': str_rc[size_rc] = 'C'; break;
      case 'T': str_rc[size_rc] = 'A'; break;
      case 'N': str_rc[size_rc] = 'N'; break;
      case 'U':
      case 'R':
      case 'Y':
      case 'K':
      case 'M':
      case 'S':
      case 'W':
      case 'B':
      case 'D':
      case 'H':
      case 'V': str_rc[size_rc] = 'N'; break;
      default: std::cout << "Illegal char:" << str[i] << "\n"; break;
    }
    size_rc++;
  }
}

static std::string revcomp(std::string instr) {
  std::string str_rc;
  for (int i = instr.size() - 1; i >= 0; i--) {
    switch (instr[i]) {
      case 'A': str_rc += 'T'; break;
      case 'C': str_rc += 'G'; break;
      case 'G': str_rc += 'C'; break;
      case 'T': str_rc += 'A'; break;
      case 'N': str_rc += 'N'; break;
      case 'U':
      case 'R':
      case 'Y':
      case 'K':
      case 'M':
      case 'S':
      case 'W':
      case 'B':
      case 'D':
      case 'H':
      case 'V': str_rc += 'N'; break;
      default: std::cout << "Illegal char:" << instr[i] << "\n"; break;
    }
  }
  return str_rc;
}

void localassm_driver::localassm_driver(vector<CtgWithReads> &data_in, uint32_t max_ctg_size, uint32_t max_read_size,
                                        uint32_t max_r_count, uint32_t max_l_count, int mer_len, int max_kmer_len,
                                        accum_data &sizes_vecs, int walk_len_limit, int qual_offset, int my_rank,
                                        size_t gpu_mem_avail) {
  gpu_utils::set_gpu_device(my_rank);
  int max_mer_len = max_kmer_len;  // mer_len;// max_mer_len needs to come from macro (121) and mer_len is the mer_len for current
                                   // go
  unsigned tot_extensions = data_in.size();
  uint32_t max_read_count = max_r_count > max_l_count ? max_r_count : max_l_count;
  int max_walk_len = walk_len_limit;
  uint64_t ht_tot_size = accumulate(sizes_vecs.ht_sizes.begin(), sizes_vecs.ht_sizes.end(), 0);
  uint64_t total_r_reads = accumulate(sizes_vecs.r_reads_count.begin(), sizes_vecs.r_reads_count.end(), 0);
  uint64_t total_l_reads = accumulate(sizes_vecs.l_reads_count.begin(), sizes_vecs.l_reads_count.end(), 0);
  uint64_t total_ctg_len = accumulate(sizes_vecs.ctg_sizes.begin(), sizes_vecs.ctg_sizes.end(), 0);

  size_t gpu_mem_req = sizeof(int32_t) * tot_extensions * 6 + sizeof(int32_t) * total_l_reads + sizeof(int32_t) * total_r_reads +
                       sizeof(char) * total_ctg_len + sizeof(char) * total_l_reads * max_read_size * 2 +
                       sizeof(char) * total_r_reads * max_read_size * 2  // for quals included
                       + sizeof(double) * tot_extensions + sizeof(char) * total_r_reads * max_read_size +
                       sizeof(char) * total_l_reads * max_read_size + sizeof(int64_t) * 3 +
                       sizeof(loc_ht) * ht_tot_size  // changed to try the new method
                       + sizeof(char) * tot_extensions * max_walk_len +
                       (max_mer_len + max_walk_len) * sizeof(char) * tot_extensions +
                       sizeof(loc_ht_bool) * tot_extensions * max_walk_len;

  assert(gpu_mem_avail > 0);
  // factor is to buffer for the extra mem that is used when allocating once and using again
  float factor = 0.80;
  unsigned iterations = ceil(((double)gpu_mem_req) / ((double)gpu_mem_avail * factor));
  assert(iterations > 0);
  unsigned slice_size = tot_extensions / iterations;
  assert(slice_size > 0);
  unsigned remaining = tot_extensions % iterations;
  vector<uint32_t> max_ht_sizes;
  // to get the largest ht size for any iteration and allocate GPU memory for that (once)
  uint64_t max_ht = 0, max_r_rds_its = 0, max_l_rds_its = 0, max_ctg_len_it = 0, test_sum = 0;
  for (unsigned i = 0; i < iterations; i++) {
    uint64_t temp_max_ht = 0, temp_max_r_rds = 0, temp_max_l_rds = 0, temp_max_ctg_len = 0;
    if (i < iterations - 1) {
      temp_max_ht = accumulate(sizes_vecs.ht_sizes.begin() + i * slice_size, sizes_vecs.ht_sizes.begin() + (i + 1) * slice_size, 0);
      temp_max_r_rds =
          accumulate(sizes_vecs.r_reads_count.begin() + i * slice_size, sizes_vecs.r_reads_count.begin() + (i + 1) * slice_size, 0);
      temp_max_l_rds =
          accumulate(sizes_vecs.l_reads_count.begin() + i * slice_size, sizes_vecs.l_reads_count.begin() + (i + 1) * slice_size, 0);
      temp_max_ctg_len =
          accumulate(sizes_vecs.ctg_sizes.begin() + i * slice_size, sizes_vecs.ctg_sizes.begin() + (i + 1) * slice_size, 0);
    } else {
      temp_max_ht = accumulate(sizes_vecs.ht_sizes.begin() + i * slice_size,
                               sizes_vecs.ht_sizes.begin() + ((i + 1) * slice_size) + remaining, 0);
      temp_max_r_rds = accumulate(sizes_vecs.r_reads_count.begin() + i * slice_size,
                                  sizes_vecs.r_reads_count.begin() + ((i + 1) * slice_size) + remaining, 0);
      temp_max_l_rds = accumulate(sizes_vecs.l_reads_count.begin() + i * slice_size,
                                  sizes_vecs.l_reads_count.begin() + ((i + 1) * slice_size) + remaining, 0);
      temp_max_ctg_len = accumulate(sizes_vecs.ctg_sizes.begin() + i * slice_size,
                                    sizes_vecs.ctg_sizes.begin() + ((i + 1) * slice_size) + remaining, 0);
    }
    if (temp_max_ht > max_ht) max_ht = temp_max_ht;
    if (temp_max_r_rds > max_r_rds_its) max_r_rds_its = temp_max_r_rds;
    if (temp_max_l_rds > max_l_rds_its) max_l_rds_its = temp_max_l_rds;
    if (temp_max_ctg_len > max_ctg_len_it) max_ctg_len_it = temp_max_ctg_len;
    test_sum += temp_max_ht;
  }
  slice_size = slice_size + remaining;  // this is the largest slice size, mostly the last iteration handles the leftovers
  // allocating maximum possible memory for a single iteration

  unique_ptr<char[]> ctg_seqs_h{new char[max_ctg_size * slice_size]};
  unique_ptr<uint64_t[]> cid_h{new uint64_t[slice_size]};
  unique_ptr<char[]> ctgs_seqs_rc_h{
      new char[max_ctg_size * slice_size]};  // revcomps not requried on GPU, ctg space will be re-used on GPU, but if we want to do
                                             // right left extensions in parallel, then we need separate space on GPU
  unique_ptr<uint32_t[]> ctg_seq_offsets_h{new uint32_t[slice_size]};
  unique_ptr<double[]> depth_h{new double[slice_size]};
  unique_ptr<char[]> reads_left_h{new char[max_l_count * max_read_size * slice_size]};
  unique_ptr<char[]> reads_right_h{new char[max_r_count * max_read_size * slice_size]};
  unique_ptr<char[]> quals_right_h{new char[max_r_count * max_read_size * slice_size]};
  unique_ptr<char[]> quals_left_h{new char[max_l_count * max_read_size * slice_size]};
  unique_ptr<uint32_t[]> reads_l_offset_h{new uint32_t[max_l_count * slice_size]};
  unique_ptr<uint32_t[]> reads_r_offset_h{new uint32_t[max_r_count * slice_size]};
  unique_ptr<uint32_t[]> rds_l_cnt_offset_h{new uint32_t[slice_size]};
  unique_ptr<uint32_t[]> rds_r_cnt_offset_h{new uint32_t[slice_size]};
  unique_ptr<uint32_t[]> term_counts_h{new uint32_t[3]};
  unique_ptr<char[]> longest_walks_r_h{new char[slice_size * max_walk_len * iterations]};  // reserve memory for all the walks
  unique_ptr<char[]> longest_walks_l_h{
      new char[slice_size * max_walk_len * iterations]};  // not needed on device, will re-use right walk memory
  unique_ptr<uint32_t[]> final_walk_lens_r_h{new uint32_t[slice_size * iterations]};  // reserve memory for all the walks.
  unique_ptr<uint32_t[]> final_walk_lens_l_h{
      new uint32_t[slice_size * iterations]};  // not needed on device, will re use right walk memory
  unique_ptr<uint32_t[]> prefix_ht_size_h{new uint32_t[slice_size]};

  gpu_mem_req = sizeof(int32_t) * slice_size * 6 + sizeof(int32_t) * 3 + sizeof(int32_t) * max_l_rds_its +
                sizeof(int32_t) * max_r_rds_its + sizeof(char) * max_ctg_len_it + sizeof(char) * max_l_rds_its * max_read_size * 2 +
                sizeof(char) * max_r_rds_its * max_read_size * 2 + sizeof(double) * slice_size + sizeof(loc_ht) * max_ht +
                sizeof(char) * slice_size * max_walk_len + (max_mer_len + max_walk_len) * sizeof(char) * slice_size +
                sizeof(loc_ht_bool) * slice_size * max_walk_len;

  uint32_t *ctg_seq_offsets_d, *reads_l_offset_d, *reads_r_offset_d;
  uint64_t *cid_d;
  uint32_t *rds_l_cnt_offset_d, *rds_r_cnt_offset_d, *prefix_ht_size_d;
  char *ctg_seqs_d, *reads_left_d, *reads_right_d, *quals_left_d, *quals_right_d;
  char *longest_walks_d, *mer_walk_temp_d;
  double *depth_d;
  uint32_t *term_counts_d;
  loc_ht *d_ht;
  loc_ht_bool *d_ht_bool;
  uint32_t *final_walk_lens_d;
  // allocate GPU  memory
  cudaErrchk(cudaMalloc(&prefix_ht_size_d, sizeof(uint32_t) * slice_size));
  cudaErrchk(cudaMalloc(&cid_d, sizeof(uint64_t) * slice_size));
  cudaErrchk(cudaMalloc(&ctg_seq_offsets_d, sizeof(uint32_t) * slice_size));
  cudaErrchk(cudaMalloc(&reads_l_offset_d, sizeof(uint32_t) * max_l_rds_its));
  cudaErrchk(cudaMalloc(&reads_r_offset_d, sizeof(uint32_t) * max_r_rds_its));
  cudaErrchk(cudaMalloc(&rds_l_cnt_offset_d, sizeof(uint32_t) * slice_size));
  cudaErrchk(cudaMalloc(&rds_r_cnt_offset_d, sizeof(uint32_t) * slice_size));
  cudaErrchk(cudaMalloc(&ctg_seqs_d, sizeof(char) * max_ctg_len_it));
  cudaErrchk(cudaMalloc(&reads_left_d, sizeof(char) * max_read_size * max_l_rds_its));
  cudaErrchk(cudaMalloc(&reads_right_d, sizeof(char) * max_read_size * max_r_rds_its));
  cudaErrchk(cudaMalloc(&depth_d, sizeof(double) * slice_size));
  cudaErrchk(cudaMalloc(&quals_right_d, sizeof(char) * max_read_size * max_r_rds_its));
  cudaErrchk(cudaMalloc(&quals_left_d, sizeof(char) * max_read_size * max_l_rds_its));
  cudaErrchk(cudaMalloc(&term_counts_d, sizeof(uint32_t) * 3));
  // one local hashtable for each thread, so total hash_tables equal to vec_size i.e. total contigs
  cudaErrchk(cudaMalloc(&d_ht, sizeof(loc_ht) * max_ht));
  cudaErrchk(cudaMalloc(&longest_walks_d, sizeof(char) * slice_size * max_walk_len));
  cudaErrchk(cudaMalloc(&mer_walk_temp_d, (max_mer_len + max_walk_len) * sizeof(char) * slice_size));
  cudaErrchk(cudaMalloc(&d_ht_bool, sizeof(loc_ht_bool) * slice_size * max_walk_len));
  cudaErrchk(cudaMalloc(&final_walk_lens_d, sizeof(uint32_t) * slice_size));

  slice_size = tot_extensions / iterations;

  for (unsigned slice = 0; slice < iterations; slice++) {
    uint32_t left_over;
    if (iterations - 1 == slice)
      left_over = tot_extensions % iterations;
    else
      left_over = 0;
    assert(slice_size > 0);

    vector<CtgWithReads>::const_iterator slice_iter = data_in.begin() + slice * slice_size;
    auto this_slice_size = slice_size + left_over;
    uint32_t vec_size = this_slice_size;  // slice_data.size();
    uint32_t ctgs_offset_sum = 0;
    uint32_t prefix_ht_sum = 0;
    uint32_t reads_r_offset_sum = 0;
    uint32_t reads_l_offset_sum = 0;
    uint32_t read_l_index = 0, read_r_index = 0;
    for (unsigned i = 0; i < this_slice_size; i++) {
      CtgWithReads temp_data = slice_iter[i];  // slice_data[i];
      cid_h[i] = temp_data.cid;
      depth_h[i] = temp_data.depth;
      // convert string to c-string
      char *ctgs_ptr = ctg_seqs_h.get() + ctgs_offset_sum;
      memcpy(ctgs_ptr, temp_data.seq.c_str(), temp_data.seq.size());
      ctgs_offset_sum += temp_data.seq.size();
      ctg_seq_offsets_h[i] = ctgs_offset_sum;
      prefix_ht_sum += temp_data.max_reads * max_read_size;
      prefix_ht_size_h[i] = prefix_ht_sum;

      for (unsigned j = 0; j < temp_data.reads_left.size(); j++) {
        char *reads_l_ptr = reads_left_h.get() + reads_l_offset_sum;
        char *quals_l_ptr = quals_left_h.get() + reads_l_offset_sum;
        memcpy(reads_l_ptr, temp_data.reads_left[j].seq.c_str(), temp_data.reads_left[j].seq.size());
        // quals offsets will be same as reads offset because quals and reads have same length
        memcpy(quals_l_ptr, temp_data.reads_left[j].quals.c_str(), temp_data.reads_left[j].quals.size());
        reads_l_offset_sum += temp_data.reads_left[j].seq.size();
        reads_l_offset_h[read_l_index] = reads_l_offset_sum;
        read_l_index++;
      }
      rds_l_cnt_offset_h[i] = read_l_index;  // running sum of left reads count

      for (unsigned j = 0; j < temp_data.reads_right.size(); j++) {
        char *reads_r_ptr = reads_right_h.get() + reads_r_offset_sum;
        char *quals_r_ptr = quals_right_h.get() + reads_r_offset_sum;
        memcpy(reads_r_ptr, temp_data.reads_right[j].seq.c_str(), temp_data.reads_right[j].seq.size());
        // quals offsets will be same as reads offset because quals and reads have same length
        memcpy(quals_r_ptr, temp_data.reads_right[j].quals.c_str(), temp_data.reads_right[j].quals.size());
        reads_r_offset_sum += temp_data.reads_right[j].seq.size();
        reads_r_offset_h[read_r_index] = reads_r_offset_sum;
        read_r_index++;
      }
      rds_r_cnt_offset_h[i] = read_r_index;  // running sum of right reads count
    }                                        // data packing for loop ends

    uint32_t total_r_reads_slice = read_r_index;
    uint32_t total_l_reads_slice = read_l_index;
    for (int i = 0; i < 3; i++) {
      term_counts_h[i] = 0;
    }

    cudaErrchk(cudaMemcpy(prefix_ht_size_d, prefix_ht_size_h.get(), sizeof(uint32_t) * vec_size, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(cid_d, cid_h.get(), sizeof(uint64_t) * vec_size, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(ctg_seq_offsets_d, ctg_seq_offsets_h.get(), sizeof(uint32_t) * vec_size, cudaMemcpyHostToDevice));
    cudaErrchk(
        cudaMemcpy(reads_l_offset_d, reads_l_offset_h.get(), sizeof(uint32_t) * total_l_reads_slice, cudaMemcpyHostToDevice));
    cudaErrchk(
        cudaMemcpy(reads_r_offset_d, reads_r_offset_h.get(), sizeof(uint32_t) * total_r_reads_slice, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(rds_l_cnt_offset_d, rds_l_cnt_offset_h.get(), sizeof(uint32_t) * vec_size, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(rds_r_cnt_offset_d, rds_r_cnt_offset_h.get(), sizeof(uint32_t) * vec_size, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(ctg_seqs_d, ctg_seqs_h.get(), sizeof(char) * ctgs_offset_sum, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(reads_left_d, reads_left_h.get(), sizeof(char) * reads_l_offset_sum, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(reads_right_d, reads_right_h.get(), sizeof(char) * reads_r_offset_sum, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(depth_d, depth_h.get(), sizeof(double) * vec_size, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(quals_right_d, quals_right_h.get(), sizeof(char) * reads_r_offset_sum, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(quals_left_d, quals_left_h.get(), sizeof(char) * reads_l_offset_sum, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(term_counts_d, term_counts_h.get(), sizeof(uint32_t) * 3, cudaMemcpyHostToDevice));
    // call kernel here, one thread per contig
    unsigned total_threads = vec_size * 32;  // we need one warp (32 threads) per extension, vec_size = extensions
    unsigned thread_per_blk = 512;
    unsigned blocks = (total_threads + thread_per_blk) / thread_per_blk;

    int64_t sum_ext = 0, num_walks = 0;
    uint32_t qual_offset_ = qual_offset;
    iterative_walks_kernel<<<blocks, thread_per_blk>>>(
        cid_d, ctg_seq_offsets_d, ctg_seqs_d, reads_right_d, quals_right_d, reads_r_offset_d, rds_r_cnt_offset_d, depth_d, d_ht,
        prefix_ht_size_d, d_ht_bool, mer_len, max_mer_len, term_counts_d, num_walks, max_walk_len, sum_ext, max_read_size,
        max_read_count, qual_offset_, longest_walks_d, mer_walk_temp_d, final_walk_lens_d, vec_size);

    // perform revcomp of contig sequences and launch kernel with left reads,

    for (unsigned j = 0; j < vec_size; j++) {
      int size_lst;
      char *curr_seq;
      char *curr_seq_rc;
      if (j == 0) {
        size_lst = ctg_seq_offsets_h[j];
        curr_seq = ctg_seqs_h.get();
        curr_seq_rc = ctgs_seqs_rc_h.get();
      } else {
        size_lst = ctg_seq_offsets_h[j] - ctg_seq_offsets_h[j - 1];
        curr_seq = ctg_seqs_h.get() + ctg_seq_offsets_h[j - 1];
        curr_seq_rc = ctgs_seqs_rc_h.get() + ctg_seq_offsets_h[j - 1];
      }
      revcomp(curr_seq, curr_seq_rc, size_lst);
    }
    cudaErrchk(cudaMemcpy(longest_walks_r_h.get() + slice * max_walk_len * slice_size, longest_walks_d,
                          sizeof(char) * vec_size * max_walk_len, cudaMemcpyDeviceToHost));
    cudaErrchk(cudaMemcpy(final_walk_lens_r_h.get() + slice * slice_size, final_walk_lens_d, sizeof(int32_t) * vec_size,
                          cudaMemcpyDeviceToHost));

    // cpying rev comped ctgs to device on same memory as previous ctgs
    cudaErrchk(cudaMemcpy(ctg_seqs_d, ctgs_seqs_rc_h.get(), sizeof(char) * ctgs_offset_sum, cudaMemcpyHostToDevice));

    iterative_walks_kernel<<<blocks, thread_per_blk>>>(
        cid_d, ctg_seq_offsets_d, ctg_seqs_d, reads_left_d, quals_left_d, reads_l_offset_d, rds_l_cnt_offset_d, depth_d, d_ht,
        prefix_ht_size_d, d_ht_bool, mer_len, max_mer_len, term_counts_d, num_walks, max_walk_len, sum_ext, max_read_size,
        max_read_count, qual_offset_, longest_walks_d, mer_walk_temp_d, final_walk_lens_d, vec_size);

    cudaErrchk(cudaMemcpy(longest_walks_l_h.get() + slice * max_walk_len * slice_size, longest_walks_d,
                          sizeof(char) * vec_size * max_walk_len, cudaMemcpyDeviceToHost));  // copy back left walks
    cudaErrchk(cudaMemcpy(final_walk_lens_l_h.get() + slice * slice_size, final_walk_lens_d, sizeof(int32_t) * vec_size,
                          cudaMemcpyDeviceToHost));
  }  // the for loop over all slices ends here

  // once all the alignments are on cpu, then go through them and stitch them with contigs in front and back.
  int loc_left_over = tot_extensions % iterations;
  for (unsigned j = 0; j < iterations; j++) {
    int loc_size = (j == iterations - 1) ? slice_size + loc_left_over : slice_size;

    for (int i = 0; i < loc_size; i++) {
      if (final_walk_lens_l_h[j * slice_size + i] > 0) {
        string left(longest_walks_l_h.get() + j * slice_size * max_walk_len + max_walk_len * i,
                    final_walk_lens_l_h[j * slice_size + i]);
        string left_rc = revcomp(left);
        data_in[j * slice_size + i].seq.insert(0, left_rc);
      }
      if (final_walk_lens_r_h[j * slice_size + i] > 0) {
        string right(longest_walks_r_h.get() + j * slice_size * max_walk_len + max_walk_len * i,
                     final_walk_lens_r_h[j * slice_size + i]);
        data_in[j * slice_size + i].seq += right;
      }
    }
  }

  cudaErrchk(cudaFree(prefix_ht_size_d));
  cudaErrchk(cudaFree(term_counts_d));
  cudaErrchk(cudaFree(cid_d));
  cudaErrchk(cudaFree(ctg_seq_offsets_d));
  cudaErrchk(cudaFree(reads_l_offset_d));
  cudaErrchk(cudaFree(reads_r_offset_d));
  cudaErrchk(cudaFree(rds_l_cnt_offset_d));
  cudaErrchk(cudaFree(rds_r_cnt_offset_d));
  cudaErrchk(cudaFree(ctg_seqs_d));
  cudaErrchk(cudaFree(reads_left_d));
  cudaErrchk(cudaFree(reads_right_d));
  cudaErrchk(cudaFree(depth_d));
  cudaErrchk(cudaFree(quals_right_d));
  cudaErrchk(cudaFree(quals_left_d));
  cudaErrchk(cudaFree(d_ht));
  cudaErrchk(cudaFree(longest_walks_d));
  cudaErrchk(cudaFree(mer_walk_temp_d));
  cudaErrchk(cudaFree(d_ht_bool));
  cudaErrchk(cudaFree(final_walk_lens_d));
}

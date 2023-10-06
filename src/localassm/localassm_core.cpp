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

#include <fstream>
#include <iostream>
#include <regex>
#include <upcxx/upcxx.hpp>
#include <memory>

#include "alignments.hpp"
#include "contigs.hpp"
#include "packed_reads.hpp"
#include "upcxx_utils.hpp"
#include "utils.hpp"

#include "localassm_core.hpp"

using namespace std;
using namespace upcxx;
using namespace upcxx_utils;

namespace localassm_core {

enum class AlnStatus { NO_ALN, OVERLAPS_CONTIG, EXTENDS_CONTIG };

using ext_count_t = uint16_t;

struct ExtCounts {
  ext_count_t count_A;
  ext_count_t count_C;
  ext_count_t count_G;
  ext_count_t count_T;

  ext_count_t inc_with_limit(int count1, int count2) {
    count1 += count2;
    return std::min(count1, (int)std::numeric_limits<ext_count_t>::max());
  }

  void inc(char ext, int count) {
    switch (ext) {
      case 'A': count_A = inc_with_limit(count_A, count); break;
      case 'C': count_C = inc_with_limit(count_C, count); break;
      case 'G': count_G = inc_with_limit(count_G, count); break;
      case 'T': count_T = inc_with_limit(count_T, count); break;
    }
  }
};

size_t ReadsToCtgsDHT::get_target_rank(const string &read_id) { return std::hash<string>{}(read_id) % rank_n(); }

ReadsToCtgsDHT::ReadsToCtgsDHT(int64_t initial_size)
    : reads_to_ctgs_map({})
    , rtc_store() {
  reads_to_ctgs_map->reserve(initial_size);
  rtc_store.set_update_func([&reads_to_ctgs_map = this->reads_to_ctgs_map](ReadCtgInfo &&read_ctg_info) {
    const auto it = reads_to_ctgs_map->find(read_ctg_info.read_id);
    if (it == reads_to_ctgs_map->end())
      reads_to_ctgs_map->insert({std::move(read_ctg_info.read_id), {std::move(read_ctg_info.ctg_info)}});
    else
      it->second.push_back(std::move(read_ctg_info.ctg_info));
  });
  int est_update_size = sizeof(ReadCtgInfo) + 13;  // read_id
  int64_t mem_to_use = 0.05 * get_free_mem() / local_team().rank_n();
  auto max_store_bytes = max(mem_to_use, (int64_t)est_update_size * 100);
  rtc_store.set_size("ReadsToContigs", max_store_bytes);
}

void ReadsToCtgsDHT::clear() {
  reads_to_ctgs_map_t().swap(*reads_to_ctgs_map);
  rtc_store.clear();
}

void ReadsToCtgsDHT::add(const string &read_id, int64_t cid, char orient, char side) {
  ReadCtgInfo read_ctg_info;
  read_ctg_info.read_id = read_id;
  read_ctg_info.ctg_info.cid = cid;
  read_ctg_info.ctg_info.orient = orient;
  read_ctg_info.ctg_info.side = side;
  rtc_store.update(get_target_rank(read_id), read_ctg_info);
}

void ReadsToCtgsDHT::flush_updates() {
  rtc_store.flush_updates();
  rtc_store.clear();
}

int64_t ReadsToCtgsDHT::get_num_mappings() { return reduce_one(reads_to_ctgs_map->size(), op_fast_add, 0).wait(); }

vector<CtgInfo> ReadsToCtgsDHT::get_ctgs(string &read_id) {
  return upcxx::rpc(
             get_target_rank(read_id),
             [](upcxx::dist_object<reads_to_ctgs_map_t> &reads_to_ctgs_map, string read_id) -> vector<CtgInfo> {
               const auto it = reads_to_ctgs_map->find(read_id);
               if (it == reads_to_ctgs_map->end()) return {};
               return it->second;
             },
             reads_to_ctgs_map, read_id)
      .wait();
}

future<vector<vector<CtgInfo>>> ReadsToCtgsDHT::get_ctgs(intrank_t target_rank, vector<string> &read_ids) {
  DBG_VERBOSE("Sending get_ctgs ", read_ids.size(), " to ", target_rank, "\n");
  return upcxx::rpc(
      target_rank,
      [](upcxx::dist_object<reads_to_ctgs_map_t> &reads_to_ctgs_map, intrank_t source_rank,
         view<string> read_ids) -> vector<vector<CtgInfo>> {
        DBG_VERBOSE("Received request for ", read_ids.size(), " reads from ", source_rank, "\n");
        size_t bytes = 0, nonempty = 0;
        vector<vector<CtgInfo>> results(read_ids.size());
        size_t i = 0;
        for (const auto &read_id : read_ids) {
          assert(get_target_rank(read_id) == rank_me());
          const auto it = reads_to_ctgs_map->find(read_id);
          assert(i < results.size());
          assert(results[i].empty());
          if (it != reads_to_ctgs_map->end()) {
            nonempty++;
            bytes += it->second.size() * sizeof(CtgInfo);
            results[i] = it->second;
          }
          i++;
        }
        DBG_VERBOSE("Returning ", results.size(), " results nonempty=", nonempty, " bytes=", bytes, " to ", source_rank, "\n");
        return results;
      },
      reads_to_ctgs_map, rank_me(), make_view(read_ids.begin(), read_ids.end()));
}

size_t CtgsWithReadsDHT::get_target_rank(int64_t cid) { return std::hash<int64_t>{}(cid) % rank_n(); }

CtgsWithReadsDHT::CtgsWithReadsDHT(int64_t num_ctgs)
    : ctgs_map({})
    , ctg_store()
    , ctg_read_store() {
  // pad the local ctg count a bit for this estimate
  ctgs_map->reserve(num_ctgs * 1.2);

  ctg_store.set_update_func([&ctgs_map = this->ctgs_map](CtgData &&ctg_data) {
    auto it = ctgs_map->find(ctg_data.cid);
    if (it != ctgs_map->end()) DIE("Found duplicate ctg ", ctg_data.cid);
    ctgs_map_t::value_type record = {ctg_data.cid,
                                     {.cid = ctg_data.cid,
                                      .seq = std::move(ctg_data.seq),
                                      .depth = ctg_data.depth,
                                      .max_reads = 0,
                                      .reads_left = {},
                                      .reads_right = {}}};
    it = ctgs_map->insert(it, std::move(record));
    DBG_VERBOSE("Added contig cid=", it->first, ": ", it->second.seq, " depth=", it->second.depth, "\n");
  });
  // with contig sequence
  int est_update_size = sizeof(CtgData) + 400;
  // approx 10% of free memory
  int64_t mem_to_use = 0.05 * get_free_mem() / local_team().rank_n();
  auto max_store_bytes = max(mem_to_use, (int64_t)est_update_size * 100);
  ctg_store.set_size("CtgsWithReads add ctg", max_store_bytes);
  ctg_read_store.set_update_func([&ctgs_map = this->ctgs_map](CtgReadData &&ctg_read_data) {
    const auto it = ctgs_map->find(ctg_read_data.cid);
    if (it == ctgs_map->end()) DIE("Could not find ctg ", ctg_read_data.cid);
    ReadSeq &read_seq = ctg_read_data.read_seq;
    if (ctg_read_data.side == 'L')
      it->second.reads_left.push_back(std::move(read_seq));
    else
      it->second.reads_right.push_back(std::move(read_seq));
    DBG_VERBOSE("Added read_seq cid=", ctg_read_data.cid, " read_id=", read_seq.read_id, "\n");
  });
}

void CtgsWithReadsDHT::add_ctg(Contig &ctg) {
  CtgData ctg_data = {ctg.id, ctg.seq, ctg.depth};
  DBG_VERBOSE("updating contig to ", get_target_rank(ctg.id), " cid=", ctg.id, " seq=", ctg_data.seq, " depth=", ctg_data.depth,
              "\n");
  ctg_store.update(get_target_rank(ctg.id), ctg_data);
}

void CtgsWithReadsDHT::add_read(int64_t cid, char side, const ReadSeq read_seq) {
  CtgReadData ctg_read_data(cid, side, read_seq);
  add_read(ctg_read_data);
}
void CtgsWithReadsDHT::add_read(const CtgReadData &ctg_read_data) {
  ctg_read_store.update(get_target_rank(ctg_read_data.cid), ctg_read_data);
}

void CtgsWithReadsDHT::add_reads(vector<CtgReadData> &_ctg_read_datas) {
  vector<CtgReadData> ctg_read_datas;
  ctg_read_datas.swap(_ctg_read_datas);  // swap first since add_read may call progress and modify ctg_read_datas in flight
  DBG_VERBOSE("add_reads ", ctg_read_datas.size(), "\n");
  for (auto &ctg_read_data : ctg_read_datas) {
    add_read(ctg_read_data);
  }
}

void CtgsWithReadsDHT::flush_ctg_updates() {
  ctg_store.flush_updates();
  ctg_store.clear();
  // read seq + qual sequences
  int est_update_size = sizeof(CtgReadData) + 500;
  int64_t mem_to_use = 0.075 * get_free_mem() / local_team().rank_n();
  auto max_store_bytes = max(mem_to_use, (int64_t)est_update_size * 150);
  ctg_read_store.set_size("CtgsWithReads add read", max_store_bytes);
}

void CtgsWithReadsDHT::flush_read_updates() {
  ctg_read_store.flush_updates();
  ctg_read_store.clear();
}

int64_t CtgsWithReadsDHT::get_num_ctgs() { return reduce_one(ctgs_map->size(), op_fast_add, 0).wait(); }

int64_t CtgsWithReadsDHT::get_local_num_ctgs() { return ctgs_map->size(); }

CtgWithReads *CtgsWithReadsDHT::get_first_local_ctg() {
  ctgs_map_iter = ctgs_map->begin();
  if (ctgs_map_iter == ctgs_map->end()) return nullptr;
  auto ctg = &ctgs_map_iter->second;
  ctgs_map_iter++;
  return ctg;
}

CtgWithReads *CtgsWithReadsDHT::get_next_local_ctg() {
  if (ctgs_map_iter == ctgs_map->end()) return nullptr;
  auto ctg = &ctgs_map_iter->second;
  ctgs_map_iter++;
  return ctg;
}

struct MerFreqs {
  // how many times this kmer has occurred: don't need to count beyond 65536
  // count of high quality extensions and low quality extensions - structure comes from kmer_dht.hpp
  ExtCounts hi_q_exts, low_q_exts;
  // the final extensions chosen - A,C,G,T, or F,X
  char ext;

  struct MerBase {
    char base;
    uint16_t nvotes_hi_q, nvotes, rating;

    uint16_t get_base_rating(int depth) {
      double min_viable = max(LASSM_MIN_VIABLE_DEPTH * depth, 2.0);
      double min_expected_depth = max(LASSM_MIN_EXPECTED_DEPTH * depth, 2.0);
      if (nvotes == 0) return 0;
      if (nvotes == 1) return 1;
      if (nvotes < min_viable) return 2;
      if (min_expected_depth > nvotes && nvotes >= min_viable && nvotes_hi_q < min_viable) return 3;
      if (min_expected_depth > nvotes && nvotes >= min_viable && nvotes_hi_q >= min_viable) return 4;
      if (nvotes >= min_expected_depth && nvotes_hi_q < min_viable) return 5;
      if (nvotes >= min_expected_depth && min_viable < nvotes_hi_q && nvotes_hi_q < min_expected_depth) return 6;
      return 7;
    }
  };

  void set_ext(int seq_depth) {
    // set extension similarly to how it is done with localassm in mhm
    MerBase mer_bases[4] = {{.base = 'A', .nvotes_hi_q = hi_q_exts.count_A, .nvotes = low_q_exts.count_A},
                            {.base = 'C', .nvotes_hi_q = hi_q_exts.count_C, .nvotes = low_q_exts.count_C},
                            {.base = 'G', .nvotes_hi_q = hi_q_exts.count_G, .nvotes = low_q_exts.count_G},
                            {.base = 'T', .nvotes_hi_q = hi_q_exts.count_T, .nvotes = low_q_exts.count_T}};
    for (int i = 0; i < 4; i++) {
      mer_bases[i].rating = mer_bases[i].get_base_rating(seq_depth);
    }
    // sort bases in descending order of quality
    sort(mer_bases, mer_bases + sizeof(mer_bases) / sizeof(mer_bases[0]), [](const auto &elem1, const auto &elem2) -> bool {
      if (elem1.rating != elem2.rating) return elem1.rating > elem2.rating;
      if (elem1.nvotes_hi_q != elem2.nvotes_hi_q) return elem1.nvotes_hi_q > elem2.nvotes_hi_q;
      if (elem1.nvotes != elem2.nvotes) return elem1.nvotes > elem2.nvotes;
      return true;
    });
    int top_rating = mer_bases[0].rating;
    int runner_up_rating = mer_bases[1].rating;
    if (top_rating < runner_up_rating) DIE("top_rating ", top_rating, " < ", runner_up_rating, "\n");
    assert(top_rating >= runner_up_rating);
    int top_rated_base = mer_bases[0].base;
    ext = 'X';
    // no extension (base = 0) if the runner up is close to the top rating
    // except, if rating is 7 (best quality), then all bases of rating 7 are forks
    if (top_rating > LASSM_RATING_THRES) {  // must have at least minViable bases
      if (top_rating <= 3) {                // must be uncontested
        if (runner_up_rating == 0) ext = top_rated_base;
      } else if (top_rating < 6) {
        if (runner_up_rating < 3) ext = top_rated_base;
      } else if (top_rating == 6) {  // viable and fair hiQ support
        if (runner_up_rating < 4) ext = top_rated_base;
      } else {  // strongest rating trumps
        if (runner_up_rating < 7) {
          ext = top_rated_base;
        } else {
          if (mer_bases[2].rating == 7 || mer_bases[0].nvotes == mer_bases[1].nvotes)
            ext = 'F';
          else if (mer_bases[0].nvotes > mer_bases[1].nvotes)
            ext = mer_bases[0].base;
          else if (mer_bases[1].nvotes > mer_bases[0].nvotes)
            ext = mer_bases[1].base;
        }
      }
    }
  }
};

void process_reads(unsigned kmer_len, vector<PackedReads *> &packed_reads_list, ReadsToCtgsDHT &reads_to_ctgs,
                   CtgsWithReadsDHT &ctgs_dht) {
  BarrierTimer timer(__FILEFUNC__);
  int64_t num_reads = 0;
  int64_t num_read_maps_found = 0;
  future<> all_done = make_future();
  vector<CtgReadData> ctgs_to_add;
  int64_t num_local_reads = 0;
  for (auto packed_reads : packed_reads_list) {
    num_local_reads += packed_reads->get_local_num_reads();
  }
  ProgressBar progbar(num_local_reads * 2, "Processing reads - two stage");

  for (auto packed_reads : packed_reads_list) {
    packed_reads->reset();
    DBG_VERBOSE("Processing packed_reads=", packed_reads, " with ", packed_reads->get_local_num_reads(), "\n");
    string id, seq, quals;
    auto sh_rank_read_ids = make_shared<vector<vector<pair<string, uint64_t>>>>(rank_n());
    vector<vector<pair<string, uint64_t>>> &rank_read_ids = *sh_rank_read_ids;
    assert(rank_read_ids.size() == rank_n());
    while (true) {
      progress();
      auto read_idx = packed_reads->get_read_index();
      if (!packed_reads->get_next_read(id, seq, quals)) break;
      progbar.update();
      // this happens when we have a placeholder entry because reads merged
      if (kmer_len > seq.length()) continue;
      num_reads++;
      auto target_rank = ReadsToCtgsDHT::get_target_rank(id);
      rank_read_ids[target_rank].push_back({id, read_idx});
    }

    for (auto target_rank : foreach_rank_by_node()) {
      progress();
      ctgs_dht.add_reads(ctgs_to_add);
      auto &read_ids = rank_read_ids[target_rank];
      if (read_ids.empty()) continue;
      vector<string> just_ids(read_ids.size());
      for (int i = 0; i < read_ids.size(); i++) {
        just_ids[i] = read_ids[i].first;
      }
      DBG_VERBOSE("Getting contigs from ", target_rank, " for ", just_ids.size(), " reads\n");
      auto read_ctgs_fut = reads_to_ctgs.get_ctgs(target_rank, just_ids);
      auto fut = read_ctgs_fut.then([sh_rank_read_ids, target_rank, packed_reads, &progbar, &num_read_maps_found,
                                     &ctgs_to_add](vector<vector<CtgInfo>> read_ctgs) {
        DBG_VERBOSE("Processing gotten contigs from ", target_rank, " for ", read_ctgs.size(), "\n");
        auto &read_ids = (*sh_rank_read_ids)[target_rank];
        assert(read_ctgs.size() == read_ids.size());
        string id, seq, quals, seq_rc, quals_rc;
        for (size_t i = 0; i < read_ctgs.size(); i++) {
          DBG_VERBOSE("processing i=", i, " of ", read_ctgs.size(), "\n");
          progbar.update();
          assert(read_ctgs.size() == read_ids.size());
          assert(i < read_ids.size());
          const auto &read_id = read_ids[i].first;
          const auto &read_idx = read_ids[i].second;
          DBG_VERBOSE("Loading packed_read idx=", read_idx, "\n");
          assert(i < read_ctgs.size());
          vector<CtgInfo> &ctgs = read_ctgs[i];
          DBG_VERBOSE(" ctgs=", ctgs.size(), "\n");
          if (ctgs.size()) {
            num_read_maps_found++;
            DBG_VERBOSE(" packed_reads=", packed_reads, " of ", packed_reads->get_local_num_reads(), "\n");
            packed_reads->get_read(read_idx, id, seq, quals);
            assert(id.compare(read_id) == 0);
            bool was_revcomp = false;
            for (auto &ctg : ctgs) {
              if ((ctg.orient == '-' && ctg.side == 'R') || (ctg.orient == '+' && ctg.side == 'L')) {
                if (!was_revcomp) {
                  seq_rc = revcomp(seq);
                  quals_rc = quals;
                  reverse(quals_rc.begin(), quals_rc.end());
                  was_revcomp = true;
                }
                ctgs_to_add.push_back(CtgReadData(ctg.cid, ctg.side, {id, seq_rc, quals_rc}));
              } else {
                ctgs_to_add.push_back(CtgReadData(ctg.cid, ctg.side, {id, seq, quals}));
              }
            }
          }
        }
        // clear out some memory
        vector<pair<string, uint64_t>>().swap(read_ids);
      });
      upcxx_utils::limit_outstanding_futures(fut).wait();
      ctgs_dht.add_reads(ctgs_to_add);
    }
  }
  all_done = when_all(all_done, progbar.set_done());
  auto all_outstanding = flush_outstanding_futures_async();
  while (!all_outstanding.ready()) {
    ctgs_dht.add_reads(ctgs_to_add);
    progress();
  }
  ctgs_dht.add_reads(ctgs_to_add);
  assert(flush_outstanding_futures_async().ready());
  assert(ctgs_to_add.empty());
  ctgs_dht.flush_read_updates();

  auto tot_num_reads_fut = reduce_one(num_reads, op_fast_add, 0);
  auto tot_num_read_maps_found_fut = reduce_one(num_read_maps_found, op_fast_add, 0);
  all_done = when_all(all_done, tot_num_reads_fut, tot_num_read_maps_found_fut)
                 .then([](int64_t tot_num_reads, int64_t tot_num_read_maps_found) {
                   SLOG_VERBOSE("Found ", perc_str(tot_num_read_maps_found, tot_num_reads), " reads that map to contigs\n");
                 });

  all_done.wait();
  // implicit barrier on exit
}

static void get_best_aln_for_read(const Alns &alns, int64_t &i, Aln &best_aln, AlnStatus &best_start_status,
                                  AlnStatus &best_end_status, int64_t &num_alns_found, int64_t &num_alns_invalid) {
  auto classify_aln = [](int runaligned, int cunaligned) -> AlnStatus {
    if (runaligned > cunaligned && cunaligned < KLIGN_UNALIGNED_THRES) return AlnStatus::EXTENDS_CONTIG;
    if (runaligned <= cunaligned && runaligned < KLIGN_UNALIGNED_THRES) return AlnStatus::OVERLAPS_CONTIG;
    return AlnStatus::NO_ALN;
  };

  // choose the highest scoring aln for this read that is useful
  best_start_status = AlnStatus::NO_ALN;
  best_end_status = AlnStatus::NO_ALN;
  string start_read_id = "";
  int best_aln_score = 0;
  best_aln.read_id = "";
  for (; i < (int64_t)alns.size(); i++) {
    const Aln aln = alns.get_aln(i);
    // alns for a new read
    if (start_read_id != "" && aln.read_id != start_read_id) return;
    num_alns_found++;
    if (aln.score1 < best_aln_score) continue;
    AlnStatus start_status, end_status;
    if (aln.orient == '+') {
      start_status = classify_aln(aln.rstart - 1, aln.cstart - 1);
      end_status = classify_aln(aln.rlen - aln.rstop, aln.clen - aln.cstop);
    } else {
      // for '-' strand, aln is between read and revcomp of contig
      start_status = classify_aln(aln.rstart - 1, aln.clen - aln.cstop);
      end_status = classify_aln(aln.rlen - aln.rstop, aln.cstart - 1);
    }
    if (start_status == AlnStatus::NO_ALN || end_status == AlnStatus::NO_ALN) {
      num_alns_invalid++;
      continue;
    }
    best_aln = aln;
    best_aln_score = aln.score1;
    best_start_status = start_status;
    best_end_status = end_status;
    start_read_id = aln.read_id;
  }
}

void process_alns(const Alns &alns, ReadsToCtgsDHT &reads_to_ctgs, int insert_avg, int insert_stddev) {
  auto pair_overlap = [](Aln &aln, int min_pair_len) -> bool {
    // make sure that the mate won't overlap the same contig
    if (aln.orient == '+') {
      if (min_pair_len - aln.rlen - aln.rstart + 1 <= aln.clen - aln.cstart) return true;
    } else {
      if (min_pair_len - 2 * aln.rlen + aln.rstart - 1 <= aln.cstart) return true;
    }
    return false;
  };

  BarrierTimer timer(__FILEFUNC__);
  int64_t num_alns_found = 0, num_alns_invalid = 0, num_direct = 0, num_proj = 0;
  int min_pair_len = insert_avg + 3 * insert_stddev;
  IntermittentTimer t_get_alns(__FILENAME__ + string(":") + "get alns reads to contigs");
  int64_t aln_i = 0;
  AlnStatus start_status, end_status;
  ProgressBar progbar(alns.size(), "Getting read-to-contig mappings from alignments");
  while (aln_i < (int64_t)alns.size()) {
    progress();
    Aln aln;
    t_get_alns.start();
    get_best_aln_for_read(alns, aln_i, aln, start_status, end_status, num_alns_found, num_alns_invalid);
    t_get_alns.stop();
    progbar.update(aln_i);
    if (aln.read_id.empty()) continue;
    // add a direct extension to the contig, start or end
    if (start_status == AlnStatus::EXTENDS_CONTIG) {
      reads_to_ctgs.add(aln.read_id, aln.cid, aln.orient, aln.orient == '+' ? 'L' : 'R');
      num_direct++;
    } else if (end_status == AlnStatus::EXTENDS_CONTIG) {
      reads_to_ctgs.add(aln.read_id, aln.cid, aln.orient, aln.orient == '+' ? 'R' : 'L');
      num_direct++;
    }
    // FIXME: if read is longer than one pair read length, don't look for mate since it is a merged read
    // add mate pair if feasible
    if (!pair_overlap(aln, min_pair_len)) {
      // indicate the other pair number
      int len = aln.read_id.length();
      assert(len > 1);
      if (aln.read_id[len - 1] == '1')
        aln.read_id[len - 1] = '2';
      else if (aln.read_id[len - 1] == '2')
        aln.read_id[len - 1] = '1';
      else
        DIE("Bad pair number ", (int)aln.read_id[len - 1], " in read: ", aln.read_id);
      reads_to_ctgs.add(aln.read_id, aln.cid, aln.orient == '+' ? '-' : '+', aln.orient == '+' ? 'R' : 'L');
      num_proj++;
    }
  }
  reads_to_ctgs.flush_updates();
  progbar.done();
  barrier();
  t_get_alns.done_all();
  auto tot_alns_found = reduce_one(num_alns_found, op_fast_add, 0).wait();
  SLOG_VERBOSE("Processed ", tot_alns_found, " alignments:\n");
  SLOG_VERBOSE("  invalid:   ", perc_str(reduce_one(num_alns_invalid, op_fast_add, 0).wait(), tot_alns_found), "\n");
  SLOG_VERBOSE("  direct:    ", perc_str(reduce_one(num_direct, op_fast_add, 0).wait(), tot_alns_found), "\n");
  SLOG_VERBOSE("  projected: ", perc_str(reduce_one(num_proj, op_fast_add, 0).wait(), tot_alns_found), "\n");
  SLOG_VERBOSE("Added ", reads_to_ctgs.get_num_mappings(), " mappings\n");
}

using MerMap = HASH_TABLE<string, MerFreqs>;

static void count_mers(vector<ReadSeq> &reads, MerMap &mers_ht, int seq_depth, int mer_len, int qual_offset,
                       int64_t &excess_reads) {
  int num_reads = 0;
  // rough estimate of number of kmers
  int max_mers = 0;
  for (int i = 0; i < min((int)reads.size(), (int)LASSM_MAX_COUNT_MERS_READS); i++) {
    max_mers += reads[i].seq.length() - mer_len;
  }
  mers_ht.reserve(max_mers);
  // split reads into kmers and count frequency of high quality extensions
  for (auto &read_seq : reads) {
    num_reads++;
    if (num_reads > LASSM_MAX_COUNT_MERS_READS) {
      excess_reads += reads.size() - LASSM_MAX_COUNT_MERS_READS;
      break;
    }
    if (mer_len >= (int)read_seq.seq.length()) continue;
    int num_mers = read_seq.seq.length() - mer_len;
    for (int start = 0; start < num_mers; start++) {
      string mer = read_seq.seq.substr(start, mer_len);
      // skip mers that contain Ns
      if (mer.find("N") != string::npos) continue;
      auto it = mers_ht.find(mer);
      if (it == mers_ht.end()) it = mers_ht.insert({mer, {.hi_q_exts = {0}, .low_q_exts = {0}, .ext = 0}}).first;
      int ext_pos = start + mer_len;
      assert(ext_pos < (int)read_seq.seq.length());
      char ext = read_seq.seq[ext_pos];
      if (ext == 'N') continue;
      int qual = read_seq.quals[ext_pos] - qual_offset;
      if (qual >= LASSM_MIN_QUAL) it->second.low_q_exts.inc(ext, 1);
      if (qual >= LASSM_MIN_HI_QUAL) it->second.hi_q_exts.inc(ext, 1);
    }
  }
  // now set extension choices
  for (auto &elem : mers_ht) {
    elem.second.set_ext(seq_depth);
  }
}

// return the result of the walk (f, r or x)
static char walk_mers(MerMap &mers_ht, string &mer, string &walk, int mer_len, int walk_len_limit) {
  HASH_TABLE<string, bool> loop_check_ht;
  char walk_result = 'X';
  for (int nsteps = 0; nsteps < walk_len_limit; nsteps++) {
    // check for a cycle in the graph
    if (loop_check_ht.find(mer) != loop_check_ht.end()) {
      walk_result = 'R';
      break;
    } else {
      loop_check_ht.insert({mer, true});
    }
    auto it = mers_ht.find(mer);
    if (it == mers_ht.end()) {
      walk_result = 'X';
      break;
    }
    char ext = it->second.ext;
    if (ext == 'F' || ext == 'X') {
      walk_result = ext;
      break;
    }
    mer.erase(0, 1);
    mer += ext;
    walk += ext;
  }
  return walk_result;
}

static string iterative_walks(string &seq, int seq_depth, vector<ReadSeq> &reads, int max_mer_len, int kmer_len, int qual_offset,
                              int walk_len_limit, array<int64_t, 3> &term_counts, int64_t &num_walks, int64_t &max_walk_len,
                              int64_t &sum_ext, IntermittentTimer &count_mers_timer, IntermittentTimer &walk_mers_timer,
                              int64_t &excess_reads) {
  int min_mer_len = LASSM_MIN_KMER_LEN;
  max_mer_len = min(max_mer_len, (int)seq.length());
  // iteratively walk starting from kmer_size, increasing mer size on a fork (F) or repeat (R),
  // and decreasing on an end of path (X)
  // look for the longest walk. Note we have to restart from the beginning for each walk to ensure
  // that all loops will be detected
  string longest_walk = "";
  int shift = 0;
  DBG_VERBOSE("  reads:\n");
#ifdef DEBUG
  for (auto &read_seq : reads) {
    DBG_VERBOSE("    ", read_seq.read_id, "\n", read_seq.seq, "\n");
  }
#endif
  for (int mer_len = kmer_len; mer_len >= min_mer_len && mer_len <= max_mer_len; mer_len += shift) {
    count_mers_timer.start();
    MerMap mers_ht;
    count_mers(reads, mers_ht, seq_depth, mer_len, qual_offset, excess_reads);
    count_mers_timer.stop();
    string mer = seq.substr(seq.length() - mer_len);
    string walk = "";
    walk_mers_timer.start();
    char walk_result = walk_mers(mers_ht, mer, walk, mer_len, walk_len_limit);
    walk_mers_timer.stop();
    int walk_len = walk.length();
    if (walk_len > (int)longest_walk.length()) longest_walk = walk;
    if (walk_result == 'X') {
      term_counts[0]++;
      // walk reaches a dead-end, downshift, unless we were upshifting
      if (shift == LASSM_SHIFT_SIZE) break;
      shift = -LASSM_SHIFT_SIZE;
    } else {
      if (walk_result == 'F')
        term_counts[1]++;
      else
        term_counts[2]++;
      // otherwise walk must end with a fork or repeat, so upshift
      if (shift == -LASSM_SHIFT_SIZE) break;
      if (mer_len > (int)seq.length()) break;
      shift = LASSM_SHIFT_SIZE;
    }
  }
  if (!longest_walk.empty()) {
    num_walks++;
    max_walk_len = max(max_walk_len, (int64_t)longest_walk.length());
    sum_ext += longest_walk.length();
  }
  return longest_walk;
}

void extend_ctg(CtgWithReads *ctg, WalkMetrics &wm, int insert_avg, int insert_stddev, int max_kmer_len, int kmer_len,
                int qual_offset, int walk_len_limit, IntermittentTimer &count_mers_timer, IntermittentTimer &walk_mers_timer) {
  wm.sum_clen += ctg->seq.length();
  if (ctg->reads_right.size()) {
    wm.num_sides++;
    wm.num_reads += ctg->reads_right.size();
    wm.max_num_reads = max(wm.max_num_reads, (int64_t)ctg->reads_right.size());
    DBG_VERBOSE("walk right ctg ", ctg->cid, " ", ctg->depth, "\n", ctg->seq, "\n");
    // have to do right first because the contig needs to be revcomped for the left
    string right_walk =
        iterative_walks(ctg->seq, ctg->depth, ctg->reads_right, max_kmer_len, kmer_len, qual_offset, walk_len_limit, wm.term_counts,
                        wm.num_walks, wm.max_walk_len, wm.sum_ext, count_mers_timer, walk_mers_timer, wm.excess_reads);
    if (!right_walk.empty()) ctg->seq += right_walk;
  }
  if (ctg->reads_left.size()) {
    wm.num_sides++;
    wm.num_reads += ctg->reads_left.size();
    wm.max_num_reads = max(wm.max_num_reads, (int64_t)ctg->reads_left.size());
    string seq_rc = revcomp(ctg->seq);
    DBG_VERBOSE("walk left ctg ", ctg->cid, " ", ctg->depth, "\n", seq_rc, "\n");
    string left_walk =
        iterative_walks(seq_rc, ctg->depth, ctg->reads_left, max_kmer_len, kmer_len, qual_offset, walk_len_limit, wm.term_counts,
                        wm.num_walks, wm.max_walk_len, wm.sum_ext, count_mers_timer, walk_mers_timer, wm.excess_reads);
    if (!left_walk.empty()) {
      left_walk = revcomp(left_walk);
      ctg->seq.insert(0, left_walk);
    }
  }
}

void add_ctgs(CtgsWithReadsDHT &ctgs_dht, Contigs &ctgs) {
  BarrierTimer timer(__FILEFUNC__);
  // process the local ctgs and insert into the distributed hash table
  ProgressBar progbar(ctgs.size(), "Adding contigs to distributed hash table");
  for (auto it = ctgs.begin(); it != ctgs.end(); ++it) {
    progbar.update();
    ctgs_dht.add_ctg(*it);
    progress();
  }
  ctgs_dht.flush_ctg_updates();
  progbar.done();
  SLOG_VERBOSE("Added ", ctgs_dht.get_num_ctgs(), " contigs\n");
}

}  // namespace localassm_core

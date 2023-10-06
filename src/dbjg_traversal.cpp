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

#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <upcxx/upcxx.hpp>

#include "contigs.hpp"
#include "kcount.hpp"
#include "kmer_dht.hpp"
#include "upcxx_utils/log.hpp"
#include "upcxx_utils/progress_bar.hpp"
#include "upcxx_utils/reduce_prefix.hpp"
#include "utils.hpp"

#define DBG_TRAVERSE DBG
//#define DBG_TRAVERSE(...)

using namespace std;
using namespace upcxx;
using namespace upcxx_utils;

enum class Dirn { LEFT, RIGHT, NONE };
#define DIRN_STR(d) ((d) == Dirn::LEFT ? "left" : (d) == Dirn::RIGHT ? "right" : "none")

enum class WalkStatus { RUNNING = '-', DEADEND = 'X', FORK = 'F', CONFLICT = 'O', REPEAT = 'R', VISITED = 'V' };

struct FragElem;
struct FragElem {
  global_ptr<FragElem> left_gptr, right_gptr;
  bool left_is_rc, right_is_rc;
  global_ptr<char> frag_seq;
  unsigned frag_len;
  int64_t sum_depths;
  bool visited;

  FragElem()
      : left_gptr(nullptr)
      , right_gptr(nullptr)
      , left_is_rc(false)
      , right_is_rc(false)
      , frag_seq(nullptr)
      , frag_len(0)
      , sum_depths(0)
      , visited(false) {}
};

template <int MAX_K>
struct StepInfo {
  WalkStatus walk_status;
  uint32_t sum_depths;
  char prev_ext;
  char next_ext;
  global_ptr<FragElem> visited_frag_elem_gptr;
  string uutig;
  Kmer<MAX_K> kmer;

  StepInfo() = default;
  StepInfo(Kmer<MAX_K> kmer, char prev_ext, char next_ext)
      : walk_status{}
      , sum_depths(0)
      , prev_ext(prev_ext)
      , next_ext(next_ext)
      , visited_frag_elem_gptr{}
      , uutig{}
      , kmer(kmer) {}

  UPCXX_SERIALIZED_FIELDS(walk_status, sum_depths, prev_ext, next_ext, visited_frag_elem_gptr, uutig, kmer);
};

struct WalkTermStats {
  int64_t num_deadends, num_forks, num_conflicts, num_repeats, num_visited;

  void update(WalkStatus walk_status) {
    switch (walk_status) {
      case WalkStatus::DEADEND: num_deadends++; break;
      case WalkStatus::FORK: num_forks++; break;
      case WalkStatus::CONFLICT: num_conflicts++; break;
      case WalkStatus::REPEAT: num_repeats++; break;
      case WalkStatus::VISITED: num_visited++; break;
      default: DIE("Should never get here\n"); break;
    }
  }

  void print() {
    auto all_num_deadends = reduce_one(num_deadends, op_fast_add, 0).wait();
    auto all_num_forks = reduce_one(num_forks, op_fast_add, 0).wait();
    auto all_num_conflicts = reduce_one(num_conflicts, op_fast_add, 0).wait();
    auto all_num_repeats = reduce_one(num_repeats, op_fast_add, 0).wait();
    auto all_num_visited = reduce_one(num_visited, op_fast_add, 0).wait();
    auto tot_ends = all_num_forks + all_num_deadends + all_num_conflicts + all_num_repeats + all_num_visited;
    SLOG_VERBOSE("Walk statistics:\n");
    SLOG_VERBOSE("  deadends:  ", perc_str(all_num_deadends, tot_ends), "\n");
    SLOG_VERBOSE("  forks:     ", perc_str(all_num_forks, tot_ends), "\n");
    SLOG_VERBOSE("  conflicts: ", perc_str(all_num_conflicts, tot_ends), "\n");
    SLOG_VERBOSE("  repeats:   ", perc_str(all_num_repeats, tot_ends), "\n");
    SLOG_VERBOSE("  visited:   ", perc_str(all_num_visited, tot_ends), "\n");
  }
};

static string gptr_str(global_ptr<FragElem> gptr) {
  if (!gptr) return string(10, '0');
  ostringstream oss;
  oss << setw(11);
  oss << gptr;
  string s = oss.str();
  s.erase(0, s.length() - 6);
  return to_string(gptr.where()) + ":" + s;
}

template <int MAX_K>
static bool check_kmers(const string &seq, dist_object<KmerDHT<MAX_K>> &kmer_dht, int kmer_len) {
  vector<Kmer<MAX_K>> kmers;
  Kmer<MAX_K>::get_kmers(kmer_len, seq, kmers, true);
  for (auto &kmer : kmers) {
    assert(kmer.is_valid());
    if (!kmer_dht->kmer_exists(kmer)) return false;
  }
  return true;
}

template <int MAX_K>
StepInfo<MAX_K> get_next_step(dist_object<KmerDHT<MAX_K>> &kmer_dht, const Kmer<MAX_K> start_kmer, const Dirn dirn,
                              const char start_prev_ext, const char start_next_ext, bool revisit_allowed, bool is_rc,
                              const global_ptr<FragElem> frag_elem_gptr) {
  StepInfo<MAX_K> step_info(start_kmer, start_prev_ext, start_next_ext);
  while (true) {
    KmerCounts *kmer_counts = kmer_dht->get_local_kmer_counts(step_info.kmer);
    // this kmer doesn't exist, abort
    if (!kmer_counts) {
      step_info.walk_status = WalkStatus::DEADEND;
      break;
    }
    char left = kmer_counts->left;
    char right = kmer_counts->right;
    if (left == 'X' || right == 'X') {
      step_info.walk_status = WalkStatus::DEADEND;
      break;
    }
    if (left == 'F' || right == 'F') {
      step_info.walk_status = WalkStatus::FORK;
      break;
    }
    if (is_rc) {
      left = comp_nucleotide(left);
      right = comp_nucleotide(right);
      swap(left, right);
    }
    // check for conflicts
    if (step_info.prev_ext &&
        ((dirn == Dirn::LEFT && step_info.prev_ext != right) || (dirn == Dirn::RIGHT && step_info.prev_ext != left))) {
      step_info.walk_status = WalkStatus::CONFLICT;
      break;
    }
    // if visited by another rank first
    if (kmer_counts->uutig_frag && kmer_counts->uutig_frag != frag_elem_gptr) {
      step_info.walk_status = WalkStatus::VISITED;
      step_info.visited_frag_elem_gptr = kmer_counts->uutig_frag;
      break;
    }
    // a repeat, abort (but allowed if traversing right after adding start kmer previously)
    if (kmer_counts->uutig_frag == frag_elem_gptr && !revisit_allowed) {
      step_info.walk_status = WalkStatus::REPEAT;
      break;
    }
    // mark as visited
    kmer_counts->uutig_frag = frag_elem_gptr;
    step_info.uutig += step_info.next_ext;
    step_info.next_ext = (dirn == Dirn::LEFT ? left : right);
    if (is_rc) step_info.kmer = step_info.kmer.revcomp();
    if (dirn == Dirn::LEFT) {
      step_info.prev_ext = step_info.kmer.back();
      step_info.kmer = step_info.kmer.backward_base(step_info.next_ext);
    } else {
      step_info.prev_ext = step_info.kmer.front();
      step_info.kmer = step_info.kmer.forward_base(step_info.next_ext);
    }
    step_info.walk_status = WalkStatus::RUNNING;
    step_info.sum_depths += kmer_counts->count;

    revisit_allowed = false;
    auto kmer = step_info.kmer;
    auto kmer_rc = kmer.revcomp();
    is_rc = false;
    if (kmer_rc < kmer) {
      kmer.swap(kmer_rc);
      is_rc = true;
    }
    auto target_rank = kmer_dht->get_kmer_target_rank(kmer, &kmer_rc);
    // next kmer is remote, return to rpc caller
    if (target_rank != rank_me()) break;
    // next kmer is local to this rank, continue walking
    step_info.kmer = kmer;
  }
  return step_info;
}

static int64_t _num_rank_me_rpcs = 0;
static int64_t _num_node_rpcs = 0;
static int64_t _num_rpcs = 0;

template <int MAX_K>
static global_ptr<FragElem> traverse_dirn(dist_object<KmerDHT<MAX_K>> &kmer_dht, Kmer<MAX_K> kmer,
                                          global_ptr<FragElem> frag_elem_gptr, Dirn dirn, string &uutig, int64_t &sum_depths,
                                          WalkTermStats &walk_term_stats) {
  char prev_ext = 0;
  char next_ext = (dirn == Dirn::LEFT ? kmer.front() : kmer.back());
  bool revisit_allowed = (dirn == Dirn::LEFT ? false : true);
  if (dirn == Dirn::RIGHT) {
    string kmer_str = kmer.to_string();
    uutig += substr_view(kmer_str, 1, kmer_str.length() - 2);
  }
  while (true) {
    Kmer<MAX_K> next_kmer = kmer;
    auto kmer_rc = kmer.revcomp();
    bool is_rc = false;
    if (kmer_rc < kmer) {
      next_kmer.swap(kmer_rc);
      is_rc = true;
    }
    auto target_rank = kmer_dht->get_kmer_target_rank(next_kmer, &kmer_rc);
    if (target_rank == rank_me()) _num_rank_me_rpcs++;
    if (local_team_contains(target_rank)) _num_node_rpcs++;
    _num_rpcs++;
    StepInfo<MAX_K> step_info;
    if (target_rank == rank_me())
      step_info = get_next_step<MAX_K>(kmer_dht, next_kmer, dirn, prev_ext, next_ext, revisit_allowed, is_rc, frag_elem_gptr);
    else
      step_info = rpc(target_rank, get_next_step<MAX_K>, kmer_dht, next_kmer, dirn, prev_ext, next_ext, revisit_allowed, is_rc,
                      frag_elem_gptr)
                      .wait();
    revisit_allowed = false;
    sum_depths += step_info.sum_depths;
    uutig += step_info.uutig;
    if (step_info.walk_status != WalkStatus::RUNNING) {
      walk_term_stats.update(step_info.walk_status);
      // reverse it because we were walking backwards
      if (dirn == Dirn::LEFT) reverse(uutig.begin(), uutig.end());
      return step_info.visited_frag_elem_gptr;
    }
    // now attempt to walk to next remote kmer
    next_ext = step_info.next_ext;
    prev_ext = step_info.prev_ext;
    kmer = step_info.kmer;
  }
}

template <int MAX_K>
static void construct_frags(unsigned kmer_len, dist_object<KmerDHT<MAX_K>> &kmer_dht, vector<global_ptr<FragElem>> &frag_elems) {
  BarrierTimer timer(__FILEFUNC__);
  _num_rank_me_rpcs = 0;
  _num_node_rpcs = 0;
  _num_rpcs = 0;
  // allocate space for biggest possible uutig in global storage
  WalkTermStats walk_term_stats = {0};
  int64_t num_walks = 0;
  barrier();
  ProgressBar progbar(kmer_dht->get_local_num_kmers(), "DeBruijn graph traversal to construct uutig fragments");
  for (auto it = kmer_dht->local_kmers_begin(); it != kmer_dht->local_kmers_end(); it++) {
    progress();
    progbar.update();
    auto kmer = it->first;
    auto kmer_counts = &it->second;
    // don't start any new walk if this kmer has already been visited
    if (kmer_counts->uutig_frag) continue;
    // don't start walks on kmers without extensions on both sides
    if (kmer_counts->left == 'X' || kmer_counts->left == 'F' || kmer_counts->right == 'X' || kmer_counts->right == 'F') continue;
    string uutig;
    int64_t sum_depths = 0;
    global_ptr<FragElem> frag_elem_gptr = new_<FragElem>();
    auto left_gptr = traverse_dirn(kmer_dht, kmer, frag_elem_gptr, Dirn::LEFT, uutig, sum_depths, walk_term_stats);
    auto right_gptr = traverse_dirn(kmer_dht, kmer, frag_elem_gptr, Dirn::RIGHT, uutig, sum_depths, walk_term_stats);
    FragElem *frag_elem = frag_elem_gptr.local();
    frag_elem->frag_seq = new_array<char>(uutig.length() + 1);
    strcpy(frag_elem->frag_seq.local(), uutig.c_str());
    frag_elem->frag_seq.local()[uutig.length()] = 0;
    frag_elem->frag_len = uutig.length();
    frag_elem->sum_depths = sum_depths;
    frag_elem->left_gptr = left_gptr;
    frag_elem->right_gptr = right_gptr;
    frag_elems.push_back(frag_elem_gptr);
    num_walks++;
  }
  progbar.done();
  barrier();
  auto tot_rank_me_rpcs = reduce_one(_num_rank_me_rpcs, op_fast_add, 0).wait();
  auto tot_node_rpcs = reduce_one(_num_node_rpcs, op_fast_add, 0).wait();
  auto tot_rpcs = reduce_one(_num_rpcs, op_fast_add, 0).wait();
  SLOG_VERBOSE("Required ", tot_rpcs, " rpcs, of which ", perc_str(tot_rank_me_rpcs, tot_rpcs), " were same rank, ",
               perc_str(tot_node_rpcs, tot_rpcs), " were intra-node, and ", perc_str(tot_rpcs - tot_node_rpcs, tot_rpcs),
               " were inter-node\n");
  walk_term_stats.print();
}

static int64_t print_link_stats(int64_t num_links, int64_t num_overlaps, int64_t num_overlaps_rc, const string &dirn_str) {
  auto all_num_links = reduce_one(num_links, op_fast_add, 0).wait();
  auto all_num_overlaps = reduce_one(num_overlaps, op_fast_add, 0).wait();
  auto all_num_overlaps_rc = reduce_one(num_overlaps_rc, op_fast_add, 0).wait();
  SLOG_VERBOSE("Found ", all_num_links, " ", dirn_str, " links with ", perc_str(all_num_overlaps, all_num_links), " overlaps and ",
               perc_str(all_num_overlaps_rc, all_num_links), " revcomped overlaps\n");
  return all_num_links;
}

static bool is_overlap(const string &left_seq, const string &right_seq, int overlap_len) {
  return (left_seq.compare(left_seq.length() - overlap_len, overlap_len, right_seq, 0, overlap_len) == 0);
}

static string get_frag_seq(FragElem &frag_elem) {
  char *buf = new char[frag_elem.frag_len + 1];
  rget(frag_elem.frag_seq, buf, frag_elem.frag_len + 1).wait();
  string frag_seq(buf);
  assert(frag_seq.length() == frag_elem.frag_len);
  delete[] buf;
  return frag_seq;
}

static void set_link_status(Dirn dirn, global_ptr<FragElem> &nb_gptr, bool &is_rc, string &uutig, int kmer_len,
                            int64_t &num_overlaps, int64_t &num_overlaps_rc, int64_t &num_non_recip) {
  if (nb_gptr) {
    FragElem nb_frag_elem = rget(nb_gptr).wait();
    string nb_frag_seq = get_frag_seq(nb_frag_elem);
    string *s1 = (dirn == Dirn::LEFT ? &nb_frag_seq : &uutig);
    string *s2 = (dirn == Dirn::LEFT ? &uutig : &nb_frag_seq);
    if (is_overlap(*s1, *s2, kmer_len - 1)) {
      if ((dirn == Dirn::LEFT ? nb_frag_elem.right_gptr : nb_frag_elem.left_gptr) == nb_gptr) {
        num_non_recip++;
        nb_gptr = nullptr;
        return;
      }
      num_overlaps++;
      return;
    }
    auto nb_frag_seq_rc = revcomp(nb_frag_seq);
    s1 = (dirn == Dirn::LEFT ? &nb_frag_seq_rc : &uutig);
    s2 = (dirn == Dirn::LEFT ? &uutig : &nb_frag_seq_rc);
    if (is_overlap(*s1, *s2, kmer_len - 1)) {
      if ((dirn == Dirn::LEFT ? nb_frag_elem.left_gptr : nb_frag_elem.right_gptr) == nb_gptr) {
        num_non_recip++;
        nb_gptr = nullptr;
        return;
      }
      num_overlaps_rc++;
      is_rc = true;
      return;
    }
    DBG_TRAVERSE("No ", DIRN_STR(dirn), " overlap:\n", uutig, "\n", nb_frag_seq, "\n", nb_frag_seq_rc, "\n");
  }
}

template <int MAX_K>
static void clean_frag_links(unsigned kmer_len, dist_object<KmerDHT<MAX_K>> &kmer_dht, vector<global_ptr<FragElem>> &frag_elems) {
  BarrierTimer timer(__FILEFUNC__);
  // put all the uutigs found by this rank into my_uutigs
  int64_t num_equal_links = 0, num_non_recip = 0, num_short = 0, num_left_links = 0, num_left_overlaps = 0,
          num_left_overlaps_rc = 0, num_right_links = 0, num_right_overlaps = 0, num_right_overlaps_rc = 0;
  ProgressBar progbar(frag_elems.size(), "Cleaning fragment links");
  for (auto frag_elem_gptr : frag_elems) {
    progbar.update();
    FragElem *frag_elem = frag_elem_gptr.local();
    if (frag_elem->frag_len < kmer_len) {
      num_short++;
      continue;
    }
    if (frag_elem->left_gptr) num_left_links++;
    if (frag_elem->right_gptr) num_right_links++;
    string uutig(frag_elem->frag_seq.local());
    if (frag_elem->left_gptr && frag_elem->left_gptr == frag_elem->right_gptr) {
      num_equal_links++;
      frag_elem->left_gptr = nullptr;
      frag_elem->right_gptr = nullptr;
      continue;
    }
    set_link_status(Dirn::LEFT, frag_elem->left_gptr, frag_elem->left_is_rc, uutig, kmer_len, num_left_overlaps,
                    num_left_overlaps_rc, num_non_recip);
    set_link_status(Dirn::RIGHT, frag_elem->right_gptr, frag_elem->right_is_rc, uutig, kmer_len, num_right_overlaps,
                    num_right_overlaps_rc, num_non_recip);
  }
  progbar.done();
  barrier();
  auto all_num_frags = reduce_one(frag_elems.size(), op_fast_add, 0).wait();
  auto all_num_short = reduce_one(num_short, op_fast_add, 0).wait();
  SLOG_VERBOSE("Found ", all_num_frags, " uutig fragments of which ", perc_str(all_num_short, all_num_frags), " are short\n");
  auto all_num_left = print_link_stats(num_left_links, num_left_overlaps, num_left_overlaps_rc, "left");
  auto all_num_right = print_link_stats(num_right_links, num_right_overlaps, num_right_overlaps_rc, "right");
  auto all_num_equal_links = reduce_one(num_equal_links, op_fast_add, 0).wait();
  auto all_num_non_recip = reduce_one(num_non_recip, op_fast_add, 0).wait();
  SLOG_VERBOSE("There were ", perc_str(all_num_equal_links, all_num_left + all_num_right), " equal left and right links\n");
  SLOG_VERBOSE("There were ", perc_str(all_num_non_recip, all_num_left + all_num_right), " non-reciprocating links\n");
}

static global_ptr<FragElem> get_other_side_gptr(const FragElem &frag_elem, global_ptr<FragElem> frag_elem_gptr) {
  if (frag_elem.left_gptr == frag_elem_gptr) return frag_elem.right_gptr;
  return frag_elem.left_gptr;
}

static bool walk_frags_dirn(unsigned kmer_len, global_ptr<FragElem> frag_elem_gptr, global_ptr<FragElem> next_gptr, string &uutig,
                            int64_t &depths, int64_t &walk_steps, int64_t &num_repeats, vector<FragElem *> &my_frag_elems_visited) {
  if (!next_gptr) return true;
  global_ptr<FragElem> prev_gptr = frag_elem_gptr;
  FragElem prev_frag_elem = *frag_elem_gptr.local();
  // for checking that we haven't got a bug - frags should never be revisited in a walk
  HASH_TABLE<global_ptr<FragElem>, bool> visited;
  visited[frag_elem_gptr] = true;
#ifdef DEBUG
  string padding;
  DBG_TRAVERSE(uutig, "\n");
#endif
  Dirn dirn = Dirn::NONE;
  while (next_gptr) {
    DBG_TRAVERSE(padding, gptr_str(get_other_side_gptr(prev_frag_elem, next_gptr)), " <-- ", gptr_str(prev_gptr), " ==> ",
                 gptr_str(next_gptr), "\n");
    if (next_gptr.where() > rank_me()) {
      DBG_TRAVERSE(padding, "DROP: owner ", next_gptr.where(), " > ", rank_me(), "\n");
      return false;
    }
    if (visited.find(next_gptr) != visited.end()) {
      DBG_TRAVERSE(padding, "REPEAT: ", gptr_str(next_gptr), "\n");
      num_repeats++;
      return true;
    }
    visited[next_gptr] = true;
    FragElem next_frag_elem = rget(next_gptr).wait();
    if (next_gptr.where() == rank_me()) {
      if (next_frag_elem.visited) DIE("gptr ", next_gptr, " should not be already visited");
      my_frag_elems_visited.push_back(next_gptr.local());
    }
    string next_frag_seq = get_frag_seq(next_frag_elem);
    string next_frag_seq_rc = revcomp(next_frag_seq);
    if (dirn == Dirn::NONE) {
      if (is_overlap(uutig, next_frag_seq, kmer_len - 1))
        dirn = Dirn::RIGHT;
      else if (is_overlap(next_frag_seq, uutig, kmer_len - 1))
        dirn = Dirn::LEFT;
      if (dirn == Dirn::NONE) {
        if (is_overlap(uutig, next_frag_seq_rc, kmer_len - 1))
          dirn = Dirn::RIGHT;
        else if (is_overlap(next_frag_seq_rc, uutig, kmer_len - 1))
          dirn = Dirn::LEFT;
        else
          DIE("No overlap");
      }
      DBG_TRAVERSE(padding, "Direction is set to ", DIRN_STR(dirn), "\n");
    }
    if (dirn == Dirn::LEFT) {
      int slen = next_frag_seq.length() - kmer_len + 1;
      DBG_TRAVERSE(string(slen, ' '), uutig, "\n");
      if (is_overlap(next_frag_seq, uutig, kmer_len - 1))
        uutig.insert(0, substr_view(next_frag_seq, 0, slen));
      else if (is_overlap(next_frag_seq_rc, uutig, kmer_len - 1))
        uutig.insert(0, substr_view(next_frag_seq_rc, 0, slen));
      else
        DIE("No valid overlap in dirn ", DIRN_STR(dirn));
    } else {
      if (is_overlap(uutig, next_frag_seq, kmer_len - 1))
        uutig += substr_view(next_frag_seq, kmer_len - 1);
      else if (is_overlap(uutig, next_frag_seq_rc, kmer_len - 1))
        uutig += substr_view(next_frag_seq_rc, kmer_len - 1);
      else
        DIE("No valid overlap in dirn ", DIRN_STR(dirn));
    }
    DBG_TRAVERSE(uutig, "\n");
    depths += (next_frag_elem.sum_depths * (1.0 - (kmer_len - 1) / next_frag_elem.frag_len));
    auto other_side_gptr = get_other_side_gptr(next_frag_elem, prev_gptr);
    prev_frag_elem = next_frag_elem;
    prev_gptr = next_gptr;
    next_gptr = other_side_gptr;
#ifdef DEBUG
    padding += string(4, ' ');
#endif
    walk_steps++;
  }
  DBG_TRAVERSE(padding, "DEADEND\n");
  return true;
}

template <int MAX_K>
static void connect_frags(unsigned kmer_len, dist_object<KmerDHT<MAX_K>> &kmer_dht, vector<global_ptr<FragElem>> &frag_elems,
                          Contigs &my_uutigs) {
  BarrierTimer timer(__FILEFUNC__);
  int64_t num_steps = 0, max_steps = 0, num_drops = 0, num_prev_visited = 0, num_repeats = 0;
  ProgressBar progbar(frag_elems.size(), "Connecting fragments");
  for (auto frag_elem_gptr : frag_elems) {
    progbar.update();
    FragElem *frag_elem = frag_elem_gptr.local();
    if (frag_elem->frag_len < kmer_len) continue;
    if (frag_elem->visited) {
      num_prev_visited++;
      continue;
    }
    vector<FragElem *> my_frag_elems_visited;
    string uutig(frag_elem->frag_seq.local());
    int64_t depths = frag_elem->sum_depths;
    int64_t walk_steps = 1;
    bool walk_ok = walk_frags_dirn(kmer_len, frag_elem_gptr, frag_elem->left_gptr, uutig, depths, walk_steps, num_repeats,
                                   my_frag_elems_visited);
    if (walk_ok)
      walk_ok = walk_frags_dirn(kmer_len, frag_elem_gptr, frag_elem->right_gptr, uutig, depths, walk_steps, num_repeats,
                                my_frag_elems_visited);
    if (walk_ok) {
      num_steps += walk_steps;
      max_steps = max(walk_steps, max_steps);
      my_uutigs.add_contig({0, uutig, (double)depths / (uutig.length() - kmer_len + 2)});
      // the walk is successful, so set the visited for all the local elems
      for (auto &elem : my_frag_elems_visited) elem->visited = true;
    } else {
      num_drops++;
    }
  }
  progbar.done();
  auto all_num_steps = reduce_one(num_steps, op_fast_add, 0).wait();
  auto all_max_steps = reduce_one(max_steps, op_fast_max, 0).wait();
  auto all_num_drops = reduce_one(num_drops, op_fast_add, 0).wait();
  auto all_num_repeats = reduce_one(num_repeats, op_fast_add, 0).wait();
  auto all_num_uutigs = reduce_one(my_uutigs.size(), op_fast_add, 0).wait();
  SLOG_VERBOSE("Constructed ", all_num_uutigs, " uutigs with ", (double)all_num_steps / all_num_uutigs, " avg path length (max ",
               all_max_steps, "), dropped ", perc_str(all_num_drops, all_num_uutigs), " paths\n");
  auto all_num_prev_visited = reduce_one(num_prev_visited, op_fast_add, 0).wait();
  auto all_num_frags = reduce_one(frag_elems.size(), op_fast_add, 0).wait();
  SLOG_VERBOSE("Skipped ", perc_str(all_num_prev_visited, all_num_frags), " already visited fragments, and found ",
               perc_str(all_num_repeats, all_num_frags), " repeats\n");
  barrier();
  for (auto frag_elem_gptr : frag_elems) {
    FragElem *frag_elem = frag_elem_gptr.local();
    delete_array(frag_elem->frag_seq);
    delete_(frag_elem_gptr);
  }
}

template <int MAX_K>
void traverse_debruijn_graph(unsigned kmer_len, dist_object<KmerDHT<MAX_K>> &kmer_dht, Contigs &my_uutigs) {
  BarrierTimer timer(__FILEFUNC__);
  {
    // scope for frag_elems
    vector<global_ptr<FragElem>> frag_elems;
    construct_frags(kmer_len, kmer_dht, frag_elems);
    clean_frag_links(kmer_len, kmer_dht, frag_elems);
    // put all the uutigs found by this rank into my_uutigs
    my_uutigs.clear();
    connect_frags(kmer_len, kmer_dht, frag_elems, my_uutigs);
  }
  // now get unique ids for the uutigs
  auto num_ctgs = my_uutigs.size();
  auto fut = upcxx_utils::reduce_prefix(num_ctgs, upcxx::op_fast_add).then([num_ctgs, &my_uutigs](size_t my_prefix) {
    auto my_counter = my_prefix - num_ctgs;  // get my start
    for (auto it = my_uutigs.begin(); it != my_uutigs.end(); it++) it->id = my_counter++;
  });
  fut.wait();
  barrier();
#ifdef DEBUG
  ProgressBar progbar(my_uutigs.size(), "Checking kmers in uutigs");
  for (auto uutig : my_uutigs) {
    progbar.update();
    if (!check_kmers(uutig.seq, kmer_dht, kmer_len)) DIE("kmer not found in uutig");
  }
  progbar.done();
#endif
}

#define TDG_K(KMER_LEN) \
  template void traverse_debruijn_graph<KMER_LEN>(unsigned kmer_len, dist_object<KmerDHT<KMER_LEN>> &kmer_dht, Contigs &my_uutigs)

TDG_K(32);
#if MAX_BUILD_KMER >= 64
TDG_K(64);
#endif
#if MAX_BUILD_KMER >= 96
TDG_K(96);
#endif
#if MAX_BUILD_KMER >= 128
TDG_K(128);
#endif
#if MAX_BUILD_KMER >= 160
TDG_K(160);
#endif

#undef TDG_K

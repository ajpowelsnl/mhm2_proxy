#pragma once

#include <algorithm>
#include <atomic>
#include <iterator>
#include <limits>
#include <string>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/Allocators.hpp"
#include "upcxx_utils/bin_hash.hpp"
#include "upcxx_utils/flat_aggr_store.hpp"
#include "upcxx_utils/log.hpp"
#include "upcxx_utils/shared_array.hpp"
#include "upcxx_utils/split_rank.hpp"

using std::atomic;
using std::numeric_limits;
using std::pair;
using std::string;
using std::to_string;

using upcxx::dist_object;
using upcxx::global_ptr;
using upcxx::intrank_t;
using upcxx::make_future;
using upcxx::make_view;
using upcxx::op_fast_add;
using upcxx::op_fast_max;
using upcxx::progress;
using upcxx::rank_me;
using upcxx::rank_n;
using upcxx::reduce_all;
using upcxx::reduce_one;
using upcxx::rpc;
using upcxx::to_global_ptr;
using upcxx::view;
using upcxx::when_all;

#ifndef FLAT_MODE_NODES
#define FLAT_MODE_NODES 32  // jobs with less than 32 nodes will just use the lower overhead FlatAggrStore
#endif

namespace upcxx_utils {

using TT_SIZE_T = uint16_t;

// order(N). modifies SharedArray2 overwriting the second integer type
// as an offset-based linked list grouped by rank
// assumes the input of second type ranges from 0 to sort_team().rank_n() - 1;
// includes the size counts vector

template <typename T, typename I = TT_SIZE_T>
class PairVector : public std::pair<vector<T>, vector<I>> {
 public:
  using P = std::pair<T, I>;
  PairVector(size_t capacity = 0) { reserve(capacity); }
  size_t size() const {
    assert(this->first.size() == this->second.size());
    return this->first.size();
  }
  size_t capacity() const {
    assert(this->first.capacity() == this->second.capacity());
    return this->first.capacity();
  }
  void reserve(size_t c) {
    this->first.reserve(c);
    this->second.reserve(c);
  }
  void reset() {
    this->first.resize(0);
    this->second.resize(0);
  }
  const T *begin() const { return this->first.data(); }
  const T *end() const { return begin() + size(); }
  T *begin() { return const_cast<T *>(((const PairVector *)this)->begin()); }
  T *end() { return const_cast<T *>(((const PairVector *)this)->end()); }
  const I *begin2() const { return this->second.data(); }
  const I *end2() const { return begin2() + size(); }
  I *begin2() { return const_cast<I *>(((const PairVector *)this)->begin2()); }
  I *end2() { return const_cast<I *>(((const PairVector *)this)->end2()); }
};

template <typename T, typename I = TT_SIZE_T, typename SA = SharedArray2<T, I>>
class SortedSharedArray {
 protected:
  SA &sa;
  const upcxx::team &_sort_team;
  vector<I> starts, counts;

  void sort_sa() {
    size_t capacity = sa.capacity();
    if (capacity > std::numeric_limits<I>::max() || sa.size() > std::numeric_limits<I>::max())
      throw runtime_error("Too many entries in SharedArray to be sorted");
    starts.resize(sort_team().rank_n(), capacity);
    counts.resize(sort_team().rank_n(), 0);
    vector<I> ptrs;
    ptrs.resize(sort_team().rank_n(), capacity);

    I totalCount = sa.size();
    DBG_VERBOSE("sorting capacity=", capacity, " totalCount=", totalCount, "\n");
    assert(totalCount <= capacity);
    auto rank_ptr = sa.begin2();
    for (I i = 0; i < totalCount; i++) {
      I &rank = rank_ptr[i];
      DBG_VERBOSE2("i=", i, ", rank=", rank, ", key=", (int)*((char *)&(*(sa.begin() + i))), "'", *((char *)&(*(sa.begin() + i))),
                   "'\n");
      assert(rank >= 0);
      assert((intrank_t)rank < sort_team().rank_n());
      counts[rank]++;
      if (starts[rank] == (I)capacity) {
        assert(ptrs[rank] == (I)capacity);
        starts[rank] = i;
      } else {
        I last = ptrs[rank];
        assert(last < totalCount);
        rank_ptr[last] = i;
      }
      ptrs[rank] = i;
      rank_ptr[i] = (I)capacity;
    }
#ifdef DEBUG_VERBOSE
    for (I rank = 0; rank < ptrs.size(); rank++) {
      DBG_VERBOSE2("rank=", rank, ", ptrs[rank]=", ptrs[rank], ", starts[rank]=", starts[rank], ", counts[rank]=", counts[rank],
                   "\n");
    }
    for (I i = 0; i < totalCount; i++) {
      DBG_VERBOSE2("rank_ptr[", i, "]=", rank_ptr[i], "\n");
    }
#endif
  }

 public:
  SortedSharedArray(SA &sa_, const upcxx::team &sort_team_ = local_team())
      : sa(sa_)
      , _sort_team(sort_team_)
      , starts({})
      , counts({}) {
    sort_sa();
  }

  const upcxx::team &sort_team() const { return _sort_team; }
  const SA &get_SharedArray() const { return sa; }
  const vector<I> &get_starts() const { return starts; }
  const vector<I> &get_counts() const { return counts; }

  size_t size() const { return sa.size(); }

  class iter : public std::iterator<std::forward_iterator_tag, T> {
    using SSA = SortedSharedArray;
    using StartIter = typename vector<I>::iterator;
    SSA &ssa;
    I offset;
    StartIter start;

   public:
    iter(SSA &ssa_, I offset_, StartIter start_)
        : ssa(ssa_)
        , offset(offset_)
        , start(start_) {
      assert(offset <= ssa.sa.capacity());
    }

    const T &operator*() const {
      assert(offset < ssa.sa.capacity());
      assert(offset < ssa.sa.size());
      const T &ret = *(ssa.sa.begin() + offset);
      // DBG_VERBOSE("operator*:", this, " - ", (int) *((const char*)&ret), "'", *((const char*)&ret), "'\n");
      return ret;
    }

    T &operator*() { return const_cast<T &>(*(*((const iter *)this))); }

    I get_offset() const { return offset; }

    const I &get_second() const { return *(ssa.sa.begin2() + offset); }

    const T *operator->() const { return &(*this); }

    T *operator->() { return const_cast<T *>(&(*((const iter *)this))); }

    bool operator==(const iter &other) const {
      // DBG_VERBOSE("operator==: ", *this, "\n");
      return ssa.sa.begin() == other.ssa.sa.begin() && offset == other.offset && start == other.start;
    }

    bool operator!=(const iter &other) const { return !(*this == other); }

    // ++iter

    iter &operator++() {
      auto capacity = ssa.sa.capacity();
      if (offset == capacity) throw runtime_error("Increment past capacity");
      if (offset >= ssa.sa.size()) throw runtime_error("Increment past size");
      // get next pointer
      offset = *(ssa.sa.begin2() + offset);
      if (offset == capacity) {
        next_start();
      }
      assert(offset <= capacity);
      return *this;
    }

    // iter++

    iter operator++(int) {
      iter last_iter = iter(ssa, offset, start);
      ++(*this);
      return last_iter;
    }

    // get the next start, skipping any empty starts
    void next_start() {
      while (offset == ssa.sa.capacity() && start != ssa.starts.end()) {
        offset = *(start++);
      }
    }

    friend std::ostream &operator<<(std::ostream &os, const iter &it) {
      os << "SortedSharedArray::iter(this=" << &it << ": offset=" << it.offset
         << ", start(idx)=" << it.start - it.ssa.get_starts().begin() << "of" << it.ssa.get_starts().size()
         << ", capacity=" << it.ssa.get_SharedArray().capacity() << ")";
      if (it.offset != it.ssa.get_SharedArray().capacity()) {
        assert(it.offset < it.ssa.get_SharedArray().size());
        os << " - elem=" << (int)*((char *)&(*it)) << "'" << *((char *)&(*it))
           << "', thread=" << *(it.ssa.get_SharedArray().begin2() + it.offset) << ".";
      }
      return os;
    }
  };

  iter begin() {
    iter it(*this, sa.capacity(), starts.begin());
    it.next_start();
    return it;
  }

  iter end() { return iter(*this, sa.capacity(), starts.end()); }

  friend std::ostream &operator<<(std::ostream &os, const SortedSharedArray &ssa) {
    os << "SortedSharedArray(this=" << &ssa << ": sa=" << ssa.sa << ", starts=" << ssa.starts << " counts=" << ssa.counts << ")";
    return os;
  }
};

struct TT_RPC_Counts {
  using CountType = TargetRPCCounts::CountType;

  TT_RPC_Counts();
  void reset();
  atomic<CountType> rpcs_sent;
  atomic<CountType> rpcs_expected;
  atomic<CountType> rpcs_processed;
  atomic<CountType> rpcs_progressed;
  atomic<size_t> imbalance_factor;
};

struct TT_All_RPC_Counts;
using TTDistRPCCounts = dist_object<TT_All_RPC_Counts>;
struct TT_All_RPC_Counts {
  using CountType = TargetRPCCounts::CountType;
  TT_All_RPC_Counts() = delete;
  TT_All_RPC_Counts(DistFASRPCCounts &dist_fas_rpc_counts);

  virtual ~TT_All_RPC_Counts();
  void init(const split_team &splits);
  void reset(bool free = false);
  void print_out();
  void increment_sent_counters(intrank_t target_node);
  void increment_processed_counters(intrank_t source_node);
  void set_progressed_count(intrank_t source_node, CountType num_progressed);
  void wait_for_rpcs(intrank_t target_node, CountType max_rpcs_in_flight);
  void update_progressed_count(TTDistRPCCounts &dist_tt_rpc_counts, intrank_t target_node);

  global_ptr<TT_RPC_Counts> total;
  vector<global_ptr<TT_RPC_Counts>> targets;
  
  upcxx::future<> inner_rpc_future;
  DistFASRPCCounts &fas_rpc_counts;  // reference to flat_aggr_store dist_object counts (but using a different (local) team)
};

// this class extends FlatAggrStore
// it aggregates updates into local per-node buffers and then periodically does an rpc to dispatch them
// all aggregation for the thread_team / local_team are within the FlatAggrStore scope
// all aggregation between nodes is handled by ThreeTierAggrStore

template <typename T, typename... Data>
class ThreeTierAggrStore : public FlatAggrStore<T, Data...> {
  // The three levels are 1: n =  MY_LOCAL_TEAM (i.e. thread_team()), 2: N = REMOTE_NODE, 3: n = REMOTE_LOCAL_TEAM
  // any RPC between local team will use the FlatAggrStore functions directly

  // Each rank has N private memory microblocks (1 for each node).  This is small and should fit well within L2 cache
  //   microblocks may be 0 or >=2 in size, thus every addition to the shared message_block would require atomic offset
  // Each rank in MY_LOCAL_TEAM has (N / n) + 1 message blocks.  This block is large keeping network messages large
  // on update to an off-node rank, a rank's private microblock is appended for that node
  //   if that microblock is full
  //     while ((pos = message_block.size.atomic_fadd(microblock.size)) < 0) progress()
  //     if pos is not full but pos+size is (over)full,
  //        copy/append/fill up to capacity
  //        block on progress() until second count is expected value
  //        set pos very negative
  //        sort message_block by thread_team rank
  //        send rpc of existing messageblock
  //        return pos to 0.
  //        on return of rpc, free message_block memory
  //     if pos is not full and pos+size still has room
  //       copy/append micoblock
  //       atomic fadd second count
  //
  // within rpc a message_block
  // allocate local message_block_memory
  // copy data to local
  // return global data pointer
  // send n rpcs to local team to process_local a partitions of the data
  // when all rpcs have finished free local message_block memory

  static const bool is_trivial = std::is_trivial<T>::value;

 public:
  using FAS = FlatAggrStore<T, Data...>;
  // threads here are really ranks within a (possible subset of) local_team()
  using SharedArrayStore = SharedArray2<T, TT_SIZE_T>;
  using NodeStore = global_ptr<SharedArrayStore>;
  using TTStore = vector<NodeStore>;

  using MicroBlock = PairVector<T, TT_SIZE_T>;
  using MicroBlockStore = vector<MicroBlock>;

  using UpdateFunc = typename FAS::UpdateFunc;
  using DistUpdateFunc = typename FAS::DistUpdateFunc;
  using CountType = typename FAS::CountType;
  using DistRPCCounts = typename FAS::DistRPCCounts;

  using DistSplitTeam = dist_object<split_team>;

 protected:
  TTStore tt_store;
  MicroBlockStore tt_micro_store;
  DistPoolAllocator<T> tt_pool_allocator;

  CountType tt_max_rpcs_in_flight;
  CountType tt_max_store_size_per_node;
  CountType tt_max_micro_store_size_per_node;
  CountType tt_updates;

  TTDistRPCCounts tt_rpc_counts;

  DistSplitTeam splits;

  void tt_barrier() { barrier(splits->full_team()); }

  static void tt_wait_for_rpcs(ThreeTierAggrStore *astore, node_num_t target_node) {
    assert(target_node < astore->splits->node_n());
    auto &counts = astore->tt_rpc_counts;
    auto &tgt_imbalance = counts->targets[target_node].local()->imbalance_factor;
    auto imbal = tgt_imbalance.load();
    counts->wait_for_rpcs(target_node, astore->tt_max_rpcs_in_flight);
    if (tgt_imbalance.load() > imbal) {
      // query target for its progress count
      counts->update_progressed_count(counts, target_node);
    }
    counts->increment_sent_counters(target_node);
  }

  // operates on a single element (within the full_team i.e. directly to the controlling rank, not the node)
  static void tt_update_remote1(ThreeTierAggrStore *astore, intrank_t full_rank, const T &elem, Data &... data) {
    assert(full_rank < astore->splits->full_n() && "within full_team");
    assert(((FAS *)astore)->get_team().from_world(astore->splits->full_team()[full_rank], rank_n()) == rank_n() &&
           "not a rank covered by flat_aggr_store");
    node_num_t target_node = astore->splits->node_from_full(full_rank);
    DBG_VERBOSE("tt_update_remote1() full_rank = ", full_rank, ", target_node=", target_node, "\n");
    if (astore->splits->full_me() == full_rank) {
      // bypass self
      (*astore->update_func)(elem, data...);
      return;
    }
    tt_wait_for_rpcs(astore, target_node);
    auto progressed_count =
        astore->tt_rpc_counts->targets[astore->splits->node_from_full(full_rank)].local()->rpcs_processed.load();
    rpc_ff(astore->splits->full_team(), full_rank,
           [](DistUpdateFunc &update_func, T elem, TTDistRPCCounts &tt_rpc_counts, node_num_t source_node,
              CountType progressed_count, Data &... data) {
             
             DBG_VERBOSE("tt_update_remote1()::rpc_ff source_node = ", source_node,
                         ", already_processed=", (*tt_rpc_counts).targets[source_node].local()->rpcs_processed.load(), "\n");
             tt_rpc_counts->set_progressed_count(source_node, progressed_count);
             (*update_func)(elem, data...);
             tt_rpc_counts->increment_processed_counters(source_node);
             
           },
           astore->update_func, elem, astore->tt_rpc_counts, astore->splits->node_me(), progressed_count, data...);
    
  }

  template <typename SATYPE = SharedArray2<T, TT_SIZE_T>>
  static void tt_remote_rpc_ff(ThreeTierAggrStore *astore, node_num_t target_node,
                               SortedSharedArray<T, TT_SIZE_T, SATYPE> &sorted_array, Data &... data) {
    tt_wait_for_rpcs(astore, target_node);
    auto progressed_count = astore->tt_rpc_counts->targets[target_node].local()->rpcs_processed.load();
    auto &counts = sorted_array.get_counts();

    
    rpc_ff(
        astore->splits->node_team(), target_node,
        [](DistUpdateFunc &update_func, view<T> node_store_view, view<TT_SIZE_T> thread_sizes, TTDistRPCCounts &tt_rpc_counts,
           node_num_t source_node, CountType progressed_count, DistSplitTeam &splits, Data &... data) {
          DBG_VERBOSE("tt_update_remote_ff::rpc_ff() source_node=", source_node, " node_store_view.size=", node_store_view.size(),
                      ", already_processed=", tt_rpc_counts->targets[source_node].local()->rpcs_processed.load(), "\n");
          // calculate offsets within node_store_view
          assert(thread_sizes.size() == splits->thread_n());
          assert(!node_store_view.empty());
          tt_rpc_counts->set_progressed_count(source_node, progressed_count);

          // Double View of this relay/scatter

          

          vector<TT_SIZE_T> thread_offsets(splits->thread_n(), 0);
          for (thread_num_t i = 1; i < splits->thread_n(); i++) {
            thread_offsets[i] = thread_sizes[i - 1] + thread_offsets[i - 1];
          }

          // call multiple FlatAggrStore::update_remote_rpc() for each thread in the local team
          auto start_iter = node_store_view.begin(), end_iter = node_store_view.end();
          auto thread_iter = start_iter;

          auto &rpc_counts = tt_rpc_counts->fas_rpc_counts;
          const auto &thread_team = splits->thread_team();
          upcxx::future<> relay_done_fut = make_future();
          size_t offset = 0;

          // do not stagger the ranks since the view is in order by rank and not not randomly accessible
          for (thread_num_t thread_num = 0; thread_num < splits->thread_n(); thread_num++) {
            assert(offset == thread_offsets[thread_num]);
            if (thread_sizes[thread_num] == 0) {
              DBG_VERBOSE("Nothing for thread_num=", thread_num, "\n");
              continue;
            }

            if (thread_num == splits->thread_me()) {
              // bypass rpc on self
              DBG_VERBOSE("Bypass to self thread_num=", thread_num, ", size=", thread_sizes[thread_num], " offset=", offset, "\n");
              for (TT_SIZE_T i = 0; i < thread_sizes[thread_num]; i++) {
                assert(thread_iter != node_store_view.end());
                const auto &elem = *(thread_iter++);
                (*update_func)(elem, data...);
              }
            } else {
              DBG_VERBOSE("Sending to thread_num=", thread_num, ", size=", thread_sizes[thread_num], " offset=", offset, "\n");

              FAS::increment_rpc_counters(rpc_counts, thread_num);
              
              auto send_start_iter = thread_iter;
              for (size_t i = 0; i < thread_sizes[thread_num]; i++) {
                assert(thread_iter != node_store_view.end());
                thread_iter++;
              }
              auto send_end_iter = thread_iter;

              // The Double View version of this relay/scatter

              // make another view to ensure lifetime of the thread slice is preserved
              auto elems_view = make_view(send_start_iter, send_end_iter, thread_sizes[thread_num]);

              upcxx::future<> fut = rpc(thread_team, thread_num, FAS::rpc_update_func, update_func, elems_view,
#ifdef USE_HH
                                 hh,
#endif
                                 rpc_counts, thread_team.rank_me(), rpc_counts->targets[thread_num].rpcs_processed, data...);
              
              relay_done_fut = when_all(relay_done_fut, fut);
            }

            offset += thread_sizes[thread_num];
          }
          assert(thread_iter == end_iter);

          
          relay_done_fut = relay_done_fut.then([&tt_rpc_counts, source_node]() {
            tt_rpc_counts->increment_processed_counters(source_node);
            DBG_VERBOSE("Finished inner tt_remote_rpc_ff from ", source_node, "\n");
          });
          tt_rpc_counts->inner_rpc_future = when_all(tt_rpc_counts->inner_rpc_future, relay_done_fut);

          return;  // release full node_store_view immediately on exit here
        },
        astore->update_func, make_view(sorted_array.begin(), sorted_array.end(), sorted_array.size()),
        make_view(counts.begin(), counts.end()), astore->tt_rpc_counts, astore->splits->node_me(), progressed_count, astore->splits,
        data...);

    
  }

  // operates on a vector of elements in the store

  static void tt_update_remote(ThreeTierAggrStore *astore, node_num_t target_node, Data &... data) {
    
    assert(target_node < astore->splits->node_n());
    assert(astore->tt_pool_allocator && "Pool allocator is intialized");
    assert(target_node != astore->splits->node_me() && "Not a node covered by FlatAggrStore");
    DBG_VERBOSE("tt_update_remote() target_node = ", target_node, "\n");

    NodeStore &nodestore_ptr = astore->tt_store[target_node];

    assert(!nodestore_ptr.is_null());
    assert(nodestore_ptr.is_local());
    SharedArrayStore &nodestore = *(nodestore_ptr.local());

#ifdef USE_HH
    // TODO support HHSS which may still have entries when nodestore is empty...
#endif

    if (nodestore.empty())       return;
    
    // first allocate a new pointer there should always be >=1 available in the local team
    // but there may be races to find it
    global_ptr<T> new_ptr;
    do {
      new_ptr = astore->tt_pool_allocator.allocate(astore->splits->thread_team());
      if (!new_ptr.is_null()) break;
      
    } while (true);

    // now wait for all appends to complete, this likely has already happened
      int i=0;
   
      while (!nodestore.ready()) i+=0;
      assert(nodestore.ready());

    // swap the pointers and reset NodeStore for other ranks to use asap
    global_ptr<T> full_ptr = nodestore.get_global_ptr();
    auto full_size = nodestore.set_new_ptr(new_ptr);

    // "copy" the state of the old store to this local variable
    SharedArrayStore full_store(nodestore.capacity());
    assert(full_size <= full_store.capacity());
    full_store.set_new_ptr(full_ptr, full_size);
#ifdef DEBUG_VERBOSE
    for (int i = 0; i < full_size; i++) {
      auto &f = *(full_store.begin() + i);
      auto &s = *(full_store.begin2() + i);
      DBG_VERBOSE("full_store: ", (int)*((char *)&f), "'", *((char *)&f), "', idx=", s, "\n");
    }
#endif
    // sort by local rank, transform to counts and an iterator for view<T>
    SortedSharedArray<T, TT_SIZE_T> sorted_array(full_store, astore->splits->thread_team());
    auto &counts = sorted_array.get_counts();
    assert((intrank_t)counts.size() == astore->splits->thread_team().rank_n());
#ifdef DEBUG_VERBOSE
    int i = 0;
    for (auto it = sorted_array.begin(); it != sorted_array.end(); it++) {
      auto &f = *it;
      DBG_VERBOSE("sorted_store: i=", i, ", val=", (int)*((char *)&f), "'", *((char *)&f), "', idx=", it.get_second(), "\n");
      i++;
    }
#endif
    
    // send the rpc
    tt_remote_rpc_ff(astore, target_node, sorted_array, data...);

    // return the full_ptr back to the pool
    astore->tt_pool_allocator.deallocate(full_ptr);
  }

  template <typename iter1, typename iter2>
  void append_shared_store(node_num_t target_node, iter1 begin1, iter2 begin2, size_t size) {
    
    DBG_VERBOSE("3TAS::append_shared_store() target_node=", target_node, " size=", size, " node_store=", tt_store[target_node],
                "\n");
    SharedArrayStore &node_store = *(tt_store[target_node].local());
    size_t appended_len, offset = 0;
    do {
      assert(size - offset > 0);
      appended_len = node_store.append(begin1 + offset, begin2 + offset, size - offset);
      if (appended_len == size - offset + 1) {
        // appended all remaining and SharedArrayStore is not full
        break;
      } else {
        
        if (appended_len == 0) {
          // node_store is full and can not proceed.  Some other rank is working to resolve it already
            int k=0;
        } else {
          assert(appended_len <= size - offset);
          // send this full NodeStore
          std::apply(tt_update_remote, std::tuple_cat(std::make_tuple(this, target_node), this->data));
          offset += appended_len;
          // continue and append remainder, if any
        }
        
      }
    } while (size - offset > 0);
    
  }

  void append_shared_store(node_num_t node_num) {
    auto &micro_store = tt_micro_store[node_num];
    assert(micro_store.first.size() == micro_store.second.size());
    assert(micro_store.first.size() <= tt_max_micro_store_size_per_node);
    assert(!micro_store.first.empty());
    if (tt_max_micro_store_size_per_node >= tt_max_store_size_per_node) {
      DBG("Sorting micro store:", micro_store.first.size(), "/", micro_store.second.size(), ", ", micro_store.second.front(), "\n");
      // bypass append to store when in TwoTier mode
      // sort micro_store
      SortedSharedArray<T, TT_SIZE_T, MicroBlock> sorted_array(micro_store, splits->thread_team());
      // send the rpc
      std::apply(tt_remote_rpc_ff<MicroBlock>, std::tuple_cat(std::forward_as_tuple(this, node_num, sorted_array), this->data));
    } else {
      //  append micro_store to the thread_team's node_store
      append_shared_store(node_num, micro_store.first.begin(), micro_store.second.begin(), micro_store.first.size());
    }
    micro_store.first.resize(0);
    micro_store.second.resize(0);
  }

  static bool use_flat_only() {
#ifdef DEBUG
    return false;  // need to test 3Tier
#else
    return rank_n() / local_team().rank_n() < FLAT_MODE_NODES;
#endif
  }

 public:
  ThreeTierAggrStore(Data &... data)
      : FAS(use_flat_only() ? world() : split_rank::split_local_team(), data...)
      , tt_store({})
      , tt_micro_store({})
      , tt_pool_allocator(split_rank::split_local_team())
      , tt_max_rpcs_in_flight(0)
      , tt_max_store_size_per_node(0)
      , tt_max_micro_store_size_per_node(0)
      , tt_updates(0)
      , tt_rpc_counts(world(), FAS::rpc_counts)
      , splits(world(), world(), use_flat_only() ? world() : split_rank::split_local_team()) {
    DBG("created 3TAS. tt_pool_allocator=", tt_pool_allocator, " ", *tt_pool_allocator, "\n");
  }
  ThreeTierAggrStore(const upcxx::team &team, Data &... data)
      : FAS(use_flat_only() ? team : split_rank::split_local_team(), data...)
      , tt_store({})
      , tt_micro_store({})
      , tt_pool_allocator(split_rank::split_local_team())
      , tt_max_rpcs_in_flight(0)
      , tt_max_store_size_per_node(0)
      , tt_max_micro_store_size_per_node(0)
      , tt_updates(0)
      , tt_rpc_counts(team, FAS::rpc_counts)
      , splits(team, team, use_flat_only() ? team : split_rank::split_local_team()) {
    DBG("created 3TAS with team. tt_pool_allocator=", tt_pool_allocator, " ", *tt_pool_allocator, "\n");
  }

  virtual ~ThreeTierAggrStore() {
    DBG("destroying 3TAS. tt_pool_allocator=", tt_pool_allocator, " ", *tt_pool_allocator, "\n");
    clear();
    DBG("cleared 3TAS. tt_pool_allocator=", tt_pool_allocator, " ", *tt_pool_allocator, "\n");
  }

  void set_update_func(UpdateFunc update_func) {
    DBG("\n");
    tt_barrier();  // to avoid race of first update
    if ((tt_max_micro_store_size_per_node > 0) | (tt_max_store_size_per_node > 0)) {
      if (tt_rpc_counts->total.is_null() || (tt_max_micro_store_size_per_node > 0 && tt_micro_store.empty()) ||
          (tt_max_store_size_per_node > 0 && tt_store.empty())) {
        DIE("Invalid condition - ThreeTierAggrStore not initialized yet - call set_size() first after construction or clear()! "
            "total=",
            tt_rpc_counts->total, ", microsize=", tt_max_micro_store_size_per_node, ", micro_storesize=", tt_micro_store.size(),
            ", max_store_size=", tt_max_store_size_per_node, ", store.size()=", tt_store.size(), "\n");
      }
    }
    ((FAS *)this)->set_update_func(update_func);
    tt_barrier();  // to avoid race of first update
  }

  // TODO implement TwoTier mode where micro_stores are larger and are the only stores, so no append with threads
  void set_size(const string &desc, CountType max_store_bytes, CountType max_rpcs_in_flight = 128, bool use_heavy_hitters = true) {
    tt_rpc_counts->reset(true);
    this->description = desc;
    this->max_rpcs_in_flight = max_rpcs_in_flight;
    size_t max_message_size = 1 * 1024 * 1024 - 1024;  // 999KB

    node_num_t num_nodes = splits->node_n();
    thread_num_t num_local_threads = splits->thread_n();

    auto num_targets = num_nodes - 1;  // all but me
    size_t elem_size = sizeof(T) + sizeof(TT_SIZE_T);

    size_t min_micro_count = 10;
    size_t num_allocations_per_rank = min_micro_count;

    size_t flat_store_bytes = max_store_bytes;

    // on the local_team(), smaller messages can still be efficiently communicated
    // especially considering the bulk will be from within the relay rpc here
    // and relatively few updates through fas as scale increases
    size_t fas_allocations = (num_local_threads - 1) / 16;
    if (fas_allocations < 3) fas_allocations = 3;

    // three tier has more overhead, so drop down to two tier if that would be better
    // run two tier if the #nodes is low or there is sufficient memory to be within the max_message_size/4
    bool flat_mode = (num_nodes == 1) | use_flat_only();
    // Note 3Tier is not possible with non-trivial data types (like strings), but 2Tier should still be okay with
    // a double-view relay in the rpc execution lambda
    bool two_tier_mode = (!is_trivial) | (num_nodes < num_local_threads) |
                         ((max_store_bytes / elem_size / (num_targets + fas_allocations)) * sizeof(T) >= max_message_size / 4);
#ifdef NO_TWO_TIER_AS
    two_tier_mode = false;
    static_assert(is_trivial,
                  "The template type T must be trivially serializable because of the nature of aggregation along the local_team");
#endif
    if (flat_mode) {
      // single node -- use only flat_aggr_store
      tt_max_store_size_per_node = 0;
      tt_max_micro_store_size_per_node = 0;
      DBG("running in flat mode\n");
    } else if (two_tier_mode) {
      // medium scale drop down to 2 tiers (i.e. rank -> node messages) and avoid extra copy / aggregation (i.e. node -> node
      // messages) use only microstores with 1 entry per node (private for each rank)
      tt_max_store_size_per_node = 0;
      size_t num_2tier_allocations_per_rank = num_targets;                                       // each rank has its own store
      size_t total_num_allocations_per_rank = num_2tier_allocations_per_rank + fas_allocations;  // total including FlatAggrStore
      tt_max_micro_store_size_per_node = max_store_bytes / elem_size / total_num_allocations_per_rank;
      if (tt_max_micro_store_size_per_node * sizeof(T) > max_message_size) {
        // reduce the count to below max_message_size
        tt_max_micro_store_size_per_node = max_message_size / sizeof(T);
      }
      if (tt_max_micro_store_size_per_node > numeric_limits<TT_SIZE_T>::max()) {
        tt_max_micro_store_size_per_node = numeric_limits<TT_SIZE_T>::max();
      }
      if (tt_max_micro_store_size_per_node < min_micro_count) {
        SWARN("ThreeTierAggrStore (two tier mode) ", this->description, " max_store_size_per_target is small (",
              tt_max_store_size_per_node, ") please consider increasing the max_store_bytes (", get_size_str(max_store_bytes),
              ")\n");
        SWARN("at this scale of ", num_targets, " other nodes, at least ",
              get_size_str(min_micro_count * elem_size * total_num_allocations_per_rank), " is necessary for good performance\n");
        tt_max_micro_store_size_per_node = 0;  // no micro store
      } else {
        flat_store_bytes = tt_max_micro_store_size_per_node * sizeof(T) * fas_allocations;
      }
      num_allocations_per_rank = num_2tier_allocations_per_rank;
      DBG("Running in 2 tier mode\n");
    } else {
      // Three Tier Mode (large scales)
      assert(num_nodes > 1);
      if (!is_trivial) DIE("Three Tier mode is not available for non-trivially serializable data types");

      // allocate 5-15% of the max_store_bytes for microstores with 1 entry per node (private for each rank)
      tt_max_micro_store_size_per_node = max_store_bytes / 20 / num_targets / elem_size;
      if (tt_max_micro_store_size_per_node < min_micro_count &&
          min_micro_count * num_targets * elem_size <= 10 * max_store_bytes / 100) {
        // allow up to 10% of the max_store_bytes
        tt_max_micro_store_size_per_node = min_micro_count;
        SLOG_VERBOSE("5% micro store memory is too small, using 10% for a count of ", tt_max_micro_store_size_per_node, "\n");
      } else if (tt_max_micro_store_size_per_node < min_micro_count && 2 * num_targets * elem_size <= 15 * max_store_bytes / 100) {
        // at least 2 count takes less than 15% of the max_store_bytes.. it should still be worth it
        min_micro_count = 2;
        tt_max_micro_store_size_per_node = 15 * max_store_bytes / 100 / num_targets / elem_size;
        SLOG_VERBOSE("10% micro store memory is too small, using 15% for a count of ", tt_max_micro_store_size_per_node, "\n");
      }
      if (tt_max_micro_store_size_per_node * sizeof(T) > max_message_size) {
        // reduce the count to below 2MB messages
        tt_max_micro_store_size_per_node = max_message_size / sizeof(T);
      }
      if (tt_max_micro_store_size_per_node > numeric_limits<TT_SIZE_T>::max()) {
        tt_max_micro_store_size_per_node = numeric_limits<TT_SIZE_T>::max();
      }
      if (tt_max_micro_store_size_per_node < min_micro_count) {
        tt_max_micro_store_size_per_node = 0;  // no micro store
      } else {
        max_store_bytes -= tt_max_micro_store_size_per_node * num_targets * elem_size;
      }
      if (tt_max_micro_store_size_per_node == 0) {
        SLOG_VERBOSE("NOTICE!! ThreeTierAggrStore max_store_size_per_target is too small to support a micro_store. (requires ",
                     get_size_str(min_micro_count * num_nodes * elem_size * 20), ")\n");
      }
      DBG("tt_max_micro_store_size_per_node=", tt_max_micro_store_size_per_node, " max_store_bytes=", max_store_bytes,
          ", node_store_bytes=", max_store_bytes * local_team().rank_n(), " \n");

      size_t min_count = 3 * min_micro_count;

      size_t num_3tier_allocations_per_rank =
          2 + (num_targets + num_local_threads - 1) / num_local_threads;  // thread team shares single store + 2 extra
      size_t total_num_allocations_per_rank = num_3tier_allocations_per_rank + fas_allocations;  // total including FlatAggrStore
      tt_max_store_size_per_node = max_store_bytes / elem_size / total_num_allocations_per_rank;
      if (tt_max_store_size_per_node * sizeof(T) > max_message_size) {
        // reduce the count to below 2MB messages
        tt_max_store_size_per_node = max_message_size / sizeof(T);
      }
      if (tt_max_store_size_per_node > numeric_limits<TT_SIZE_T>::max()) {
        tt_max_store_size_per_node = numeric_limits<TT_SIZE_T>::max();
      }
      if (tt_max_store_size_per_node < min_count) {
        SWARN("ThreeTierAggrStore ", this->description, " max_store_size_per_target is small (", tt_max_store_size_per_node,
              ") please consider increasing the max_store_bytes (", get_size_str(max_store_bytes), ")\n");
        SWARN("at this scale of ", num_nodes, " nodes, at least ",
              get_size_str(min_count * elem_size * total_num_allocations_per_rank), " is necessary for good performance\n");
        tt_max_store_size_per_node = 0;
        tt_max_micro_store_size_per_node = 0;
      } else {
        flat_store_bytes =
            tt_max_store_size_per_node * sizeof(T) * fas_allocations;  // included within total_num_allocations_per_rank
      }
      num_allocations_per_rank = num_3tier_allocations_per_rank;
      DBG("running in 3 tier mode\n");
    }
    DBG("tt_max_store_size_per_node=", tt_max_store_size_per_node,
        ", tt_max_micro_store_size_per_node=", tt_max_micro_store_size_per_node, "\n");

    if (tt_max_store_size_per_node) {
      tt_store.resize(num_nodes);
      // chunk_count for allocator includes max store size for T *and* max store size for count TT_SIZE_T, in extra units of T
      size_t chunk_count =
          tt_max_store_size_per_node + (tt_max_store_size_per_node * sizeof(TT_SIZE_T) + sizeof(T) - 1) / sizeof(T);
      size_t num_chunks = num_allocations_per_rank;
      DBG("Creating pool_allocator\n");
      PoolAllocator<T> pool_allocator(num_chunks, chunk_count);
      DBG("Moving pool_allocator ", pool_allocator, " to DistPoolAllocator ", *tt_pool_allocator, "\n");
      tt_pool_allocator = std::move(pool_allocator);
      DBG("Done pool_allocator ", pool_allocator, " to DistPoolAllocator", *tt_pool_allocator, "\n");
      for (node_num_t node_idx = 0; node_idx < num_nodes; node_idx++) {
        if (node_idx == splits->node_me()) continue;  // do not allocate for my node
        thread_num_t tracking_thread = node_idx % num_local_threads;
        if (tracking_thread == splits->thread_me()) {
          auto new_ptr = tt_pool_allocator.allocate();
          DBG("Got new allocation for node_idx=", node_idx, ": ", new_ptr, " of ", tt_pool_allocator.get_chunk_count(), " count\n");
          assert(!new_ptr.is_null());
          tt_store[node_idx] = upcxx::new_<SharedArrayStore>(new_ptr, tt_max_store_size_per_node);
        }
        // all local ranks share the same NodeStore for a given target node
        tt_store[node_idx] = upcxx::broadcast(tt_store[node_idx], tracking_thread, splits->thread_team()).wait();
        DBG("Allocated tt_store[node_idx=", node_idx, "] tracking_thread=", tracking_thread, ", tt_store[]=", tt_store[node_idx],
            "\n");
        assert(tt_store[node_idx]);
        assert(tt_store[node_idx].is_local());
        assert(tt_store[node_idx].where() == splits->thread_team()[tracking_thread]);
      }
    }

    if (tt_max_micro_store_size_per_node) {
      DBG("initializing micro stores:",
          get_size_str(num_nodes * (sizeof(T) + sizeof(TT_SIZE_T)) * tt_max_micro_store_size_per_node), " per rank\n");
      tt_micro_store.resize(num_nodes);
      for (auto n = 0; n < num_nodes; n++) {
        if (n == splits->node_me()) continue;  // do not allocate for my node
        auto &ms = tt_micro_store[n];
        ms.first.reserve(tt_max_micro_store_size_per_node);
        ms.second.reserve(tt_max_micro_store_size_per_node);
      }
    }

    if (flat_mode) {
      SLOG_VERBOSE("Bypassing ThreeTierAggrStore ", this->description, " - using FlatAggrStore with ", splits->full_n(),
                   " ranks squashed to ", num_nodes, " nodes\n");
    } else {
      SLOG_VERBOSE(desc, ": using a ", (two_tier_mode ? "two" : "three"), " tier aggregating store for each off-node target (",
                   num_targets, ") of max ", get_size_str(max_store_bytes), " per aggregating rank (",
                   get_size_str(max_store_bytes * local_team().rank_n()), " per node)\n");
      SLOG_VERBOSE("  max messages of ", tt_max_store_size_per_node, " entries of ", get_size_str(sizeof(T)), " per target node (",
                   num_targets, " nodes) for ", get_size_str(sizeof(T) * tt_max_store_size_per_node), " messages ",
                   get_size_str(elem_size * tt_max_store_size_per_node * num_allocations_per_rank * local_team().rank_n()),
                   " node mem,\n");
      SLOG_VERBOSE("  micro_store ", tt_max_micro_store_size_per_node, " entries using ",
                   get_size_str(elem_size * tt_max_micro_store_size_per_node * num_targets), " per rank ",
                   get_size_str(elem_size * tt_max_micro_store_size_per_node * num_targets * local_team().rank_n()), " node mem\n");

      tt_rpc_counts->init(*splits);
    }

    SLOG_VERBOSE("  max RPCs in flight: ", (!max_rpcs_in_flight ? string("unlimited") : to_string(max_rpcs_in_flight)), "\n");

    // set the size of the underlying FlatAggrStore on the local team
    ((FAS *)this)->set_size(desc, flat_store_bytes, max_rpcs_in_flight, false);

#ifdef USE_HH
    if (use_heavy_hitters) {
      // allocate heavy hitters approx the size of the RankStore for one more node in the job
      hh_store.reserve(max_store_size_per_target * upcxx::local_team().rank_n());
    }
#endif
    tt_barrier();
  }

  void clear() {
    DBG("\n");

    // clear FlatAggrStore first
    ((FAS *)this)->clear();

    tt_barrier();

    for (auto &s_ptr : tt_store) {
      if (s_ptr.is_null()) continue;
      if (s_ptr.where() != rank_me()) continue;
      auto &s = *(s_ptr.local());
      if (!s.empty()) throw string("rank store is not empty!");
      auto gptr = s.get_global_ptr();
      assert(!gptr.is_null());
      tt_pool_allocator.deallocate(gptr);
      upcxx::delete_(s_ptr);
      s_ptr = {};
    }
    tt_barrier();
    TTStore().swap(tt_store);
    tt_rpc_counts->reset(true);
    MicroBlockStore().swap(tt_micro_store);
    PoolAllocator<T>().swap(*tt_pool_allocator);
#ifdef USE_HH
    hh_store.clear();
#endif
    tt_barrier();
  }

  const DistSplitTeam &get_split_team() const { return splits; }

  void update(intrank_t target_rank, const T &_elem) {
    // DBG_VERBOSE("TTAG::update(target_rank=", target_rank, " elem=", &_elem, " ", (int) *((char*)&_elem), "'", *((char*)&_elem),
    // "')\n");
    // thread team updates go to FlatAggrStore
    if (splits->thread_team_contains_full(target_rank)) {
      ((FAS *)this)->update(splits->thread_from_full(target_rank), _elem);
      return;
    }
    tt_updates++;
#ifdef USE_HH
    T elem;
    // heavy hitter bypass
    if (hh_store.update(target_rank, _elem, elem)) return;
#else
    const T &elem = _elem;
#endif
    if (tt_max_micro_store_size_per_node > 1) {
      

      TT_SIZE_T thread_num = splits->thread_from_full(target_rank);
      node_num_t node_num = splits->node_from_full(target_rank);

      assert(tt_micro_store.size() > node_num);
      auto &micro_store = tt_micro_store[node_num];

      assert(micro_store.first.size() < tt_max_micro_store_size_per_node);
      micro_store.first.push_back(elem);
      micro_store.second.push_back(thread_num);
     
      if (micro_store.first.size() == tt_max_micro_store_size_per_node) {
        assert(!micro_store.first.empty());
        append_shared_store(node_num);
        assert(micro_store.first.empty());
      } else {
        assert(micro_store.first.size() < tt_max_micro_store_size_per_node);
        
      }
    } else if (tt_max_store_size_per_node > 1) {
      TT_SIZE_T thread_num = splits->thread_from_full(target_rank);
      node_num_t node_num = splits->node_from_full(target_rank);
      // no micro_stores. append 1
      append_shared_store(node_num, &elem, &thread_num, 1);
    } else {
      assert(tt_max_store_size_per_node == 0);
      // DBG_VERBOSE("TTAG::update(target_rank=", target_rank, " update1 - elem=", &_elem, "'", *((char*)&_elem), "')\n");
      std::apply(tt_update_remote1, std::tuple_cat(std::forward_as_tuple(this, target_rank, elem), this->data));
    }
  }

  // flushes all pending updates
  // if no_wait is true, counts are not exchanged and quiescence has NOT been achieved
  // will always block on the thread team
  void flush_updates(bool no_wait = false) {
    DBG("3TAS::flush_updates\n");
#ifdef USE_HH
    if (hh_store) {
      for (auto it = hh_store.begin_single(); it != hh.end_single(); it++) {
        assert(it->count() == 1);
        update(it->target_rank(), it->elem());
        it->target_rank() = rank_n();  // and erase it
      }
    }
#endif
    bool flat_mode = false;
    if (tt_rpc_counts->total.is_null()) {
      // never initialized tt_rpc_counts... must be in flat mode
      assert(tt_rpc_counts->targets.empty());
      assert(this->aggr_team.rank_n() == splits->full_n());
      assert(tt_max_micro_store_size_per_node == 0);
      assert(tt_max_store_size_per_node == 0);
      flat_mode = true;
    }
    DBG("3TAS::flush_updates appending my remaining micro_stores\n");
    // flush all my micro_store blocks, stagger appends
    node_num_t my_node = splits->node_me();
    node_num_t stagger = splits->full_me() % splits->node_n();
    if (tt_max_micro_store_size_per_node > 1) {
      assert(tt_micro_store.size() == splits->node_n());
      for (node_num_t i = 0; i < tt_micro_store.size(); i++) {
        node_num_t node_idx = (i + stagger) % tt_micro_store.size();

        auto &micro_store = tt_micro_store[node_idx];
        assert((node_idx != splits->node_me() || micro_store.first.empty()) && "micro store to own node is empty");
        if (micro_store.first.size() > 0) {
          
          append_shared_store(node_idx);
        }
        assert(micro_store.first.empty() && micro_store.second.empty());
      }
    }  // else no micro_stores

      upcxx::barrier(splits->thread_team());
    

    DBG("3TAS::flush_updates sending node stores\n");
    // now call update_remote for any tt_store owned by this rank
    if (tt_max_store_size_per_node > 1) {
      assert(tt_store.size() == splits->node_n());
      for (node_num_t i = 0; i < splits->node_n(); i++) {
        node_num_t node_idx = (i + stagger) % splits->node_n();
        auto &store_ptr = tt_store[node_idx];
        if (store_ptr.is_null()) {
          assert(node_idx == splits->node_me());
          continue;
        }
        assert(!store_ptr.is_null());
        assert(store_ptr.is_local());
        if (store_ptr.where() == rank_me()) {
          auto &store = *(store_ptr.local());
          if (node_idx == splits->node_me()) {
            assert(store.empty() && "All updates to own node are through FlatAggrStore");
            continue;
          }
          if (tt_max_store_size_per_node > 0) {
            
            std::apply(tt_update_remote, std::tuple_cat(std::make_tuple(this, node_idx), this->data));
          }
          assert(tt_store[node_idx].local()->empty());
        } else {
          // else not owned by this rank.
          intrank_t local_tracking_rank = store_ptr.where();
          assert(local_team_contains(local_tracking_rank));
          assert(splits->thread_team_contains_world(local_tracking_rank));
        }
      }
    }
   
    // flush flat aggr store (thread team) no-wait
    ((FAS *)this)->flush_updates(true);

    if (no_wait) {
      return;
    } else {
      // must wait for other ranks in the thread team to send their node stores
        upcxx::barrier(splits->thread_team());
      
    }
    // tell the target node how many rpcs this thread_team() has sent to it
    for (node_num_t i = 0; !flat_mode && i < splits->node_n(); i++) {
      node_num_t node_idx = (i + stagger) % splits->node_n();
      auto &counts_ptr = tt_rpc_counts->targets[node_idx];
      assert(!counts_ptr.is_null());
      assert(counts_ptr.is_local());
      assert((node_idx != splits->node_me() || counts_ptr.local()->rpcs_sent.load() == 0) && "0 sent to own node");
      if (counts_ptr.where() == rank_me()) {
        auto num_sent = counts_ptr.local()->rpcs_sent.load();
        auto num_processed = counts_ptr.local()->rpcs_processed.load();
        DBG("telling node=", node_idx, " we sent=", num_sent, "\n");
        if (num_sent == 0) continue;
        upcxx::future<> fut =
            rpc(splits->node_team(), node_idx,
                [](TTDistRPCCounts &tt_rpc_counts, CountType rpcs_sent, CountType rpcs_processed, intrank_t source_node) {
                  (*tt_rpc_counts).targets[source_node].local()->rpcs_expected += rpcs_sent;
                  (*tt_rpc_counts).total.local()->rpcs_expected += rpcs_sent;
                  tt_rpc_counts->set_progressed_count(source_node, rpcs_processed);
                  DBG("From node=", source_node,
                      ", expecting=", (*tt_rpc_counts).targets[source_node].local()->rpcs_expected.load(), "\n");
                },
                tt_rpc_counts, num_sent, num_processed, my_node);
        do {
          
          fut = limit_outstanding_futures(fut);
        } while (!fut.ready());
      }
    }
    auto fut_done = flush_outstanding_futures_async();
      fut_done.wait()
      //int j=0;
    //while (!fut_done.ready()) {
       // j+=0;
        
    //}

    DBG("Waiting for quiescence of counts\n");

      upcxx::barrier(splits->full_team());
    
    auto max_updates_fut = upcxx::reduce_all(tt_updates, op_fast_max, splits->full_team());
    auto sum_updates_fut = upcxx::reduce_all(tt_updates, op_fast_add, splits->full_team());

    CountType tot_rpcs_processed = 0, team_tot_rpcs_processed = 0;
    // now wait for all of our rpcs.
    assert(flat_mode || tt_rpc_counts->targets.size() == splits->node_n());
    for (node_num_t node_idx = 0; !flat_mode && node_idx < splits->node_n(); node_idx++) {
      auto &counts = (*tt_rpc_counts).targets[node_idx];
      assert(counts.is_local());
      DBG("Waiting for node ", node_idx, "of", splits->node_n(), " expected=", counts.local()->rpcs_expected.load(),
          " == processed (so far)=", counts.local()->rpcs_processed.load(), "\n");
      while (counts.local()->rpcs_expected != counts.local()->rpcs_processed) {
        
        // DBG_VERBOSE("node=", node_idx, " expect=", counts.local()->rpcs_expected.load(), " processed=",
        // counts.local()->rpcs_processed.load(), " progressed=", counts.local()->rpcs_progressed.load(), "\n");
        assert(counts.local()->rpcs_expected >= counts.local()->rpcs_processed && "more expected than processed");
      }
      if (counts.where() == rank_me()) {
        tot_rpcs_processed += counts.local()->rpcs_processed;
      }
      team_tot_rpcs_processed += counts.local()->rpcs_processed;
    }
    if (!flat_mode)
      SLOG_VERBOSE("Rank ", rank_me(), " sent ", tt_rpc_counts->total.local()->rpcs_sent, " rpcs and received ", tot_rpcs_processed,
                   " (whole team:", tt_rpc_counts->total.local()->rpcs_processed.load(), ")\n");
    tt_barrier();
    if (!flat_mode) assert(team_tot_rpcs_processed == tt_rpc_counts->total.local()->rpcs_processed.load());

    // now flush all updates from FlatAggrStore;
    ((FAS *)this)->flush_updates();

    // cleanup counts
    tt_rpc_counts->reset();

    auto max_updates = max_updates_fut.wait();
    auto sum_updates = sum_updates_fut.wait();
    if (max_updates > 0)
      SLOG_VERBOSE("Rank ", rank_me(), " had ", tt_updates, " remote node updates, avg ", sum_updates / splits->full_n(), " max ",
                   max_updates, ", balance ", std::setprecision(2), std::fixed, (float)tt_updates / (float)max_updates, "\n");
    tt_updates = 0;
    tt_barrier();
  }
};

};  // namespace upcxx_utils

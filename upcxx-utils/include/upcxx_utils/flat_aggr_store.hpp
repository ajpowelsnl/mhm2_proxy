#pragma once

#include <algorithm>
#include <string>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/bin_hash.hpp"
#include "upcxx_utils/heavy_hitter_streaming_store.hpp"
#include "upcxx_utils/limit_outstanding.hpp"
#include "upcxx_utils/log.hpp"


using std::string;
using std::to_string;

using upcxx::barrier;
using upcxx::dist_object;
using upcxx::future;
using upcxx::intrank_t;
using upcxx::make_future;
using upcxx::make_view;
using upcxx::op_fast_add;
using upcxx::op_fast_max;
using upcxx::progress;
using upcxx::promise;
using upcxx::rank_me;
using upcxx::rank_n;
using upcxx::reduce_all;
using upcxx::reduce_one;
using upcxx::rpc;
using upcxx::team;
using upcxx::view;

//#define USE_HH

namespace upcxx_utils {


// this class aggregates updates into local buffers and then periodically does an rpc to dispatch them

struct TargetRPCCounts {
  using CountType = uint64_t;

  CountType rpcs_sent;
  CountType rpcs_expected;
  CountType rpcs_processed;
  CountType rpcs_progressed;
  size_t imbalance_factor;
  TargetRPCCounts();
  void reset();
};

struct FASRPCCounts {
  using CountType = TargetRPCCounts::CountType;
  using DistFASRPCCounts = dist_object<FASRPCCounts>;

  TargetRPCCounts total;
  vector<TargetRPCCounts> targets;
  //IntermittentTimer rpc_prep_timer, rpc_inner_timer, rpc_outer_timer, wait_for_rpcs_timer, append_store_timer;
  //ProgressTimer progress_timer;
  const upcxx::team &tm;
  FASRPCCounts(const upcxx::team &tm = upcxx::world());
  void init();
  void reset();
  void print_out();
  void increment_sent_counters(intrank_t target_rank);
  void increment_processed_counters(intrank_t source_rank);
  void set_progressed_count(intrank_t source_rank, CountType num_progressed);
  void wait_for_rpcs(intrank_t target_rank, CountType max_rpcs_in_flight);
  void update_progressed_count(DistFASRPCCounts &dist_fas_rpc_counts, intrank_t target_rank);
};
using DistFASRPCCounts = FASRPCCounts::DistFASRPCCounts;

template <typename T>
class GlobalArrayView {
  upcxx::global_ptr<T> ptr;
  size_t len;

 public:
  GlobalArrayView()
      : ptr(nullptr)
      , len(0) {}
  GlobalArrayView(upcxx::global_ptr<T> ptr, size_t len)
      : ptr(ptr)
      , len(len) {
    assert(ptr.is_local());
  }
  upcxx::global_ptr<T> get_ptr() { return ptr; }
  T *begin() { return ptr.local(); }
  T *end() { return ptr.local() + len; }
  const T *begin() const { return ptr.local(); }
  const T *end() const { return ptr.local() + len; }
  size_t size() const { return len; }
};

template <typename T, typename... Data>
class FlatAggrStore {
 public:
  using RankStore = vector<T>;
  using RankStoreIterator = typename RankStore::iterator;
  using Store = vector<RankStore>;
  using UpdateFunc = std::function<void(T, Data &...)>;
  using DistUpdateFunc = dist_object<UpdateFunc>;

  using CountType = TargetRPCCounts::CountType;

  using DistRPCCounts = DistFASRPCCounts;

#ifdef USE_HH
  using HHStore = HeavyHitterStreamingStore<T>;
  using retrieve_t = HeavyHitterStreamingStore<T>::retrieve_t;
#endif

 protected:
  Store store;
  const team &aggr_team;
  CountType max_store_size_per_target;
  CountType max_rpcs_in_flight;
  CountType updates_self, updates_remote;
#ifdef USE_HH
  HHStore hh_store;
#endif
  DistRPCCounts rpc_counts;
  string description;

  // save the update function to use in both update and flush
  DistUpdateFunc update_func = DistUpdateFunc({});
  // save all associated data structures as a tuple of a variable number of parameters
  std::tuple<Data &...> data;

  static void wait_for_rpcs(FlatAggrStore *astore, intrank_t target_rank) {
    assert(target_rank < astore->aggr_team.rank_n());
    FASRPCCounts &counts = *astore->rpc_counts;
    auto &tgt_imbalance = counts.targets[target_rank].imbalance_factor;
    auto imbal = tgt_imbalance;
    counts.wait_for_rpcs(target_rank, astore->max_rpcs_in_flight);
    if (tgt_imbalance > imbal) {
      // query target for its progress count
      counts.update_progressed_count(astore->rpc_counts, target_rank);
    }
  }

  static void increment_rpc_counters(DistRPCCounts &rpc_counts, intrank_t target_rank) {
    rpc_counts->increment_sent_counters(target_rank);
  }

#ifdef USE_HH
  static void rpc_update_func(DistUpdateFunc &update_func, view<T> rank_store, view<std::pair<T, uint8_t> > hh_store,
                              DistRPCCounts &rpc_counts, intrank_t source_rank, CountType num_progressed, Data &... data)
#else
  static void rpc_update_func(DistUpdateFunc &update_func, view<T> rank_store, DistRPCCounts &rpc_counts, intrank_t source_rank,
                              CountType num_progressed, Data &... data)
#endif
  {
    DBG_VERBOSE("source_rank=", source_rank, ",  rank_store.size()=", rank_store.size(), "\n");

    

    // increment the processed counters
    rpc_counts->set_progressed_count(source_rank, num_progressed);

    auto &func = *update_func;
    for (const auto &elem : rank_store) {
      func(elem, data...);
    }
#ifdef USE_HH
    for (const auto &hh_elem_count : hh_store) {
      for (int i = 0; i < hh_elem_count.second; i++) {
        (*update_func)(hh_elem_count.first, data...);
      }
    }
#endif

    assert((intrank_t)rpc_counts->targets.size() > source_rank);
    rpc_counts->increment_processed_counters(source_rank);
    
  };

  template <typename ViewType>
#ifdef USE_HH
  static void update_remote_rpc_ff(const team &aggr_team, intrank_t target_rank, DistUpdateFunc &update_func, ViewType &elems,
                                   retrieve_t &hh, DistRPCCounts &v, Data &... data)
#else
  static void update_remote_rpc_ff(const team &aggr_team, intrank_t target_rank, DistUpdateFunc &update_func, ViewType &elems,
                                   DistRPCCounts &rpc_counts, Data &... data)
#endif

  {
    assert(target_rank < aggr_team.rank_n());
    assert(target_rank != aggr_team.rank_me() && "no rpcs to self");
    DBG_VERBOSE("update_remote_rpc_ff() target_rank=", target_rank, "\n");

    increment_rpc_counters(rpc_counts, target_rank);
    
    rpc_ff(aggr_team, target_rank, rpc_update_func, update_func, elems,
#ifdef USE_HH
           hh,
#endif
           rpc_counts, aggr_team.rank_me(), rpc_counts->targets[target_rank].rpcs_processed, data...);
    
  }

  // operates on a vector of elements in the store

  static void update_remote(FlatAggrStore *astore, intrank_t target_rank, Data &... data) {
    assert(target_rank < astore->aggr_team.rank_n());
    assert((target_rank != astore->aggr_team.rank_me() || astore->store[target_rank].empty()) &&
           "no updates to self except maybe HHSS");
    assert((intrank_t)astore->store.size() == astore->aggr_team.rank_n());
    DBG("update_remote() target_rank=", target_rank, "\n");

#ifdef USE_HH
    // get any active heavy hitters for this target and flush it
    auto hh = astore->hh_store.retrieve(target_rank);
    // no need to send an empty rpc
    if (hh.empty() && astore->store[target_rank].empty()) return;
#else
    if (astore->store[target_rank].empty()) return;
#endif
    wait_for_rpcs(astore, target_rank);

    
    auto elems_view = make_view(astore->store[target_rank].begin(), astore->store[target_rank].end());
    
#ifdef USE_HH
    update_remote_rpc_ff(astore->aggr_team, target_rank, astore->update_func, elems_view, hh, astore->rpc_counts, data...);
#else
    update_remote_rpc_ff(astore->aggr_team, target_rank, astore->update_func, elems_view, astore->rpc_counts, data...);
#endif
    astore->store[target_rank].clear();
    
  }

  // operates on a single element

  static void update_remote1(FlatAggrStore *astore, intrank_t target_rank, const T &elem, Data &... data) {
    assert(target_rank < astore->aggr_team.rank_n());
    assert(target_rank != astore->aggr_team.rank_me() && "no updates to self");
    assert((intrank_t)astore->rpc_counts->targets.size() == astore->aggr_team.rank_n());
    DBG_VERBOSE("update_remote1() target_rank=", target_rank, "\n");
    wait_for_rpcs(astore, target_rank);
    increment_rpc_counters(astore->rpc_counts, target_rank);
    
    rpc_ff(astore->aggr_team, target_rank,
           [](DistUpdateFunc &update_func, T elem, DistRPCCounts &rpc_counts, intrank_t source_rank, CountType num_progressed,
              Data &... data) {
             DBG_VERBOSE("update_remote1::rpc_ff() source_rank=", source_rank, "\n");
             
             rpc_counts->set_progressed_count(source_rank, num_progressed);

             (*update_func)(elem, data...);

             rpc_counts->increment_processed_counters(source_rank);
             
           },
           astore->update_func, elem, astore->rpc_counts, astore->aggr_team.rank_me(),
           astore->rpc_counts->targets[target_rank].rpcs_processed, data...);
   
  }

  void init_rpc_counts() {
    rpc_counts->init();
    barrier(aggr_team);
  }

  void reset_rpc_counts() {
    barrier(aggr_team);
    rpc_counts->reset();
    barrier(aggr_team);
  }

 public:
  FlatAggrStore(const team &team, Data &... data)
      : store({})
      , aggr_team(team)
      , max_store_size_per_target(0)
      , max_rpcs_in_flight(0)
      , updates_self(0)
      , updates_remote(0)
#ifdef USE_HH
      , hh_store({})
#endif
      , rpc_counts(team, team)
      , data(data...) {
    DBG("Team construct FlatAggrStore\n");
  }

  FlatAggrStore(Data &... data)
      : store({})
      , aggr_team(upcxx::world())
      , max_store_size_per_target(0)
      , max_rpcs_in_flight(0)
      , updates_self(0)
      , updates_remote(0)
#ifdef USE_HH
      , hh_store({})
#endif
      , rpc_counts(upcxx::world())
      , data(data...) {
    DBG("Default contruct FlatAggrStore\n");
  }

  virtual ~FlatAggrStore() {
    DBG("Destroying FlatAggrStore\n");
    clear();
  }

  void set_size(const string &desc, CountType max_store_bytes, CountType max_rpcs_in_flight = 128, bool use_heavy_hitters = true) {
    description = desc;
    DBG(desc, " max_store_bytes=", max_store_bytes, " max_rpcs_in_flight=", max_rpcs_in_flight, ", team=", aggr_team.rank_n(),
        "\n");
    init_rpc_counts();
    this->max_rpcs_in_flight = max_rpcs_in_flight;
    auto num_targets = aggr_team.rank_n() - 1;         // all but me
    size_t max_message_size = 1 * 1024 * 1024 - 1024;  // 999KB

    if (num_targets == 0) {
      DBG("No allocation for no targets\n");
      max_store_size_per_target = 0;
      return;
    }
    // at least 10 entries per target rank
    auto max_store_size_per_rank = max_store_bytes / sizeof(T);  // per_rank on a node
    max_store_size_per_target = max_store_size_per_rank / num_targets;
    if (max_store_size_per_target < 10) {
      if (max_store_size_per_target < 2) max_store_size_per_target = 0;
      SWARN("FlatAggrStore ", this->description, " max_store_size_per_target is small (", max_store_size_per_target,
            ") please consider increasing the max_store_bytes (", get_size_str(max_store_bytes), ")\n");
      SWARN("at this scale of ", num_targets, " other ranks, at least ", get_size_str(10 * sizeof(T) * num_targets),
            " is necessary for good performance\n");
    }
    if (max_store_size_per_target * sizeof(T) > max_message_size) {
      max_store_size_per_target = max_message_size / sizeof(T);
    }
    if (max_store_size_per_target > 1) {
      store.resize(aggr_team.rank_n(), {});
      // reserve room in store
      for (intrank_t i = 0; i < aggr_team.rank_n(); i++) {
        if (i != aggr_team.rank_me()) {  // self is always bypassed
          store[i].reserve(max_store_size_per_target);
        }
      }
    }
    SLOG_VERBOSE(desc, ": using a flat aggregating store for each rank (", aggr_team.rank_n(), ") of max ",
                 get_size_str(max_store_bytes), " per aggregating rank ", get_size_str(max_store_bytes * local_team().rank_n()),
                 " node mem\n");
    SLOG_VERBOSE("  max ", max_store_size_per_target, " entries of ", get_size_str(sizeof(T)), " per target rank, ",
                 get_size_str(max_store_size_per_target * sizeof(T)), " message size, ", num_targets, " targets, ",
                 get_size_str(max_store_size_per_target * sizeof(T) * num_targets * local_team().rank_n()), " node mem\n");
    SLOG_VERBOSE("  max RPCs in flight: ", (!max_rpcs_in_flight ? string("unlimited") : to_string(max_rpcs_in_flight)), "\n");
#ifdef USE_HH
    if (use_heavy_hitters) {
      // allocate heavy hitters approx the size of the RankStore for one more node in the job
      hh_store.reserve(max_store_size_per_target * upcxx::local_team().rank_n());
    }
#endif
    barrier(aggr_team);
  }

  void set_update_func(UpdateFunc update_func) {
    barrier(aggr_team);  // to avoid race of first update
    if (max_store_size_per_target > 1 && store.empty())
      DIE("Invalid condition - FlatAggrStore not initialized yet - call set_size() after construction or clear()!\n");
    *(this->update_func) = update_func;
    barrier(aggr_team);  // to avoid race of first update
  }

  const DistUpdateFunc &get_update_func() const { return this->update_func; }

  void clear() {
    DBG("\n");
    for (const auto &s : store) {
      if (!s.empty()) throw string("rank store is not empty!");
    }
    Store().swap(store);
    updates_self = 0;
    updates_remote = 0;
    reset_rpc_counts();
#ifdef USE_HH
    hh_store.clear();
#endif
  }

  const upcxx::team &get_team() const { return aggr_team; }

  void update(intrank_t target_rank, const T &elem) {
    // DBG_VERBOSE("update(target_rank=", target_rank, " elem=", &elem, " ", (int) *((char*)&elem), "'", *((char*)&elem), "')\n");
    assert(target_rank < aggr_team.rank_n());
    if (aggr_team.rank_me() == target_rank) {
      // always just bypass self
      std::apply(*update_func, std::tuple_cat(std::forward_as_tuple(elem), data));
      updates_self++;
      return;
    }
    updates_remote++;

    if (max_store_size_per_target > 1) {
      
      if ((intrank_t)store.size() != aggr_team.rank_n())
        DIE("Invalid state.  set_size must be called after construction or clear()\n");
#ifdef USE_HH
      T new_elem;
      if (hh_store.update(target_rank, elem, new_elem)) return;
      store[target_rank].push_back(new_elem);
#else
      store[target_rank].push_back(elem);
#endif
    
      if (store[target_rank].size() < max_store_size_per_target) {
        return;
      }
      std::apply(update_remote, std::tuple_cat(std::make_tuple(this, target_rank), data));
    } else {
      assert(max_store_size_per_target == 0);
      std::apply(update_remote1, std::tuple_cat(std::forward_as_tuple(this, target_rank, elem), data));
    }
  }

  // flushes all pending updates
  // if no_wait is true, counts are not exchanged and quiescence has NOT been achieved

  void flush_updates(bool no_wait = false) {
    DBG("flush_update()\n");

#ifdef USE_HH
    if (hh_store) {
      for (auto it = hh_store.begin_single(); it != hh.end_single(); it++) {
        assert(it->count() == 1);
        update(it->target_rank(), it->elem());
        it->target_rank() = rank_n();  // and erase it
      }
    }
#endif

    // when we update, every rank starts at a different rank to avoid bottlenecks
    if ((intrank_t)rpc_counts->targets.size() != aggr_team.rank_n())
      DIE("Inconsistent number or targets ", rpc_counts->targets.size(), " and team size ", aggr_team.rank_n(), "\n");

    for (int i = 0; i < aggr_team.rank_n(); i++) {
      intrank_t target_rank = (aggr_team.rank_me() + i) % aggr_team.rank_n();
      if (max_store_size_per_target > 0) {
        std::apply(update_remote, std::tuple_cat(std::make_tuple(this, target_rank), data));
      }
    }
    
    if (no_wait) {
      return;
    }

    for (int i = 0; i < aggr_team.rank_n(); i++) {
      intrank_t target_rank = (aggr_team.rank_me() + i) % aggr_team.rank_n();
      auto num_sent = (*rpc_counts).targets[target_rank].rpcs_sent;
      auto num_processed = (*rpc_counts).targets[target_rank].rpcs_processed;
      // tell the target how many rpcs we sent to it
      DBG("Sent ", num_sent, " to ", target_rank, "of", aggr_team.rank_n(), "\n");
      if (num_sent == 0 && num_processed == 0) continue;
      upcxx::future<> fut =
          rpc(aggr_team, target_rank,
              [](DistRPCCounts &rpc_counts, CountType rpcs_sent, CountType rpcs_processed, intrank_t source_rank) {
                (*rpc_counts).targets[source_rank].rpcs_expected += rpcs_sent;
                (*rpc_counts).total.rpcs_expected += rpcs_sent;
                rpc_counts->set_progressed_count(source_rank, rpcs_processed);
                DBG("From rank=", source_rank, ", expecting=", (*rpc_counts).targets[source_rank].rpcs_expected, "\n");
              },
              rpc_counts, num_sent, num_processed, aggr_team.rank_me());
      do {
       
        fut = limit_outstanding_futures(fut);
      } while (!fut.ready());
    }

    CountType max_vals[2], sum_vals[2];
    auto max_fut = upcxx::reduce_all(&updates_self, max_vals, 2, upcxx::op_fast_max, aggr_team);
    auto sum_fut = upcxx::reduce_all(&updates_self, sum_vals, 2, upcxx::op_fast_add, aggr_team);

    DBG("flush_updates() waiting for counts\n");
    auto fut_done = flush_outstanding_futures_async();
      fut_done.wait();
      //int dummy=0;
    //while (!fut_done.ready()) {
        //dummy++;
    //}

    DBG("flush_updates() waiting for quiescence of counts\n");

    // fully timed barrier after all counts have been sent and surrounding quiescence
      upcxx::barrier(aggr_team);
    
    CountType tot_rpcs_processed = 0;
    if ((intrank_t)rpc_counts->targets.size() != aggr_team.rank_n())
      DIE("Inconsistent number or targets ", rpc_counts->targets.size(), " and team size ", aggr_team.rank_n(), "\n");
    // now wait for all of our rpcs.
    for (int i = 0; i < aggr_team.rank_n(); i++) {
      auto &rcounts = (*rpc_counts).targets[i];
      DBG("Waiting for rank ", i, "of", rpc_counts->targets.size(), " expected=", rcounts.rpcs_expected,
          " == processed (so far)=", rcounts.rpcs_processed, "\n");
      while (rcounts.rpcs_expected != rcounts.rpcs_processed) {
        
        assert(rcounts.rpcs_expected >= rcounts.rpcs_processed && "more expected than processed");
      }
      tot_rpcs_processed += rcounts.rpcs_processed;
    }
    SLOG_VERBOSE("Rank ", rank_me(), " sent ", rpc_counts->total.rpcs_sent, " rpcs and received ", tot_rpcs_processed, " (",
                 rpc_counts->total.rpcs_processed, ") \n");
    assert(tot_rpcs_processed == rpc_counts->total.rpcs_processed);

    max_fut.wait();
    sum_fut.wait();
    if (max_vals[0] > 0)
      SLOG_VERBOSE("Rank ", rank_me(), " had ", updates_self, " self updates, avg ", sum_vals[0] / aggr_team.rank_n(), " max ",
                   max_vals[0], " balance ", std::setprecision(2), std::fixed, (float)updates_self / (float)max_vals[0], "\n");
    if (max_vals[1] > 0)
      SLOG_VERBOSE("Rank ", rank_me(), " had ", updates_remote, ", remote updates, avg ", sum_vals[1] / aggr_team.rank_n(), " max ",
                   max_vals[1], " balance ", std::setprecision(2), std::fixed, (float)updates_remote / (float)max_vals[1], "\n");
    updates_self = updates_remote = 0;
    reset_rpc_counts();
  }
};

};  // namespace upcxx_utils

#include "upcxx_utils/three_tier_aggr_store.hpp"

namespace upcxx_utils {

TT_RPC_Counts::TT_RPC_Counts() { reset(); }

void TT_RPC_Counts::reset() {
  DBG_VERBOSE(this, " \n");
  rpcs_sent.store(0);
  rpcs_expected.store(0);
  rpcs_processed.store(0);
  rpcs_progressed.store(0);
  imbalance_factor.store(1);
}

TT_All_RPC_Counts::TT_All_RPC_Counts(DistFASRPCCounts &dist_fas_rpc_counts)
    : total()
    , targets{}
    , inner_rpc_future(make_future())
    , fas_rpc_counts(dist_fas_rpc_counts) {
  DBG_VERBOSE(this, " Default Constructor\n");
}

TT_All_RPC_Counts::~TT_All_RPC_Counts() {
  DBG_VERBOSE(this, "\n");
  reset(true);
}

void TT_All_RPC_Counts::init(const split_team &splits) {
  DBG_VERBOSE(this, "\n");
  node_num_t num_nodes = splits.node_n();
  if (splits.thread_team().rank_me() == 0) {
    assert(!total);
    total = upcxx::new_<TT_RPC_Counts>();
  }
  future<> all_done_fut = upcxx::broadcast(&total, 1, 0, splits.thread_team());
  targets.resize(num_nodes, {});
  for (node_num_t i = 0; i < num_nodes; i++) {
    // FIXME this is an inefficient way to gather, even on local ranks
    intrank_t tracking_thread = i % splits.thread_team().rank_n();
    auto &counts_ptr = targets[i];
    if (tracking_thread == splits.thread_team().rank_me()) {
      assert(!counts_ptr);
      counts_ptr = upcxx::new_<TT_RPC_Counts>();
    }
    auto fut = upcxx::broadcast(&counts_ptr, 1, tracking_thread, splits.thread_team());
    all_done_fut = when_all(all_done_fut, fut);
  }
  all_done_fut.wait();
#ifdef DEBUG
  assert(!total.is_null());
  assert(total.is_local());
  for (node_num_t i = 0; i < num_nodes; i++) {
    auto &counts_ptr = targets[i];
    assert(!counts_ptr.is_null());
    assert(counts_ptr.is_local());
  }
#endif
}

void TT_All_RPC_Counts::reset(bool free) {
  DBG_VERBOSE(this, " free=", free, " total=", total, "\n");
  inner_rpc_future.wait();
  if (total) print_out();
  if (total && total.where() == rank_me()) {
    total.local()->reset();
    if (free) upcxx::delete_(total);
  }
  if (free) total = {};

  for (auto &t : targets) {
    if (t && t.where() == rank_me()) {
      t.local()->reset();
      if (free) {
        upcxx::delete_(t);
      }
    }
  }
  if (free) targets.clear();
  
}

void TT_All_RPC_Counts::print_out() {
  
}

void TT_All_RPC_Counts::increment_sent_counters(intrank_t target_node) {
  assert(target_node < (intrank_t)targets.size());
  // increment the counters
  total.local()->rpcs_sent++;
  targets[target_node].local()->rpcs_sent++;
}

void TT_All_RPC_Counts::increment_processed_counters(intrank_t source_node) {
  assert(source_node < (intrank_t)targets.size());
  total.local()->rpcs_processed++;
  targets[source_node].local()->rpcs_processed++;
}

void TT_All_RPC_Counts::set_progressed_count(intrank_t source_node, CountType count) {
  assert(source_node < (intrank_t)targets.size());
  auto &rpcs_progressed = targets[source_node].local()->rpcs_progressed;
  CountType old = rpcs_progressed.load();
  while (count > old) {
    auto ret = rpcs_progressed.compare_exchange_weak(old, count);
    if (ret) {
      total.local()->rpcs_progressed += count - old;
      break;
    }
  }
}

void TT_All_RPC_Counts::wait_for_rpcs(intrank_t target_node, CountType max_rpcs_in_flight) {
  assert(target_node < (intrank_t)targets.size());
  DBG("tt_wait_for_rpcs() target_node=", target_node, " tot_rpcs_sent=", total.local()->rpcs_sent.load(),
      " rpcs_to_node=", targets[target_node].local()->rpcs_sent.load(),
      " processed_from_node=", targets[target_node].local()->rpcs_processed.load(), "\n");
  // limit the number in flight by making sure we don't have too many more sent than received (with good load balance,
  // every process is sending and receiving about the same number)
  // we don't actually want to check every possible rank's count while waiting, so just check the target rank

  
  bool imbalanced = false;
  auto &tgt = *targets[target_node].local();
  auto tgt_imbalance_factor = tgt.imbalance_factor.load();  // copy the imbalance factor
  auto &tgt_rpcs_expected = tgt.rpcs_expected;              // set when target is in flush
  if (max_rpcs_in_flight) {
    auto &tot = *total.local();
    auto &tot_rpcs_expected = tot.rpcs_expected;      // total expected during flush
    auto &tot_rpcs_sent = tot.rpcs_sent;              // total sent by rank
    auto &tot_rpcs_processed = tot.rpcs_processed;    // total processed by rank
    auto &tot_rpcs_progressed = tot.rpcs_progressed;  // total sent and processed remotely (delayed)

    auto &tgt_rpcs_sent = tgt.rpcs_sent;              // sent to target
    auto &tgt_rpcs_processed = tgt.rpcs_processed;    // processed from target
    auto &tgt_rpcs_progressed = tgt.rpcs_progressed;  // sent to target and processed remotely (delayed)
    CountType max_per_rank = (max_rpcs_in_flight + targets.size() - 1) / targets.size();
    max_per_rank = std::min(max_per_rank, (CountType)4);  // minimum of 4
    size_t iter = 0;
    while (tgt_rpcs_sent - tgt_rpcs_processed > max_per_rank && tgt_rpcs_sent - tgt_rpcs_progressed > max_per_rank &&
           tot_rpcs_sent - tot_rpcs_processed > max_rpcs_in_flight && tot_rpcs_sent - tot_rpcs_progressed > max_rpcs_in_flight) {
      iter++;
      
      bool target_is_in_flush = iter > max_per_rank && tgt_rpcs_expected > 0;
      bool target_is_imbalanced = !target_is_in_flush && iter > (max_rpcs_in_flight + 1) * tgt_imbalance_factor;
      if ((target_is_in_flush || target_is_imbalanced) && !upcxx::progress_required()) {
        if (target_is_imbalanced && tgt_imbalance_factor >= 4) {
          INFO_OR_LOG("Breaking out of wait_for_rpc 3TAS - iter=", iter, " rpcs sent=", tot_rpcs_sent,
                      " processed=", tot_rpcs_processed, " progressed=", tot_rpcs_progressed, ", node=", target_node,
                      " sent=", tgt_rpcs_sent, " processed=", tgt_rpcs_processed, " progressed=", tgt_rpcs_progressed, "\n");
        } else {
          LOG("Breaking out of wait_for_rpc 3TAS node=", target_node, " iter=", iter, " flush=", target_is_in_flush,
              " imbalfact=", tgt.imbalance_factor.load(), "\n");
        }
        if (target_is_imbalanced) {
          tgt.imbalance_factor = tgt.imbalance_factor.load() * 2;  // call progress twice as much next time.
          imbalanced = true;
        }
        break;  // escape eventually if load is imbalanced
      }
    }
  }
  if (!imbalanced) {
    tgt.imbalance_factor = 1;
  }
  
  DBG("tt_wait_for_rpcs() finished waiting target_node=", target_node, " tot_rpcs_sent=", total.local()->rpcs_sent.load(),
      " rpcs_to_node=", targets[target_node].local()->rpcs_sent.load(),
      " processed_from_node=", targets[target_node].local()->rpcs_processed.load(), "\n");
}

void TT_All_RPC_Counts::update_progressed_count(TTDistRPCCounts &dist_tt_rpc_counts, intrank_t target_node) {
  assert(this == &(*dist_tt_rpc_counts));
  const upcxx::team &t = dist_tt_rpc_counts.team();
  assert(target_node < t.rank_n());
  intrank_t me = t.rank_me();

  upcxx::rpc_ff(t, target_node,
                [](TTDistRPCCounts &dtt_rpc_counts, intrank_t source_node, CountType known_progressed) {
                  CountType processed = dtt_rpc_counts->targets[source_node].local()->rpcs_processed;
                  DBG("request for progress source_node=", source_node, " known_progressed=", known_progressed,
                      " processed=", processed, "\n");
                  assert(known_progressed <= processed);
                  if (known_progressed == processed) return;
                  // send update back
                  rpc_ff(dtt_rpc_counts.team(), source_node,
                         [](TTDistRPCCounts &dtt_rpc_counts, intrank_t return_node, CountType processed) {
                           DBG("Updating progressed for ", return_node, " to ", processed, "\n");
                           TT_All_RPC_Counts &rpc_counts = *dtt_rpc_counts;
                           rpc_counts.set_progressed_count(return_node, processed);
                         },
                         dtt_rpc_counts, dtt_rpc_counts.team().rank_me(), processed);
                },
                dist_tt_rpc_counts, me, targets[target_node].local()->rpcs_progressed.load());
  
}

};  // namespace upcxx_utils

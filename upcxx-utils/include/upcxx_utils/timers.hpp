#pragma once

#include <chrono>
#include <ctime>
#include <upcxx/upcxx.hpp>

//GCC
#include <cassert>
#include "math.h"

#include "colors.h"
#include "log.hpp"
#include "version.h"

using upcxx::barrier;
using upcxx::intrank_t;
using upcxx::make_future;
using upcxx::op_fast_add;
using upcxx::op_fast_max;
using upcxx::op_fast_min;
using upcxx::progress;
using upcxx::rank_me;
using upcxx::rank_n;
using upcxx::reduce_all;
using upcxx::reduce_one;
using upcxx::to_future;
using upcxx::when_all;
using upcxx::world;

using std::make_shared;
using std::ostringstream;
using std::shared_ptr;
using std::string;
using std::stringstream;
using timepoint_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
using std::chrono::seconds;
using duration_seconds = std::chrono::duration<double>;

namespace upcxx_utils {

template <typename T>
class MinSumMax {
 public:
  MinSumMax(T my_val = 0) { reset(my_val); }

  // construct a MinSumMax from another type and factor
  template <typename T2>
  MinSumMax(const T2 &other, T factor)
      : my(other.my * factor)
      , min(other.min * factor)
      , sum(other.sum * factor)
      , max(other.max * factor)
      , avg(other.avg * factor) {}

  void reset(T my_val) {
    my = my_val;
    min = my_val;
    sum = my_val;
    max = my_val;
    avg = 0.0;
  }

  string to_string() const {
    if (!is_valid()) return "NOT VALID MinSumMax";
    ostringstream oss;
    oss << std::setprecision(2) << std::fixed;
    double bal = (max != 0 ? avg / max : 1.0);
    oss << min << "/" << my << "/" << avg << "/" << max << ", bal=" << bal;
    auto s = oss.str();
    return s;
  }


  bool is_valid() const { return min <= my && min <= max && min <= avg && max >= my && max >= avg && sum >= fabs(avg); }

  struct op_MinSumMax {
    template <typename Ta, typename Tb>
    MinSumMax operator()(Ta a, Tb &&b) const {
      a.min = upcxx::op_fast_min(a.min, std::forward<T>(b.min));
      a.sum = upcxx::op_fast_add(a.sum, std::forward<T>(b.sum));
      a.max = upcxx::op_fast_max(a.max, std::forward<T>(b.max));
      return static_cast<MinSumMax &&>(a);
    };
  };
  void apply_avg(intrank_t n) { avg = n > 0 ? ((double)sum) / n : sum; }

  void apply_avg(const upcxx::team &team) { apply_avg(team.rank_n()); }

  T my, min, sum, max;
  double avg;
};
template <typename T>
ostream &operator<<(ostream &os, const MinSumMax<T> &msm) {
  return os << msm.to_string();
}

template <typename T>
upcxx::future<> min_sum_max_reduce_one(const MinSumMax<T> *msm_in, MinSumMax<T> *msm_out, int count, intrank_t root = 0,
                                       const upcxx::team &team = world()) {
  using MSM = MinSumMax<T>;
  typename MSM::op_MinSumMax op;
  vector<T> orig_vals(count);
  for (int i = 0; i < count; i++) {
    orig_vals[i] = msm_in[i].my;
  }
  return upcxx::reduce_one(msm_in, msm_out, count, op, root, team)
      .then([msm_out, count, &team, orig_vals = std::move(orig_vals)]() {
        for (auto i = 0; i < count; i++) {
          msm_out[i].apply_avg(team);
          msm_out[i].my = orig_vals[i];
        }
      });
}

template <typename T>
upcxx::future<MinSumMax<T>> min_sum_max_reduce_one(const T my, intrank_t root = 0, const upcxx::team &team = world()) {
  using MSM = MinSumMax<T>;
  auto sh_msm = make_shared<MSM>(my);
  return min_sum_max_reduce_one(sh_msm.get(), sh_msm.get(), 1, root, team).then([sh_msm, &team]() { return *sh_msm; });
}

template <typename T>
upcxx::future<> min_sum_max_reduce_all(const MinSumMax<T> *msm_in, MinSumMax<T> *msm_out, int count,
                                       const upcxx::team &team = world()) {
  using MSM = MinSumMax<T>;
  typename MSM::op_MinSumMax op;
  vector<T> orig_vals(count);
  for (int i = 0; i < count; i++) {
    orig_vals[i] = msm_in[i].my;
  }
  return upcxx::reduce_all(msm_in, msm_out, count, op, team).then([msm_out, count, &team, orig_vals = std::move(orig_vals)]() {
    for (auto i = 0; i < count; i++) {
      msm_out[i].apply_avg(team);
      msm_out[i].my = orig_vals[i];
    }
  });
}

template <typename T>
upcxx::future<MinSumMax<T>> min_sum_max_reduce_all(const T my, const upcxx::team &team = world()) {
  using MSM = MinSumMax<T>;
  auto sh_msm = make_shared<MSM>(my);
  return min_sum_max_reduce_all(sh_msm.get(), sh_msm.get(), 1, team).then([sh_msm, &team]() { return *sh_msm; });
}

// Reduce compile time by declaring extern templates of common types
// template instantiations happen in src/CMakeList and timers-extern-template.in.cpp

#define MACRO_MIN_SUM_MAX(TYPE, MODIFIER)                                                                                          \
  MODIFIER class MinSumMax<TYPE>;                                                                                                  \
  MODIFIER upcxx::future<> min_sum_max_reduce_one<TYPE>(const MinSumMax<TYPE> *, MinSumMax<TYPE> *, int, intrank_t,                \
                                                        const upcxx::team &);                                                      \
  MODIFIER upcxx::future<> min_sum_max_reduce_all<TYPE>(const MinSumMax<TYPE> *, MinSumMax<TYPE> *, int, const upcxx::team &team); \
  MODIFIER upcxx::future<MinSumMax<TYPE>> min_sum_max_reduce_one<TYPE>(const TYPE, intrank_t, const upcxx::team &);                \
  MODIFIER upcxx::future<MinSumMax<TYPE>> min_sum_max_reduce_all<TYPE>(const TYPE, const upcxx::team &team);

MACRO_MIN_SUM_MAX(float, extern template);
MACRO_MIN_SUM_MAX(double, extern template);
MACRO_MIN_SUM_MAX(int64_t, extern template);
MACRO_MIN_SUM_MAX(uint64_t, extern template);
MACRO_MIN_SUM_MAX(int, extern template);



};  // namespace upcxx_utils

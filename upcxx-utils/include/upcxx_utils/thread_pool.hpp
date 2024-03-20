#pragma once

/*
 * File:   thread_pool.hpp
 * Author: regan
 *
 */

#include <functional>
#include <memory>

#include <upcxx/upcxx.hpp>

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/timers.hpp"

// define this to disable all extra thread workers and any calls to enqueue_* will execute immediately
// #define UPCXX_UTILS_NO_THREAD_POOL

namespace upcxx_utils {

class ThreadPool_detail;
class ThreadPool {
  //
  // ThreadPool is based mostly on the ThreadPool.h from https://github.com/progschj/ThreadPool
  // by Jakob Progsch, Václav Zeman.
  //
  // This class was rewritten in 2020 by Rob Egan for use within upcxx and upcxx_utils
  //

  //
  //  Copyright (c) 2012 Jakob Progsch, Václav Zeman
  //
  // This software is provided 'as-is', without any express or implied
  // warranty. In no event will the authors be held liable for any damages
  // arising from the use of this software.
  //
  // Permission is granted to anyone to use this software for any purpose,
  // including commercial applications, and to alter it and redistribute it
  // freely, subject to the following restrictions:
  //
  // 1. The origin of this software must not be misrepresented; you must not
  //    claim that you wrote the original software. If you use this software
  //    in a product, an acknowledgment in the product documentation would be
  //    appreciated but is not required.
  //
  // 2. Altered source versions must be plainly marked as such, and must not be
  //    misrepresented as being the original software.
  //
  // 3. This notice may not be removed or altered from any source
  //    distribution.

  using Task = std::function<void()>;  // void(void) function/lambda wrappers
  std::unique_ptr<ThreadPool_detail> tp_detail;

  // is_ready() returns true when a thread needs to wake ( stop condition or a task is enqueued)
  // bool is_ready() const;
  // is_terminal() returns true when a thread needs to stop (stop condition and no tasks are enqueued)
  // bool is_terminal() const;
  // returns true if there are threads in this ThreadPool (spawned or available to be spawned)
  // bool is_active() const;

  void enqueue_task(std::shared_ptr<Task> sh_task);

 public:
  static size_t &global_task_id() {
    static size_t task_id = 0;
    return task_id;
  }
  static size_t &global_tasks_completed() {
    static size_t completed = 0;
    return completed;
  }
  static ThreadPool &get_single_pool(int num_threads = -1);
  static void join_single_pool();
  static auto tasks_outstanding() {
    auto complete = global_tasks_completed();
    auto id = global_task_id();
    assert(id >= complete);
    return id - complete;
  }

  static void yield_if_needed();

  int get_max_workers() const;
  bool is_done() const;

  template <typename Func, class... Args>
  static auto enqueue_in_single_pool(int num_workers, Func &&func, Args &&... args) {
    auto &tp = ThreadPool::get_single_pool(num_workers);
    return tp.enqueue(std::forward<Func>(func), std::forward<Args>(args)...);
  }

  template <typename Func, class... Args>
  static upcxx::future<> &enqueue_in_single_pool_serially(upcxx::future<> &serial_fut, Func &&func, Args &&... args) {
    assert(upcxx::master_persona().active_with_caller() && "Called from master persona");
    DBG_VERBOSE("enqueue_in_single_pool_serially: ", &serial_fut, " ", (serial_fut.ready() ? "ready" : "NOT READY"), "\n");

    using return_t = typename std::invoke_result<Func, Args...>::type;
    static_assert(std::is_void<return_t>::value, "void is the required return type for enqueue_in_serial_pool");

    auto &tp = ThreadPool::get_single_pool();  // use any thread in the single pool, enforce serialization by future chain

    auto args_tuple = std::make_tuple(args...);  // *copy* arguments to avoid races in argument references being reused
    serial_fut = serial_fut.then([&tp, func{std::move(func)}, args_tuple{std::move(args_tuple)}]() {
      return tp.enqueue_no_return([&func, args_tuple{std::move(args_tuple)}]() { std::apply(func, args_tuple); });
    });
    return serial_fut;
  }
  static upcxx::future<> &get_serial_future() {
    assert(upcxx::master_persona().active_with_caller() && "Called from master persona");
    static upcxx::future<> _serial_future = upcxx::make_future();
    return _serial_future;
  }
  template <typename Func, class... Args>
  static upcxx::future<> enqueue_serially(Func &&func, Args &&... args) {
    assert(upcxx::master_persona().active_with_caller() && "Called from master persona");
    return enqueue_in_single_pool_serially(get_serial_future(), func, args...);
  }

  ThreadPool(int num_threads);
  ThreadPool(const ThreadPool &copy) = delete;
  ThreadPool(ThreadPool &&move) = delete;
  ThreadPool &operator=(const ThreadPool &copy) = delete;
  ThreadPool &&operator=(ThreadPool &move) = delete;
  ~ThreadPool();

  template <class Func, class... Args>
  auto enqueue_with_return(Func &&func, Args &&... args) -> upcxx::future<typename std::invoke_result<Func, Args...>::type> {
    bool run_now = is_done();
#ifdef UPCXX_UTILS_NO_THREAD_POOL
    assert(run_now && "is never active when UPCXX_UTILS_NO_THREAD_POOL");
#endif
    if (run_now) {
      // execute and return immediately
      return make_future(func(args...));
    }

    using return_t = typename std::invoke_result<Func, Args...>::type;
    static_assert(!std::is_void<return_t>::value,
                  "void is not a valid return type for enqueue_with_return... did you mean enqueue()?");
    upcxx::persona &persona = upcxx::current_persona();
    std::shared_ptr<upcxx::promise<return_t>> sh_prom = std::make_shared<upcxx::promise<return_t>>();
    sh_prom->require_anonymous(1);  // additional requirement to complete

    auto task_id = global_task_id()++;
    auto start_t = 0;
    DBG("sh_prom=", sh_prom.get(), " task_id=", task_id, "\n");

    auto args_tuple = std::make_tuple(args...);  // *copy* arguments to avoid races in argument references being reused
    auto task =
        std::make_shared<Task>([task_id, start_t, sh_prom, &persona, func{std::move(func)}, args_tuple{std::move(args_tuple)}]() {
          DBG_VERBOSE("Executing sh_prom=", sh_prom.get(), "\n");
          sh_prom->fulfill_result(std::apply(func, args_tuple));
          DBG_VERBOSE("Finished sh_prom=", sh_prom.get(), "\n");
          // fulfill only in calling persona
          persona.lpc_ff([task_id, start_t, sh_prom]() {
            duration_seconds s = 0;
            DBG("Fulfilled sh_prom=", sh_prom.get(), " task_id=", task_id, " in ", s.count(), " s\n");
            sh_prom->fulfill_anonymous(1);
            global_tasks_completed()++;
          });
        });
    enqueue_task(task);
    return sh_prom->get_future();
  }

  template <class Func, class... Args>
  upcxx::future<> enqueue_no_return(Func &&func, Args &&... args) {
    bool run_now = is_done();
#ifdef UPCXX_UTILS_NO_THREAD_POOL
    assert(run_now && "is never active when UPCXX_UTILS_NO_THREAD_POOL");
#endif
    if (run_now) {
      // execute and return immediately
      func(args...);
      return make_future();
    }

    using return_t = typename std::invoke_result<Func, Args...>::type;
    static_assert(std::is_void<return_t>::value,
                  "void is the required return type for enqueue... did you mean enqueue_with_return()?");
    upcxx::persona &persona = upcxx::current_persona();
    std::shared_ptr<upcxx::promise<>> sh_prom = std::make_shared<upcxx::promise<>>();

    auto task_id = global_task_id()++;
    auto start_t = 0;
    DBG("sh_prom=", sh_prom.get(), " task_id=", task_id, "of", global_task_id(), "\n");

    auto args_tuple = std::make_tuple(args...);  // *copy* arguments to avoid races in argument references being reused
    auto sh_task =
        std::make_shared<Task>([sh_prom, task_id, start_t, &persona, func{std::move(func)}, args_tuple{std::move(args_tuple)}]() {
          auto compute_start_t = 0;
          duration_seconds delay_s = compute_start_t - start_t;
          DBG_VERBOSE("Executing sh_prom=", sh_prom.get(), "\n");
          std::apply(func, args_tuple);
          DBG_VERBOSE("Finished sh_prom=", sh_prom.get(), "\n");
          // fulfill only in calling persona
          persona.lpc_ff([task_id, start_t, compute_start_t, delay_s, sh_prom]() {
            duration_seconds s = 0 - compute_start_t;
            DBG("Fulfilled sh_prom=", sh_prom.get(), " task_id=", task_id, "of", global_task_id(), " in ", delay_s.count(), " delay + ", s.count(), " s\n");
            sh_prom->fulfill_anonymous(1);
            global_tasks_completed()++;
          });
        });
    enqueue_task(sh_task);
    return sh_prom->get_future();
  }

  template <class Func, class... Args>
  auto enqueue(Func &&func, Args &&... args) {
    using result_t = typename std::invoke_result<Func, Args...>::type;
    if constexpr (std::is_void<result_t>::value)
      return enqueue_no_return(std::forward<Func>(func), std::forward<Args>(args)...);
    else
      return enqueue_with_return(std::forward<Func>(func), std::forward<Args>(args)...);
  }

  void join_workers();

  void reset(int num_workers);
};

//
// methods in upcxx_utils namespace
//

// executes in a separate thread in the singleton ThreadPool and sets the maximum worker threads
template <typename Func, class... Args>
auto execute_in_thread_pool(int num_workers, Func &&func, Args &&... args) {
  return ThreadPool::enqueue_in_single_pool(num_workers, std::forward<Func>(func), std::forward<Args>(args)...);
};

// executes in a separate thread in the singleton ThreadPool
template <typename Func, class... Args>
auto execute_in_thread_pool(Func &&func, Args &&... args) {
  return execute_in_thread_pool(-1, std::forward<Func>(func), std::forward<Args>(args)...);
};

// executes in a separate thread in the singleton ThreadPool
// All actions will happen as if they were called in series regardless of the size of the ThreadPool
template <typename Func, class... Args>
upcxx::future<> execute_serially_in_thread_pool(Func &&func, Args &&... args) {
  return ThreadPool::enqueue_serially(std::forward<Func>(func), std::forward<Args>(args)...);
};

// Create a new temporary single threaded thread pool with lifetime of the task
template <typename Func, class... Args>
auto execute_in_new_thread(Func &&func, Args &&... args) {
  using result_t = typename std::invoke_result<Func, Args...>::type;
#ifdef UPCXX_UTILS_NO_THREAD_POOL
  // execute immediately
  if constexpr (std::is_void<result_t>::value) {
    return func(args...);
  } else {
    func(args...);
    return;
  }
#else
  auto sh_tp = make_shared<ThreadPool>(1);
  if constexpr (std::is_void<result_t>::value) {
    upcxx::future<> fut = sh_tp->enqueue_no_return(std::forward<Func>(func), std::forward<Args>(args)...);
    return fut.then([sh_tp]() {});
  } else {
    upcxx::future<result_t> fut = sh_tp->enqueue_with_return(std::forward<Func>(func), std::forward<Args>(args)...);
    return fut.then([sh_tp](result_t &&res) { return res; });
  }
#endif
}

};  // namespace upcxx_utils

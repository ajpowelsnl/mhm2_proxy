#include "upcxx_utils/thread_pool.hpp"

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

#include <atomic>
#include <functional>
#include <memory>
#include <queue>
#include <type_traits>
#include <upcxx/upcxx.hpp>
#include <vector>

#ifdef UPCXX_UTILS_NO_THREAD_POOL
#else
#include <condition_variable>
#include <mutex>
#include <thread>
#endif

using upcxx::future;
using upcxx::promise;

class upcxx_utils::ThreadPool_detail {
 public:
  // the threads
#ifdef UPCXX_UTILS_NO_THREAD_POOL
  std::vector<int> workers;
#else
  std::vector<std::thread> workers;
#endif
  //std::vector<IntermittentTimer> worker_timers;

  // the task queue
  using Task = std::function<void()>;  // void(void) function/lambda wrappers
#ifdef UPCXX_UTILS_NO_THREAD_POOL
  int task_mutex, workers_mutex, task_ready;
#else
  using Mutex = std::mutex;
  using Lock = std::unique_lock<Mutex>;
  Mutex task_mutex, workers_mutex;
  std::condition_variable task_ready;
#endif
  std::queue<Task> tasks;
  std::atomic<int> max_workers;
  std::atomic<int> ready_workers;

  ThreadPool_detail()
      : workers()
      , tasks()
      , task_mutex()
      , workers_mutex()
      , task_ready()
      , max_workers(0)
      , ready_workers(0) {
    assert(upcxx::master_persona().active_with_caller() && "Called from master persona");
  }

  ~ThreadPool_detail() { join_workers(); }

  // task queue state accessors for condition variable wake up
  bool is_ready() const { return !tasks.empty() | is_stop(); }
  bool is_terminal() const { return tasks.empty() & is_stop(); }

  // threadpool state accessors
  // true if the threadpool is no longer accepting new tasks to run asynchrously
  bool is_stop() const { return max_workers.load() == 0; }
  // true if the threadpool is no longer running any tasks
  bool is_done() const { return is_stop() & tasks.empty(); }

  void set_max_workers(int new_max_workers) {
    assert(upcxx::master_persona().active_with_caller() && "Called from master persona while upcxx is still active");
#ifdef UPCXX_UTILS_NO_THREAD_POOL
    assert(workers.empty() && "Never have workers when UPCXX_UTILS_NO_THREAD_POOL");
    assert(is_stop());
    return;  // spawn no threads
#else
    if (new_max_workers != max_workers.load()) {
      Lock lock(workers_mutex);  // modifying workers so lock
      if (new_max_workers == 0 && !workers.empty()) DIE("Cannot set max_workers to 0 before joining workers");
      max_workers.store(new_max_workers);
      workers.reserve(new_max_workers);
      //worker_timers.reserve(new_max_workers);
    }
#endif
  }

  // may be called by any thread, possibly recursively...
  void enqueue_task(std::shared_ptr<Task> sh_task) {
    bool run_now = is_stop();
    bool needs_one_more = false;
    if (!run_now) {
#if UPCXX_UTILS_NO_THREAD_POOL
      assert(workers.empty() && "Never have workers when UPCXX_UTILS_NO_THREAD_POOL");
      assert(is_stop());
      DIE("Invalid state!\n");
#else
      Lock lock(task_mutex);
      if (is_stop()) {
        DBG("State changed while acquiring task_mutex lock. ThreadPool is no longer active or it terminal\n");
        run_now = true;
      } else {
        tasks.emplace([sh_task]() { (*sh_task)(); });
        assert(!tasks.empty() && "Lock works tasks cannot be empty before returning it");
        needs_one_more = (workers.size() < max_workers) & (ready_workers.load() == 0);
        // lazy construct new threads for the pool with task lock!
        if (needs_one_more) {
          add_one_worker();  // acquires worker lock, worker thread will block until task lock has been released...
        }
      }
#endif
    }
    if (run_now) {
      // execute immediately, after draining
      do {
        assert(is_stop());
        if (is_done()) break;
        upcxx::progress();
      } while (true);
      (*sh_task)();
    } else {
#ifdef UPCXX_UTILS_NO_THREAD_POOL
      assert(workers.empty() && "Never have workers when UPCXX_UTILS_NO_THREAD_POOL");
      assert(is_stop());
#else
      task_ready.notify_one();
#endif
    }
  }

  void join_workers() {
#ifdef UPCXX_UTILS_NO_THREAD_POOL
    assert(workers.empty() && "Never have workers when UPCXX_UTILS_NO_THREAD_POOL");
    assert(is_stop());
#else
    {
      Lock lock(workers_mutex);  // modifying workers, so lock

      max_workers.store(0);  // signal to drain idle workers
      assert(is_stop());

      // notify all after signal
      task_ready.notify_all();

      if (workers.empty()) {
        //assert(worker_timers.empty());
        if (upcxx::initialized()) DBG("No workers to join\n");
      } else {
        assert(upcxx::master_persona().active_with_caller() && "Called from master persona while upcxx is still active");
        if (upcxx::initialized()) DBG("Joining ", workers.size(), " workers, ", tasks.size(), " tasks enqueued\n");

        // join all
        for (auto &worker : workers) {
          task_ready.notify_all();
          worker.join();
        }
        assert(ready_workers.load() == 0 && "All workers completed");
        workers.clear();
      }

      // keep worker lock, get task lock to prevent new tasks from starting
      Lock lock2(task_mutex);
      if (!tasks.empty()) {
        assert(upcxx::master_persona().active_with_caller() && "Called from master persona while upcxx is still active");
        WARN("Running ", tasks.size(), " outstanding tasks after workers all joined\n");
        while (!tasks.empty()) {
          auto &task = tasks.front();
          task();
          tasks.pop();
        }
      }
    }
#endif
    assert(is_stop() && "ThreadPool is now stopped");
    assert(workers.empty() && "Workers are all joined");
    assert(tasks.empty() && "All tasks are complete");
    assert(is_done() && "ThreadPool is now done");
    //for (auto &timer : worker_timers) {
    //  assert(upcxx::master_persona().active_with_caller() && "Called from master persona while upcxx is still active");
     // LOG("Worker joined after executing ", timer.get_count(), " tasks over ", timer.get_elapsed(), " s.\n");
   // }
    //worker_timers.clear();
  }

  void reset(int num_workers) {
    if (upcxx::initialized()) assert(upcxx::master_persona().active_with_caller() && "Called from master persona");
    join_workers();
    assert(is_done() && "ThreadPool is stopped and done");
    set_max_workers(num_workers);
  }

 protected:
  // may be called by any thread (via enqueue task)
  void add_one_worker() {
#ifdef UPCXX_UTILS_NO_THREAD_POOL
    assert(workers.empty() && "Never have workers when UPCXX_UTILS_NO_THREAD_POOL");
    assert(is_stop());
#else
    Lock lock(workers_mutex);                                 // modifying workers, so lock
    if (is_stop() | (workers.size() >= max_workers)) return;  // cannot exceed the max_workers
    //worker_timers.emplace_back("ThreadWorkerTimer");
    //workers.emplace_back([this, &timer = worker_timers.back()] {
      workers.emplace_back([this] {
      DBG_VERBOSE(std::this_thread::get_id(), " just started\n");
      //auto start_time = Timer::now();
      duration_seconds wait_for_max(0.25);
      while (true) {
        Task task;
        bool needs_notify = false;
        {
          this->ready_workers++;
          Lock lock(this->task_mutex);
          // avoid possible race conditions in worker life cycle
          // periodically wake up workers to check for is_stop or available tasks
          while (!this->task_ready.wait_for(lock, wait_for_max, [this] { return this->is_ready(); }))
            ;
          assert(this->is_ready() && "Woke thread has something to do");
          this->ready_workers--;
          if (this->is_terminal()) {
            break;
          }

          // pop the next task
          assert(!this->tasks.empty() && "Woke non-terminal thread has a task to do");
          task = std::move(this->tasks.front());
          this->tasks.pop();
          needs_notify = (this->ready_workers.load() > 0) & !this->tasks.empty();
        }
        if (needs_notify) this->task_ready.notify_one();  // just popped one of several tasks, and a worker was observed ready
        //DBG_VERBOSE(std::this_thread::get_id(), " popped task# ", timer.get_count(), "\n");
        //timer.start();
        task();  // execute
        //DBG_VERBOSE(std::this_thread::get_id(), " done with task# ", timer.get_count(), " in ", timer.get_elapsed_since_start(),
                   // " s\n");
        //timer.stop();
        this->task_ready.notify_one();  // this thread will soon be ready for a new task itself
      }
      //duration_seconds lifetime = Timer::now() - start_time;
      //DBG(std::this_thread::get_id(), " terminated. executed ", timer.get_count(), " tasks in ", timer.get_elapsed(),
      //    " s, lifetime ", lifetime.count(), " s\n");
      this->task_ready.notify_all();  // exiting, notify any straggling threads too
    });
#endif
    DBG_VERBOSE("Added a new worker: ", workers.back().get_id(), "\n");
  }
};

// bool upcxx_utils::ThreadPool::is_ready() const { return tp_detail->is_ready(); }
// bool upcxx_utils::ThreadPool::is_terminal() const { return tp_detail->is_terminal(); }

void upcxx_utils::ThreadPool::enqueue_task(std::shared_ptr<Task> sh_task) { tp_detail->enqueue_task(sh_task); }

upcxx_utils::ThreadPool &upcxx_utils::ThreadPool::get_single_pool(int num_threads) {
  static ThreadPool _the_singleton_pool_(num_threads <= 0 ? 1 : num_threads);
  if (num_threads < 0 && _the_singleton_pool_.tp_detail->max_workers == 0)
    num_threads = 1;  // ensure 1 worker if called by default without num_threads
  if (num_threads > 0) _the_singleton_pool_.tp_detail->set_max_workers(num_threads);
  return _the_singleton_pool_;
}
void upcxx_utils::ThreadPool::join_single_pool() {
  if (upcxx::initialized()) DBG("joining single pool\n");
  get_single_pool(0).join_workers();
}

upcxx_utils::ThreadPool::ThreadPool(int num_threads)
    : tp_detail(std::make_unique<ThreadPool_detail>()) {
  assert(upcxx::master_persona().active_with_caller() && "Called from master persona");
  // reserve and start the threads in the pool
  tp_detail->set_max_workers(num_threads);
  DBG("Constructed with ", num_threads, "\n");
}

upcxx_utils::ThreadPool::~ThreadPool() { join_workers(); }

void upcxx_utils::ThreadPool::yield_if_needed() {
#ifdef UPCXX_UTILS_NO_THREAD_POOL
#else
  if (tasks_outstanding() > 0) std::this_thread::yield();
#endif
}

void upcxx_utils::ThreadPool::join_workers() {
  if (upcxx::initialized()) DBG("join workers\n");
  tp_detail->join_workers();
}

void upcxx_utils::ThreadPool::reset(int num_workers) { tp_detail->reset(num_workers); }

bool upcxx_utils::ThreadPool::is_done() const { return tp_detail->is_done(); }

int upcxx_utils::ThreadPool::get_max_workers() const { return tp_detail->max_workers.load(); }

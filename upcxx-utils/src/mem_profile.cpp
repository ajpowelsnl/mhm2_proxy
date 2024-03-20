#include "upcxx_utils/mem_profile.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <math.h>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/timers.hpp"
#include "upcxx_utils/version.h"

using namespace std;

namespace upcxx_utils {

std::ofstream _memlog;

void open_memlog(string name) {
  assert(!_memlog.is_open());
  // only 1 process per node opens
  if (upcxx::local_team().rank_me()) return;

  if (!get_rank_path(name, upcxx::rank_me())) DIE("Could not get rank_path for: ", name, "\n");

  bool old_file = file_exists(name);
  DBG("Opening ", name, " old_file=", old_file, "\n");
  _memlog.open(name, std::ofstream::out | std::ofstream::app);
  if (!_memlog.is_open()) DIE("Could not open: ", name, "\n");
  LOG_MEM("Open");
}

void close_memlog() {
  if (_memlog.is_open()) {
    if (upcxx::initialized()) LOG_MEM("Close");
    _memlog.flush();
    _memlog.close();
  }
  assert(!_memlog.is_open());
}

double get_free_mem(void) {
  string buf;
  ifstream f("/proc/meminfo");
  double mem_free = 0;
  while (!f.eof()) {
    getline(f, buf);
    if (buf.find("MemFree") == 0 || buf.find("Buffers") == 0 || buf.find("Cached") == 0) {
      stringstream fields;
      string units;
      string name;
      double mem;
      fields << buf;
      fields >> name >> mem >> units;
      if (units[0] == 'k') mem *= 1024;
      mem_free += mem;
    }
  }
  return mem_free;
}

string get_self_stat(void) {
  std::stringstream buffer;
  std::ifstream i("/proc/self/stat");
  buffer << i.rdbuf();
  return buffer.str();
}

#define IN_NODE_TEAM() (!(upcxx::rank_me() % upcxx::local_team().rank_n()))

#ifndef UPCXX_UTILS_NO_THREADS

void MemoryTrackerThread::start() {
	upcxx::barrier();
  if (IN_NODE_TEAM()) start_free_mem = get_free_mem();
  start_free_mem = upcxx::broadcast(start_free_mem, 0, local_team()).wait();
  auto msm_fut = upcxx_utils::min_sum_max_reduce_one(start_free_mem, 0);

  auto thread_lambda = [&] {
    ofstream _tracker_file;
    ofstream &tracker_file = ::upcxx_utils::_memlog.is_open() ? ::upcxx_utils::_memlog : _tracker_file;
    if (!tracker_file.is_open() && !tracker_filename.empty() && IN_NODE_TEAM()) {
      get_rank_path(tracker_filename, upcxx::rank_me());
      tracker_file.open(tracker_filename, ios_base::out | ios_base::app);
      if (!tracker_file.is_open() || !tracker_file.good()) DIE("Could not open tracker file:", tracker_filename);
      opened = true;
    }

    double prev_free_mem = 0;
    LOG_MEM_OS(tracker_file, "MemTracker start");
    while (!fin) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms));
      double free_mem = get_free_mem();
      // only report memory if it changed sufficiently - otherwise this produces a great deal of
      // gumpf in the logs
      if (fabs(free_mem - prev_free_mem) > ONE_GB) {
        DBGLOG("MemoryTrackerThread free_mem=", get_size_str(free_mem), "\n");
        prev_free_mem = free_mem;
        flush_logger();
      }
      if (free_mem < min_free_mem) min_free_mem = free_mem;
      if (tracker_file.is_open()) {
        LOG_MEM_OS(tracker_file, "MemTracker");
      }
      if (ticks % 60 * 1000 / sample_ms == 0) {
        flush_logger();
        if (tracker_file.is_open()) tracker_file.flush();
      }
      ticks++;
    }
    LOG_MEM_OS(tracker_file, "MemTracker end");

    if (tracker_file.is_open()) tracker_file.flush();
    if (opened) {
      tracker_file.close();
      opened = false;
    }
  };

  if (IN_NODE_TEAM()) {
    t = new std::thread(thread_lambda);
  }

  auto msm = msm_fut.wait();
  upcxx::barrier(local_team());
  double delta_mem;
  if (IN_NODE_TEAM()) delta_mem = start_free_mem - get_free_mem();
  delta_mem = upcxx::broadcast(delta_mem, 0, local_team()).wait();
  auto msm_fut2 = upcxx_utils::min_sum_max_reduce_one(delta_mem, 0);
  auto msm2 = msm_fut2.wait();

  int num_nodes = upcxx::rank_n() / upcxx::local_team().rank_n();
  SLOG("Initial free memory across all ", num_nodes, " nodes: ", get_size_str(msm.sum / upcxx::local_team().rank_n()), " (",
       get_size_str((double)msm.avg), " avg, ", get_size_str(msm.min), " min, ", get_size_str(msm.max), " max)\n");
  SLOG_VERBOSE("Change in free memory after reduction and thread construction ",
               get_size_str(msm2.sum / upcxx::local_team().rank_n()), " (", get_size_str((double)msm2.avg), " avg, ",
               get_size_str(msm2.min), " min, ", get_size_str(msm2.max), " max)\n");
  min_free_mem = start_free_mem;
  upcxx::barrier();
}

void MemoryTrackerThread::stop() {
  if (IN_NODE_TEAM()) {
    if (t) {
      fin = true;
      t->join();
      delete t;
    }
    t = nullptr;
  }
  upcxx::barrier(local_team());
  double peak_mem;
  if (IN_NODE_TEAM()) peak_mem = start_free_mem - min_free_mem;
  peak_mem = upcxx::broadcast(peak_mem, 0, local_team()).wait();
  auto msm = upcxx_utils::min_sum_max_reduce_one(peak_mem, 0).wait();
  int num_nodes = upcxx::rank_n() / upcxx::local_team().rank_n();
  SLOG("Peak memory used across all ", num_nodes, " nodes: ", get_size_str(msm.sum / upcxx::local_team().rank_n()), " (",
       get_size_str((double)msm.avg), " avg, ", get_size_str(msm.min), " min, ", get_size_str(msm.max), " max)\n");
  upcxx::barrier();
}

#endif
};  // namespace upcxx_utils

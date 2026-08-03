// A persistent worker pool.
//
// The first version of parallel_for spawned std::threads per op. On a model
// where a whole inference takes 50 microseconds, thread creation dominates
// everything: MNIST went from 48us single-threaded to 80us on four threads.
// Workers are created once and parked on a condition variable instead.
#include "nanoinfer/pool.h"

#include <algorithm>
#include <functional>

namespace ni {

ThreadPool& ThreadPool::instance() {
  static ThreadPool pool;
  return pool;
}

ThreadPool::~ThreadPool() { shutdown(); }

void ThreadPool::resize(int n) {
  n = std::max(1, n);
  std::unique_lock<std::mutex> lk(mu_);
  if (n - 1 == static_cast<int>(workers_.size())) return;
  lk.unlock();
  shutdown();
  lk.lock();
  stop_ = false;
  // n includes the calling thread, so only n-1 workers are needed
  for (int i = 0; i < n - 1; ++i)
    workers_.emplace_back([this, i] { worker_loop(i); });
}

void ThreadPool::shutdown() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_)
    if (t.joinable()) t.join();
  workers_.clear();
}

void ThreadPool::worker_loop(int slot) {
  for (;;) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return stop_ || (job_ && slot < claimed_ && !done_[slot]); });
    if (stop_) return;
    auto* fn = job_;
    const int begin = ranges_[static_cast<size_t>(slot)].first;
    const int end = ranges_[static_cast<size_t>(slot)].second;
    lk.unlock();

    if (begin < end) (*fn)(begin, end);

    lk.lock();
    done_[slot] = true;
    ++finished_;
    lk.unlock();
    finished_cv_.notify_one();
  }
}

void ThreadPool::run(int n, const std::function<void(int, int)>& body) {
  const int workers = static_cast<int>(workers_.size());
  const int slices = std::min(workers + 1, std::max(1, n));
  if (slices <= 1 || workers == 0) {
    body(0, n);
    return;
  }

  const int chunk = (n + slices - 1) / slices;
  {
    std::lock_guard<std::mutex> lk(mu_);
    job_ = &body;
    claimed_ = slices - 1;  // the caller takes the last slice itself
    ranges_.assign(static_cast<size_t>(claimed_), {0, 0});
    done_.assign(static_cast<size_t>(claimed_), false);
    finished_ = 0;
    for (int s = 0; s < claimed_; ++s)
      ranges_[static_cast<size_t>(s)] = {s * chunk, std::min(n, (s + 1) * chunk)};
  }
  cv_.notify_all();

  // the calling thread does the tail slice rather than idling
  const int mine_begin = claimed_ * chunk;
  if (mine_begin < n) body(mine_begin, n);

  std::unique_lock<std::mutex> lk(mu_);
  finished_cv_.wait(lk, [&] { return finished_ >= claimed_; });
  job_ = nullptr;
  claimed_ = 0;
}

}  // namespace ni

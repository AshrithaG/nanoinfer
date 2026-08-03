#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace ni {

// Workers are created once and reused. Sized by set_threads(); a size of 1
// means everything runs inline on the calling thread with no synchronization.
class ThreadPool {
 public:
  static ThreadPool& instance();
  ~ThreadPool();

  void resize(int n);
  int size() const { return static_cast<int>(workers_.size()) + 1; }
  // splits [0, n) across the pool and blocks until every slice is done
  void run(int n, const std::function<void(int, int)>& body);

 private:
  ThreadPool() = default;
  void worker_loop(int slot);
  void shutdown();

  std::vector<std::thread> workers_;
  std::vector<std::pair<int, int>> ranges_;
  std::vector<char> done_;
  const std::function<void(int, int)>* job_ = nullptr;
  int claimed_ = 0;
  int finished_ = 0;
  bool stop_ = false;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable finished_cv_;
};

}  // namespace ni

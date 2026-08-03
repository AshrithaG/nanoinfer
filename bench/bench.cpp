// Benchmark harness. Prints one CSV row per configuration so the ladder can be
// assembled by the plotting script.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "nanoinfer/graph.h"

using namespace ni;
using clk = std::chrono::steady_clock;

namespace {

struct Timing {
  double median_us, p99_us, mean_us;
};

Timing measure(const std::function<void()>& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  std::vector<double> us;
  us.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const auto t0 = clk::now();
    fn();
    const auto t1 = clk::now();
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  std::sort(us.begin(), us.end());
  Timing t;
  t.median_us = us[us.size() / 2];
  t.p99_us = us[std::min(us.size() - 1, static_cast<size_t>(us.size() * 0.99))];
  t.mean_us = std::accumulate(us.begin(), us.end(), 0.0) / static_cast<double>(us.size());
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  int iters = 200, warmup = 20;
  std::vector<int> thread_counts = {1};
  std::string label = "engine";
  bool ladder = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (a == "--label" && i + 1 < argc) label = argv[++i];
    else if (a == "--ladder") ladder = true;
    else if (a == "--threads" && i + 1 < argc) {
      thread_counts.clear();
      std::string s = argv[++i];
      size_t pos = 0;
      while (!s.empty()) {
        pos = s.find(',');
        thread_counts.push_back(std::atoi(s.substr(0, pos).c_str()));
        if (pos == std::string::npos) break;
        s = s.substr(pos + 1);
      }
    } else if (a.rfind("--", 0) != 0) model_path = a;
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "usage: ni_bench model.ngm [--iters N] [--threads 1,2,4] "
                         "[--ladder] [--label name]\n");
    return 2;
  }

  auto model = Model::load(model_path);
  const size_t n = numel(model->input_shape());
  std::mt19937 rng(0);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> input(n);
  for (auto& v : input) v = dist(rng);

  std::printf("model,variant,threads,median_us,p99_us,mean_us,arena_kb,naive_kb,"
              "workspace_kb,reused\n");

  const auto& plan = model->plan();
  auto emit = [&](const std::string& variant, int threads, const Timing& t) {
    std::printf("%s,%s,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d\n", model->name().c_str(),
                variant.c_str(), threads, t.median_us, t.p99_us, t.mean_us,
                plan.arena_bytes / 1024.0, plan.sum_of_activation_bytes / 1024.0,
                plan.workspace_bytes / 1024.0, plan.reused_buffers);
  };

  auto bench_current = [&](const std::string& variant) {
    for (int th : thread_counts) {
      set_threads(th);
      const auto t = measure([&] { model->run(input.data(), input.size()); }, warmup, iters);
      emit(variant, th, t);
    }
  };

  if (ladder && !model->quantized()) {
    model->force_conv_impl(ConvImpl::Naive);
    bench_current("naive_conv");
    model->force_conv_impl(ConvImpl::Im2colGemm);
    bench_current("im2col_gemm");
  } else {
    bench_current(label);
  }
  return 0;
}

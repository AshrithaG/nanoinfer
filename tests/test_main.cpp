// Dependency-free test harness. No network fetch in CI, no framework to learn.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanoinfer/graph.h"
#include "nanoinfer/ops.h"

using namespace ni;

namespace {

int g_failures = 0;
int g_checks = 0;
std::string g_current;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL [%s] %s\n", g_current.c_str(), what.c_str());
  }
}

void check_close(float a, float b, float tol, const std::string& what) {
  const bool ok = std::fabs(a - b) <= tol;
  check(ok, ok ? what : what + " (" + std::to_string(a) + " vs " + std::to_string(b) + ")");
}

struct Case {
  std::string name;
  std::function<void()> fn;
};
std::vector<Case>& cases() {
  static std::vector<Case> c;
  return c;
}
struct Reg {
  Reg(const char* n, std::function<void()> f) { cases().push_back({n, std::move(f)}); }
};

#define TEST(name)                                     \
  static void name();                                  \
  static Reg reg_##name(#name, name);                  \
  static void name()

std::vector<float> randvec(size_t n, unsigned seed = 0) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

Tensor wrap(std::vector<float>& v, Shape s) {
  Tensor t;
  t.data = v.data();
  t.shape = std::move(s);
  t.dtype = DType::F32;
  return t;
}

// ------------------------------------------------------------------ tests

TEST(gemm_matches_naive) {
  const int M = 17, N = 23, K = 31;
  auto A = randvec(static_cast<size_t>(M) * K, 1);
  auto B = randvec(static_cast<size_t>(K) * N, 2);
  std::vector<float> C1(static_cast<size_t>(M) * N), C2(static_cast<size_t>(M) * N);
  gemm_naive(M, N, K, A.data(), B.data(), C1.data());
  gemm(M, N, K, A.data(), B.data(), C2.data());
  float worst = 0.f;
  for (size_t i = 0; i < C1.size(); ++i) worst = std::max(worst, std::fabs(C1[i] - C2[i]));
  check_close(worst, 0.f, 1e-3f, "gemm vs naive max abs diff");
}

// The whole point of having two conv paths is that they must agree. If they
// diverge, the fast path is wrong and every benchmark number is meaningless.
TEST(im2col_conv_matches_naive_conv) {
  struct Cfg { int C, H, W, OC, KH, KW, sh, sw, ph, pw, groups; };
  const Cfg cfgs[] = {
      {3, 12, 12, 8, 3, 3, 1, 1, 1, 1, 1},
      {4, 9, 9, 6, 3, 3, 2, 2, 1, 1, 1},
      {1, 28, 28, 16, 5, 5, 1, 1, 2, 2, 1},
      {8, 7, 7, 8, 3, 3, 1, 1, 0, 0, 1},
      {6, 10, 10, 12, 3, 3, 1, 1, 1, 1, 3},  // grouped
      // non-square kernel with asymmetric stride and padding: the keyword
      // spotting stem. Collapsing these to one value per axis is a silent
      // wrong-answer bug, so it gets its own case.
      {1, 49, 10, 16, 10, 4, 2, 2, 5, 1, 1},
  };
  for (const auto& c : cfgs) {
    const int OH = (c.H + 2 * c.ph - c.KH) / c.sh + 1;
    const int OW = (c.W + 2 * c.pw - c.KW) / c.sw + 1;
    auto xv = randvec(static_cast<size_t>(c.C) * c.H * c.W, 3);
    auto wv = randvec(static_cast<size_t>(c.OC) * (c.C / c.groups) * c.KH * c.KW, 4);
    auto bv = randvec(static_cast<size_t>(c.OC), 5);
    std::vector<float> o1(static_cast<size_t>(c.OC) * OH * OW);
    std::vector<float> o2(o1.size());

    Tensor x = wrap(xv, {1, c.C, c.H, c.W});
    Tensor w = wrap(wv, {c.OC, c.C / c.groups, c.KH, c.KW});
    Tensor b = wrap(bv, {c.OC});
    Tensor out1 = wrap(o1, {1, c.OC, OH, OW});
    Tensor out2 = wrap(o2, {1, c.OC, OH, OW});

    ConvParams p;
    p.stride_h = c.sh;
    p.stride_w = c.sw;
    p.pad_h = c.ph;
    p.pad_w = c.pw;
    p.groups = c.groups;

    conv2d(x, w, &b, out1, p, ConvImpl::Naive, nullptr, 0);
    const size_t need = conv2d_workspace_floats(x, w, out2, p, ConvImpl::Im2colGemm);
    std::vector<float> ws(need);
    conv2d(x, w, &b, out2, p, ConvImpl::Im2colGemm, ws.data(), ws.size());

    float worst = 0.f;
    for (size_t i = 0; i < o1.size(); ++i) worst = std::max(worst, std::fabs(o1[i] - o2[i]));
    check_close(worst, 0.f, 2e-3f,
                "im2col vs naive, C=" + std::to_string(c.C) + " k=" +
                    std::to_string(c.KH) + "x" + std::to_string(c.KW) + " s=" +
                    std::to_string(c.sh) + "," + std::to_string(c.sw) + " p=" +
                    std::to_string(c.ph) + "," + std::to_string(c.pw) + " g=" +
                    std::to_string(c.groups));
  }
}

TEST(depthwise_matches_naive_grouped_conv) {
  const int C = 12, H = 14, W = 14, K = 3, stride = 1, pad = 1;
  const int OH = (H + 2 * pad - K) / stride + 1;
  auto xv = randvec(static_cast<size_t>(C) * H * W, 6);
  auto wv = randvec(static_cast<size_t>(C) * K * K, 7);
  auto bv = randvec(static_cast<size_t>(C), 8);
  std::vector<float> o1(static_cast<size_t>(C) * OH * OH), o2(o1.size());

  Tensor x = wrap(xv, {1, C, H, W});
  Tensor w = wrap(wv, {C, 1, K, K});
  Tensor b = wrap(bv, {C});
  Tensor out1 = wrap(o1, {1, C, OH, OH});
  Tensor out2 = wrap(o2, {1, C, OH, OH});
  ConvParams p;
  p.stride_h = p.stride_w = stride;
  p.pad_h = p.pad_w = pad;
  p.groups = C;

  conv2d(x, w, &b, out1, p, ConvImpl::Naive, nullptr, 0);
  conv2d(x, w, &b, out2, p, ConvImpl::DepthwiseNeon, nullptr, 0);
  float worst = 0.f;
  for (size_t i = 0; i < o1.size(); ++i) worst = std::max(worst, std::fabs(o1[i] - o2[i]));
  check_close(worst, 0.f, 2e-3f, "depthwise vs naive grouped");
}

TEST(depthwise_matches_with_stride_two) {
  const int C = 8, H = 15, W = 15, K = 3, stride = 2, pad = 1;
  const int OH = (H + 2 * pad - K) / stride + 1;
  auto xv = randvec(static_cast<size_t>(C) * H * W, 9);
  auto wv = randvec(static_cast<size_t>(C) * K * K, 10);
  std::vector<float> o1(static_cast<size_t>(C) * OH * OH), o2(o1.size());
  Tensor x = wrap(xv, {1, C, H, W});
  Tensor w = wrap(wv, {C, 1, K, K});
  Tensor out1 = wrap(o1, {1, C, OH, OH});
  Tensor out2 = wrap(o2, {1, C, OH, OH});
  ConvParams p;
  p.stride_h = p.stride_w = stride;
  p.pad_h = p.pad_w = pad;
  p.groups = C;
  conv2d(x, w, nullptr, out1, p, ConvImpl::Naive, nullptr, 0);
  conv2d(x, w, nullptr, out2, p, ConvImpl::DepthwiseNeon, nullptr, 0);
  float worst = 0.f;
  for (size_t i = 0; i < o1.size(); ++i) worst = std::max(worst, std::fabs(o1[i] - o2[i]));
  check_close(worst, 0.f, 2e-3f, "depthwise stride 2 vs naive");
}

TEST(fused_relu_equals_separate_relu) {
  const int C = 4, H = 8, W = 8, OC = 6, K = 3;
  auto xv = randvec(static_cast<size_t>(C) * H * W, 11);
  auto wv = randvec(static_cast<size_t>(OC) * C * K * K, 12);
  auto bv = randvec(static_cast<size_t>(OC), 13);
  std::vector<float> o1(static_cast<size_t>(OC) * H * W), o2(o1.size());
  Tensor x = wrap(xv, {1, C, H, W});
  Tensor w = wrap(wv, {OC, C, K, K});
  Tensor b = wrap(bv, {OC});
  Tensor out1 = wrap(o1, {1, OC, H, W});
  Tensor out2 = wrap(o2, {1, OC, H, W});
  ConvParams p;
  p.pad_h = p.pad_w = 1;
  const size_t need = conv2d_workspace_floats(x, w, out1, p, ConvImpl::Im2colGemm);
  std::vector<float> ws(need);

  conv2d(x, w, &b, out1, p, ConvImpl::Im2colGemm, ws.data(), ws.size());
  relu(out1);
  p.relu = true;
  conv2d(x, w, &b, out2, p, ConvImpl::Im2colGemm, ws.data(), ws.size());
  float worst = 0.f;
  for (size_t i = 0; i < o1.size(); ++i) worst = std::max(worst, std::fabs(o1[i] - o2[i]));
  check_close(worst, 0.f, 1e-5f, "fused relu vs separate relu");
}

TEST(threading_does_not_change_results) {
  const int C = 6, H = 20, W = 20, OC = 10, K = 3;
  auto xv = randvec(static_cast<size_t>(C) * H * W, 14);
  auto wv = randvec(static_cast<size_t>(OC) * C * K * K, 15);
  std::vector<float> o1(static_cast<size_t>(OC) * H * W), o2(o1.size());
  Tensor x = wrap(xv, {1, C, H, W});
  Tensor w = wrap(wv, {OC, C, K, K});
  Tensor out1 = wrap(o1, {1, OC, H, W});
  Tensor out2 = wrap(o2, {1, OC, H, W});
  ConvParams p;
  p.pad_h = p.pad_w = 1;

  set_threads(1);
  conv2d(x, w, nullptr, out1, p, ConvImpl::Naive, nullptr, 0);
  set_threads(4);
  conv2d(x, w, nullptr, out2, p, ConvImpl::Naive, nullptr, 0);
  set_threads(1);
  float worst = 0.f;
  for (size_t i = 0; i < o1.size(); ++i) worst = std::max(worst, std::fabs(o1[i] - o2[i]));
  check_close(worst, 0.f, 1e-6f, "1 thread vs 4 threads");
}

TEST(maxpool_and_softmax_behave) {
  std::vector<float> xv = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<float> ov(4);
  Tensor x = wrap(xv, {1, 1, 4, 4});
  Tensor o = wrap(ov, {1, 1, 2, 2});
  maxpool2d(x, o, 2, 2, 0);
  check_close(ov[0], 6.f, 1e-6f, "maxpool top-left");
  check_close(ov[3], 16.f, 1e-6f, "maxpool bottom-right");

  std::vector<float> lv = {1.f, 2.f, 3.f};
  std::vector<float> sv(3);
  Tensor l = wrap(lv, {1, 3});
  Tensor s = wrap(sv, {1, 3});
  softmax(l, s);
  check_close(sv[0] + sv[1] + sv[2], 1.f, 1e-6f, "softmax sums to one");
  check(sv[2] > sv[1] && sv[1] > sv[0], "softmax preserves order");
}

TEST(quantize_dequantize_roundtrip_is_close) {
  auto xv = randvec(256, 16);
  float amax = 0.f;
  for (float v : xv) amax = std::max(amax, std::fabs(v));
  std::vector<int8_t> qv(xv.size());
  std::vector<float> back(xv.size());

  Tensor x = wrap(xv, {1, 256});
  Tensor q;
  q.data = qv.data();
  q.shape = {1, 256};
  q.dtype = DType::I8;
  q.scale = amax / 127.f;
  q.zero_point = 0;
  Tensor d = wrap(back, {1, 256});

  quantize(x, q);
  dequantize(q, d);
  float worst = 0.f;
  for (size_t i = 0; i < xv.size(); ++i) worst = std::max(worst, std::fabs(xv[i] - back[i]));
  check(worst <= q.scale, "int8 roundtrip within one quantization step");
}

TEST(memory_planner_reuses_buffers_without_aliasing) {
  // Model::load runs check_no_aliasing internally and throws if the plan hands
  // a node's input buffer back as its output, so simply loading these is the
  // regression test for the bug that made chained Linear layers wrong.
  for (const char* path : {"models/mnist_cnn.ngm", "models/kws_dscnn.ngm",
                           "models/vww_mobilenet.ngm"}) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
      std::printf("  skip [memory_planner] %s not exported yet\n", path);
      continue;
    }
    std::fclose(f);
    try {
      auto m = Model::load(path);
      const auto& p = m->plan();
      check(p.arena_bytes > 0, std::string(path) + ": arena allocated");
      check(p.arena_bytes <= p.sum_of_activation_bytes,
            std::string(path) + ": planner never exceeds the naive sum");
      check(p.reused_buffers > 0, std::string(path) + ": some buffer got recycled");
    } catch (const std::exception& e) {
      check(false, std::string(path) + ": load failed: " + e.what());
    }
  }
}

}  // namespace

int main() {
  std::printf("nanoinfer tests (%zu cases)\n", cases().size());
  for (auto& c : cases()) {
    g_current = c.name;
    c.fn();
  }
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

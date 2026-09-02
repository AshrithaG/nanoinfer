// Fused conv+bias+ReLU against cuDNN, on layer shapes from edge CNNs.

#include "nanoinfer/cuda_ops.h"

#ifdef NI_WITH_CUDA

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace ni::cuda;

namespace {

const ConvImplCuda kImpls[] = {
    ConvImplCuda::Direct, ConvImplCuda::DirectFused, ConvImplCuda::SmemFused,
    ConvImplCuda::CudnnUnfused, ConvImplCuda::CudnnFused,
    ConvImplCuda::CudnnFusedF32};

struct Case { const char* label; ConvShape s; };

// Shapes taken from the model families the CPU engine targets: an early wide
// layer, a mid-network 3x3, a pointwise 1x1, and a depthwise 3x3.
const Case kCases[] = {
    {"first 3x3   (3->32, 96x96)",   {1, 3, 96, 96, 32, 3, 3, 1, 1, false}},
    {"mid 3x3     (64->64, 56x56)",  {1, 64, 56, 56, 64, 3, 3, 1, 1, false}},
    {"pointwise   (128->128, 28x28)",{1, 128, 28, 28, 128, 1, 1, 1, 0, false}},
    {"depthwise   (128, 28x28)",     {1, 128, 28, 28, 128, 3, 3, 1, 1, true}},
};

struct Buf {
  std::vector<float> hX, hW, hB, hY, hRef;
  float *dX = nullptr, *dW = nullptr, *dB = nullptr, *dY = nullptr;
  explicit Buf(const ConvShape& s)
      : hX(s.in_elems()), hW(s.w_elems()),
        hB(s.depthwise ? s.C : s.K), hY(s.out_elems()), hRef(s.out_elems()) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& v : hX) v = d(rng);
    for (auto& v : hW) v = d(rng);
    for (auto& v : hB) v = d(rng);
    dX = static_cast<float*>(device_alloc(hX.size() * 4));
    dW = static_cast<float*>(device_alloc(hW.size() * 4));
    dB = static_cast<float*>(device_alloc(hB.size() * 4));
    dY = static_cast<float*>(device_alloc(hY.size() * 4));
    upload(dX, hX.data(), hX.size() * 4);
    upload(dW, hW.data(), hW.size() * 4);
    upload(dB, hB.data(), hB.size() * 4);
  }
  ~Buf() { device_free(dX); device_free(dW); device_free(dB); device_free(dY); }
};

// Relative error, because these are float accumulations in different orders.
double max_rel_err(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    const double scale = std::fmax(1.0, std::fabs(double(b[i])));
    worst = std::fmax(worst, d / scale);
  }
  return worst;
}

}  // namespace

int main() {
  const DeviceInfo d = device_info();
  std::printf("device: %s  sm_%d%d  %d SMs\n", d.name, d.major, d.minor, d.sms);
  std::printf("cuDNN: %s\n\n", cudnn_available() ? "available" : "NOT BUILT IN");

  for (const auto& c : kCases) {
    const ConvShape& s = c.s;
    Buf b(s);
    std::printf("%s  ->  %dx%dx%d, %.1f MFLOP\n", c.label,
                s.depthwise ? s.C : s.K, s.out_h(), s.out_w(),
                2.0 * s.out_elems() * (s.depthwise ? s.R * s.S : size_t(s.C) * s.R * s.S) / 1e6);

    // reference: our own fused kernel, checked against cuDNN where possible
    bool have_ref = false;
    double cudnn_fused_ms = -1, best_mine_ms = 1e30;

    for (auto impl : kImpls) {
      const double ms = time_conv2d_f32(s, b.dX, b.dW, b.dB, b.dY, true, impl, 50);
      if (ms < 0) { std::printf("    %-15s declined\n", conv_impl_name(impl)); continue; }
      download(b.hY.data(), b.dY, b.hY.size() * 4);

      std::string note;
      if (!have_ref) { b.hRef = b.hY; have_ref = true; note = "reference"; }
      else {
        const double err = max_rel_err(b.hY, b.hRef);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "rel err %.2e%s", err,
                      err < 1e-4 ? "" : (err < 5e-2 ? "  (TF32)" : "  <-- CHECK"));
        note = buf;
      }
      std::printf("    %-15s %8.4f ms   %s\n", conv_impl_name(impl), ms, note.c_str());

      // Compare like for like: cuDNN's true-fp32 fused path, which is the only
      // vendor row computing the same arithmetic our kernels do.
      const bool is_cudnn = impl == ConvImplCuda::CudnnUnfused ||
                            impl == ConvImplCuda::CudnnFused ||
                            impl == ConvImplCuda::CudnnFusedF32;
      if (impl == ConvImplCuda::CudnnFusedF32) cudnn_fused_ms = ms;
      else if (!is_cudnn && ms < best_mine_ms) best_mine_ms = ms;
    }
    if (cudnn_fused_ms > 0 && best_mine_ms < 1e29) {
      std::printf("    best mine vs cuDNN fused f32: %.2fx\n",
                  cudnn_fused_ms / best_mine_ms);
    }
    std::printf("\n");
  }
  return 0;
}

#else
int main() { std::printf("built without CUDA\n"); return 0; }
#endif

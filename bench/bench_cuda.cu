// int8 GEMM benchmark: the ladder, and the gap to cuBLAS.
//
// Correctness first. Every implementation is checked against a CPU int32
// reference on a small shape before anything is timed, because a fast wrong
// kernel is worse than a slow right one.

#include "nanoinfer/cuda_ops.h"

#ifdef NI_WITH_CUDA

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace ni::cuda;

namespace {

void cpu_ref(int M, int N, int K, const int8_t* A, const int8_t* B, int32_t* C) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<int32_t>(A[m * K + k]) * static_cast<int32_t>(B[k * N + n]);
      }
      C[m * N + n] = acc;
    }
  }
}

struct Buffers {
  std::vector<int8_t> hA, hB;
  std::vector<int32_t> hC, hRef;
  int8_t *dA = nullptr, *dB = nullptr;
  int32_t* dC = nullptr;

  Buffers(int M, int N, int K, unsigned seed) : hA(size_t(M) * K), hB(size_t(K) * N),
                                                hC(size_t(M) * N), hRef(size_t(M) * N) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(-127, 127);
    for (auto& v : hA) v = static_cast<int8_t>(d(rng));
    for (auto& v : hB) v = static_cast<int8_t>(d(rng));
    dA = static_cast<int8_t*>(device_alloc(hA.size()));
    dB = static_cast<int8_t*>(device_alloc(hB.size()));
    dC = static_cast<int32_t*>(device_alloc(hC.size() * sizeof(int32_t)));
    upload(dA, hA.data(), hA.size());
    upload(dB, hB.data(), hB.size());
  }
  ~Buffers() { device_free(dA); device_free(dB); device_free(dC); }
};

const I8GemmImpl kImpls[] = {I8GemmImpl::Naive, I8GemmImpl::Tiled,
                             I8GemmImpl::TiledDp4a, I8GemmImpl::Wmma,
                             I8GemmImpl::WmmaSmem, I8GemmImpl::Cublas};

bool verify_shape(int M, int N, int K, const char* label) {
  std::printf("\n  %s (M=%d N=%d K=%d):\n", label, M, N, K);
  Buffers b(M, N, K, 1234);
  cpu_ref(M, N, K, b.hA.data(), b.hB.data(), b.hRef.data());

  bool all_ok = true;
  for (auto impl : kImpls) {
    if (!gemm_i8(M, N, K, b.dA, b.dB, b.dC, impl)) {
      std::printf("    %-12s declined this shape\n", i8_gemm_impl_name(impl));
      continue;
    }
    device_sync();
    download(b.hC.data(), b.dC, b.hC.size() * sizeof(int32_t));
    long bad = 0;
    for (size_t i = 0; i < b.hC.size(); ++i) if (b.hC[i] != b.hRef[i]) ++bad;
    std::printf("    %-12s %s (%ld/%zu mismatched)\n", i8_gemm_impl_name(impl),
                bad ? "FAIL" : "exact", bad, b.hC.size());
    all_ok = all_ok && bad == 0;
  }
  return all_ok;
}

bool verify() {
  // Two shapes. The ragged one catches indexing that only works when the shape
  // divides the tile size; cuBLAS and wmma decline it, which is itself part of
  // the comparison. The aligned one is the only way to check those two.
  bool ok = verify_shape(61, 76, 132, "ragged");
  ok = verify_shape(64, 80, 128, "aligned") && ok;
  ok = verify_shape(128, 128, 256, "block-aligned") && ok;
  return ok;
}

void sweep() {
  const int sizes[] = {256, 512, 1024, 2048, 4096};
  const int iters_for[] = {200, 100, 50, 20, 10};

  std::printf("\n\n%-7s", "size");
  for (auto impl : kImpls) std::printf("%14s", i8_gemm_impl_name(impl));
  std::printf("%12s%12s\n", "best TOPS", "vs cuBLAS");
  std::printf("%s\n", "-----------------------------------------------------------------------------------------------------------");

  for (size_t s = 0; s < sizeof(sizes) / sizeof(int); ++s) {
    const int n = sizes[s];
    Buffers b(n, n, n, 7);
    std::printf("%-7d", n);
    double cublas_ms = 0, mine_best_ms = 1e30;
    for (auto impl : kImpls) {
      const double ms = time_gemm_i8(n, n, n, b.dA, b.dB, b.dC, impl, iters_for[s]);
      if (ms < 0) { std::printf("%14s", "n/a"); continue; }
      std::printf("%12.3f ms", ms);
      if (impl == I8GemmImpl::Cublas) cublas_ms = ms;
      else if (ms < mine_best_ms) mine_best_ms = ms;
    }
    // 2*M*N*K flops; int8 MACs count as two ops, same convention as cuBLAS docs
    const double ops = 2.0 * n * n * n;
    const double best_tops = ops / (mine_best_ms * 1e-3) / 1e12;
    std::printf("%12.2f%11.2fx\n", best_tops, cublas_ms / mine_best_ms);
  }
}

}  // namespace

int main() {
  const DeviceInfo d = device_info();
  std::printf("device: %s  sm_%d%d  %d SMs  %.0f GB/s peak\n",
              d.name, d.major, d.minor, d.sms, d.mem_bandwidth_gbps);

  std::printf("\ncorrectness:");
  if (!verify()) {
    std::printf("\nabort: a kernel disagrees with the CPU reference\n");
    return 1;
  }
  sweep();
  std::printf("\nvs cuBLAS > 1.00x means the hand-written kernel wins.\n");
  return 0;
}

#else
int main() { std::printf("built without CUDA\n"); return 0; }
#endif

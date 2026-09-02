// fp32 NCHW convolution, with and without a fused bias + ReLU epilogue.
//
// The CPU engine in this repo runs bias and activation as a separate pass over
// the output tensor, and its README lists that as one of the reasons it trails
// ONNX Runtime. On a GPU the cost is easier to see: the unfused version writes
// the output, then reads and writes it again for bias, then a third time for
// ReLU. Three trips through memory where one would do. This file measures how
// much that is worth, and what cuDNN does with the same shapes.

#include "nanoinfer/cuda_ops.h"

#ifdef NI_WITH_CUDA

#include <cuda_runtime.h>
#ifdef NI_WITH_CUDNN
#include <cudnn.h>
#endif

#include <cstdio>
#include <cstdlib>

namespace ni::cuda {
namespace {

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "cuda error in %s: %s\n", what, cudaGetErrorString(e));
    std::abort();
  }
}

// ------------------------------------------------------------------ direct
// One thread per output element. `fuse` is a compile-time switch so the fused
// and unfused kernels are the same code with the epilogue included or not,
// which keeps the comparison honest: nothing else differs between them.
template <bool kFuse>
__global__ void k_conv(int N, int C, int H, int W, int K, int R, int S,
                       int stride, int pad, int OH, int OW, bool depthwise,
                       const float* __restrict__ X, const float* __restrict__ Wt,
                       const float* __restrict__ bias, float* __restrict__ Y,
                       bool relu) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int outC = depthwise ? C : K;
  const int total = N * outC * OH * OW;
  if (idx >= total) return;

  const int ow = idx % OW;
  const int oh = (idx / OW) % OH;
  const int k = (idx / (OW * OH)) % outC;
  const int n = idx / (OW * OH * outC);

  float acc = 0.f;
  if (depthwise) {
    // one input channel per output channel
    for (int r = 0; r < R; ++r) {
      const int ih = oh * stride - pad + r;
      if (ih < 0 || ih >= H) continue;
      for (int s = 0; s < S; ++s) {
        const int iw = ow * stride - pad + s;
        if (iw < 0 || iw >= W) continue;
        acc += X[((size_t(n) * C + k) * H + ih) * W + iw] * Wt[(size_t(k) * R + r) * S + s];
      }
    }
  } else {
    for (int c = 0; c < C; ++c) {
      for (int r = 0; r < R; ++r) {
        const int ih = oh * stride - pad + r;
        if (ih < 0 || ih >= H) continue;
        for (int s = 0; s < S; ++s) {
          const int iw = ow * stride - pad + s;
          if (iw < 0 || iw >= W) continue;
          acc += X[((size_t(n) * C + c) * H + ih) * W + iw] *
                 Wt[((size_t(k) * C + c) * R + r) * S + s];
        }
      }
    }
  }

  if constexpr (kFuse) {
    if (bias) acc += bias[k];
    if (relu) acc = acc > 0.f ? acc : 0.f;
  }
  Y[idx] = acc;
}

// The two extra passes the unfused path needs, so the comparison includes the
// real cost of not fusing rather than pretending the epilogue is free.
__global__ void k_bias(int total, int planeElems, int outC, float* Y,
                       const float* __restrict__ bias) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  Y[idx] += bias[(idx / planeElems) % outC];
}

__global__ void k_relu(int total, float* Y) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  Y[idx] = Y[idx] > 0.f ? Y[idx] : 0.f;
}

// -------------------------------------------------------------- smem fused
// Weights are the same for every output pixel, so every thread in the block
// re-reads them from global memory in the kernels above. For a 3x3 layer with
// C=K=128 that is 147 KB re-read per block. Staging them once per block is the
// obvious fix; it only works when the filter fits in shared memory, so the
// launcher falls back when it does not.
extern __shared__ float smem[];

__global__ void k_conv_smem(int N, int C, int H, int W, int K, int R, int S,
                            int stride, int pad, int OH, int OW,
                            const float* __restrict__ X, const float* __restrict__ Wt,
                            const float* __restrict__ bias, float* __restrict__ Y,
                            bool relu) {
  const int k = blockIdx.y;                 // one block row per output channel
  const size_t wCount = size_t(C) * R * S;
  for (size_t i = threadIdx.x; i < wCount; i += blockDim.x) {
    smem[i] = Wt[size_t(k) * wCount + i];
  }
  __syncthreads();

  const int pix = blockIdx.x * blockDim.x + threadIdx.x;
  const int planeElems = OH * OW;
  if (pix >= N * planeElems) return;
  const int ow = pix % OW;
  const int oh = (pix / OW) % OH;
  const int n = pix / planeElems;

  float acc = 0.f;
  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      const int ih = oh * stride - pad + r;
      if (ih < 0 || ih >= H) continue;
      for (int s = 0; s < S; ++s) {
        const int iw = ow * stride - pad + s;
        if (iw < 0 || iw >= W) continue;
        acc += X[((size_t(n) * C + c) * H + ih) * W + iw] * smem[(size_t(c) * R + r) * S + s];
      }
    }
  }
  if (bias) acc += bias[k];
  if (relu) acc = acc > 0.f ? acc : 0.f;
  Y[(size_t(n) * K + k) * planeElems + oh * OW + ow] = acc;
}

#ifdef NI_WITH_CUDNN
void ck(cudnnStatus_t s, const char* what) {
  if (s != CUDNN_STATUS_SUCCESS) {
    std::fprintf(stderr, "cudnn error in %s: %s\n", what, cudnnGetErrorString(s));
    std::abort();
  }
}

cudnnHandle_t cudnn() {
  static cudnnHandle_t h = nullptr;
  if (!h) ck(cudnnCreate(&h), "cudnnCreate");
  return h;
}

// Everything cuDNN needs to describe one convolution, built once per shape and
// cached, so the timing loop measures the convolution rather than descriptor
// construction.
struct CudnnPlan {
  cudnnTensorDescriptor_t x{}, y{}, b{};
  cudnnFilterDescriptor_t w{};
  cudnnConvolutionDescriptor_t conv{};
  cudnnActivationDescriptor_t act{};
  cudnnConvolutionFwdAlgo_t algo{};
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  bool valid = false;
};

// Two cached plans: one left on cuDNN's default math, which on Ampere and later
// silently promotes fp32 convolution to TF32 tensor cores, and one forced to
// CUDNN_FMA_MATH so the arithmetic actually matches our kernels. TF32 keeps 10
// mantissa bits against fp32's 23, which is worth roughly 1e-3 of relative
// error and a large amount of speed. Reporting only one of these would be
// comparing different number formats and calling it a benchmark.
CudnnPlan& plan_for(const ConvShape& s, bool true_fp32) {
  static CudnnPlan plans[2];
  static ConvShape cachedShapes[2]{};
  static bool haveFlag[2] = {false, false};
  const int slot = true_fp32 ? 1 : 0;
  CudnnPlan& p = plans[slot];
  ConvShape& cached = cachedShapes[slot];
  bool& have = haveFlag[slot];

  const bool same = have && __builtin_memcmp(&cached, &s, sizeof(ConvShape)) == 0;
  if (same) return p;

  if (have) {
    cudnnDestroyTensorDescriptor(p.x); cudnnDestroyTensorDescriptor(p.y);
    cudnnDestroyTensorDescriptor(p.b); cudnnDestroyFilterDescriptor(p.w);
    cudnnDestroyConvolutionDescriptor(p.conv); cudnnDestroyActivationDescriptor(p.act);
    if (p.workspace) cudaFree(p.workspace);
    p = CudnnPlan{};
  }
  cached = s; have = true;

  const int outC = s.depthwise ? s.C : s.K;
  const int groups = s.depthwise ? s.C : 1;
  const int filterC = s.depthwise ? 1 : s.C;

  ck(cudnnCreateTensorDescriptor(&p.x), "createX");
  ck(cudnnSetTensor4dDescriptor(p.x, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                s.N, s.C, s.H, s.W), "setX");
  ck(cudnnCreateTensorDescriptor(&p.y), "createY");
  ck(cudnnSetTensor4dDescriptor(p.y, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                s.N, outC, s.out_h(), s.out_w()), "setY");
  ck(cudnnCreateTensorDescriptor(&p.b), "createB");
  ck(cudnnSetTensor4dDescriptor(p.b, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                1, outC, 1, 1), "setB");
  ck(cudnnCreateFilterDescriptor(&p.w), "createW");
  ck(cudnnSetFilter4dDescriptor(p.w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                outC, filterC, s.R, s.S), "setW");
  ck(cudnnCreateConvolutionDescriptor(&p.conv), "createConv");
  ck(cudnnSetConvolution2dDescriptor(p.conv, s.pad, s.pad, s.stride, s.stride,
                                     1, 1, CUDNN_CROSS_CORRELATION,
                                     CUDNN_DATA_FLOAT), "setConv");
  ck(cudnnSetConvolutionGroupCount(p.conv, groups), "groups");
  ck(cudnnSetConvolutionMathType(p.conv,
                                 true_fp32 ? CUDNN_FMA_MATH : CUDNN_DEFAULT_MATH),
     "mathType");
  ck(cudnnCreateActivationDescriptor(&p.act), "createAct");
  ck(cudnnSetActivationDescriptor(p.act, CUDNN_ACTIVATION_RELU,
                                  CUDNN_NOT_PROPAGATE_NAN, 0.0), "setAct");

  // The fused entry point only accepts a subset of algorithms; asking for this
  // one keeps both cuDNN paths on the same algorithm so the fusion delta is the
  // only thing being measured.
  p.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  if (cudnnGetConvolutionForwardWorkspaceSize(cudnn(), p.x, p.w, p.conv, p.y,
                                              p.algo, &p.workspace_bytes) !=
      CUDNN_STATUS_SUCCESS) {
    p.valid = false;
    return p;
  }
  if (p.workspace_bytes) ck(cudaMalloc(&p.workspace, p.workspace_bytes), "wsMalloc");
  p.valid = true;
  return p;
}
#endif  // NI_WITH_CUDNN

constexpr int kThreads = 256;

}  // namespace

const char* conv_impl_name(ConvImplCuda impl) {
  switch (impl) {
    case ConvImplCuda::Direct: return "direct";
    case ConvImplCuda::DirectFused: return "direct+fused";
    case ConvImplCuda::SmemFused: return "smem+fused";
    case ConvImplCuda::CudnnUnfused: return "cuDNN unfused";
    case ConvImplCuda::CudnnFused: return "cuDNN fused";
    case ConvImplCuda::CudnnFusedF32: return "cuDNN fused f32";
  }
  return "?";
}

bool cudnn_available() {
#ifdef NI_WITH_CUDNN
  return true;
#else
  return false;
#endif
}

bool conv2d_f32(const ConvShape& s, const float* dX, const float* dW,
                const float* dBias, float* dY, bool relu, ConvImplCuda impl) {
  const int OH = s.out_h(), OW = s.out_w();
  const int outC = s.depthwise ? s.C : s.K;
  const int total = static_cast<int>(s.out_elems());
  const int blocks = (total + kThreads - 1) / kThreads;

  switch (impl) {
    case ConvImplCuda::Direct: {
      k_conv<false><<<blocks, kThreads>>>(s.N, s.C, s.H, s.W, s.K, s.R, s.S,
                                          s.stride, s.pad, OH, OW, s.depthwise,
                                          dX, dW, dBias, dY, relu);
      // the two passes that fusion removes
      if (dBias) k_bias<<<blocks, kThreads>>>(total, OH * OW, outC, dY, dBias);
      if (relu) k_relu<<<blocks, kThreads>>>(total, dY);
      ck(cudaGetLastError(), "direct");
      return true;
    }
    case ConvImplCuda::DirectFused: {
      k_conv<true><<<blocks, kThreads>>>(s.N, s.C, s.H, s.W, s.K, s.R, s.S,
                                         s.stride, s.pad, OH, OW, s.depthwise,
                                         dX, dW, dBias, dY, relu);
      ck(cudaGetLastError(), "direct fused");
      return true;
    }
    case ConvImplCuda::SmemFused: {
      if (s.depthwise) return false;  // per-channel filters, nothing to share
      const size_t wBytes = size_t(s.C) * s.R * s.S * sizeof(float);
      if (wBytes > 48u * 1024u) return false;  // will not fit in shared memory
      const dim3 grid((s.N * OH * OW + kThreads - 1) / kThreads, s.K);
      k_conv_smem<<<grid, kThreads, wBytes>>>(s.N, s.C, s.H, s.W, s.K, s.R, s.S,
                                              s.stride, s.pad, OH, OW,
                                              dX, dW, dBias, dY, relu);
      ck(cudaGetLastError(), "smem fused");
      return true;
    }
#ifdef NI_WITH_CUDNN
    case ConvImplCuda::CudnnUnfused: {
      CudnnPlan& p = plan_for(s, false);
      if (!p.valid) return false;
      const float alpha = 1.f, beta = 0.f;
      ck(cudnnConvolutionForward(cudnn(), &alpha, p.x, dX, p.w, dW, p.conv,
                                 p.algo, p.workspace, p.workspace_bytes,
                                 &beta, p.y, dY), "convForward");
      if (dBias) {
        const float one = 1.f;
        ck(cudnnAddTensor(cudnn(), &one, p.b, dBias, &one, p.y, dY), "addTensor");
      }
      if (relu) {
        const float a = 1.f, b = 0.f;
        ck(cudnnActivationForward(cudnn(), p.act, &a, p.y, dY, &b, p.y, dY), "act");
      }
      return true;
    }
    case ConvImplCuda::CudnnFused:
    case ConvImplCuda::CudnnFusedF32: {
      if (!dBias || !relu) return false;  // this entry point needs both
      CudnnPlan& p = plan_for(s, impl == ConvImplCuda::CudnnFusedF32);
      if (!p.valid) return false;
      const float alpha = 1.f, beta = 0.f;
      ck(cudnnConvolutionBiasActivationForward(
             cudnn(), &alpha, p.x, dX, p.w, dW, p.conv, p.algo,
             p.workspace, p.workspace_bytes, &beta, p.y, dY,
             p.b, dBias, p.act, p.y, dY), "convBiasAct");
      return true;
    }
#else
    case ConvImplCuda::CudnnUnfused:
    case ConvImplCuda::CudnnFused:
    case ConvImplCuda::CudnnFusedF32: return false;
#endif
  }
  return false;
}

double time_conv2d_f32(const ConvShape& s, const float* dX, const float* dW,
                       const float* dBias, float* dY, bool relu,
                       ConvImplCuda impl, int iters) {
  if (!conv2d_f32(s, dX, dW, dBias, dY, relu, impl)) return -1.0;
  ck(cudaDeviceSynchronize(), "warmup");

  cudaEvent_t a, b;
  ck(cudaEventCreate(&a), "ev"); ck(cudaEventCreate(&b), "ev");
  ck(cudaEventRecord(a), "rec");
  for (int i = 0; i < iters; ++i) conv2d_f32(s, dX, dW, dBias, dY, relu, impl);
  ck(cudaEventRecord(b), "rec");
  ck(cudaEventSynchronize(b), "sync");
  float ms = 0.f;
  ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
  cudaEventDestroy(a); cudaEventDestroy(b);
  return static_cast<double>(ms) / iters;
}


namespace {
inline void check_probe(cudaError_t e) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "cuda probe error: %s\n", cudaGetErrorString(e));
  }
}
}  // namespace

// ------------------------------------------------------ occupancy reporting
namespace {

// Everything here comes from the CUDA runtime, not from a profiler. On a shared
// machine GPU performance counters are usually restricted to root, but these
// two calls are not, and they give the same theoretical occupancy figure Nsight
// reports on its first page.
template <typename Fn>
KernelInfo probe_kernel(const char* name, Fn fn, int block_size, size_t dyn_shared) {
  cudaFuncAttributes attr{};
  check_probe(cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(fn)));

  int blocks = 0;
  check_probe(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, reinterpret_cast<const void*>(fn), block_size, dyn_shared));

  cudaDeviceProp prop{};
  check_probe(cudaGetDeviceProperties(&prop, 0));

  KernelInfo k{};
  std::snprintf(k.name, sizeof(k.name), "%s", name);
  k.regs = attr.numRegs;
  k.static_shared = static_cast<int>(attr.sharedSizeBytes);
  k.dynamic_shared = static_cast<int>(dyn_shared);
  k.block_size = block_size;
  k.blocks_per_sm = blocks;
  k.occupancy = prop.maxThreadsPerMultiProcessor
                    ? double(blocks * block_size) / prop.maxThreadsPerMultiProcessor
                    : 0.0;
  return k;
}

}  // namespace

int conv_kernel_report(KernelInfo* out, int cap) {
  int n = 0;
  // The smem kernel's shared use is dynamic and shape dependent; 128 channels
  // of 3x3 fp32 weights is the pointwise/mid case in the benchmark.
  const size_t smem_bytes = size_t(128) * 3 * 3 * sizeof(float);
  if (n < cap) out[n++] = probe_kernel("conv direct", k_conv<false>, kThreads, 0);
  if (n < cap) out[n++] = probe_kernel("conv direct+fused", k_conv<true>, kThreads, 0);
  if (n < cap) out[n++] = probe_kernel("conv smem+fused", k_conv_smem, kThreads, smem_bytes);
  return n;
}

}  // namespace ni::cuda

#endif  // NI_WITH_CUDA

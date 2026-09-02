// CUDA backend. Same contract as the CPU kernels in ops.h: a reference
// implementation that is obviously correct, plus faster paths measured against
// it and against the vendor library. Nothing here is selected automatically;
// the benchmark picks a path by name so the ladder is a real measurement.
#pragma once

#ifdef NI_WITH_CUDA

#include <cstdint>
#include <cstddef>

namespace ni::cuda {

// Which int8 GEMM to run. Ordered by how much work went into it.
enum class I8GemmImpl {
  Naive,      // one thread per output element, global memory only
  Tiled,      // shared-memory tiles, one output per thread
  TiledDp4a,  // shared-memory tiles + __dp4a, four int8 MACs per instruction
  Wmma,       // IMMA tensor cores via the wmma API, m16n16k16
  WmmaSmem,   // same, but tiles staged through shared memory
  Cublas,     // cublasGemmEx, CUDA_R_8I in / CUDA_R_32I out
};

const char* i8_gemm_impl_name(I8GemmImpl impl);

// C[M,N] int32 = A[M,K] int8 * B[K,N] int8, all row-major, all device pointers.
// K must be a multiple of 4 for TiledDp4a; the caller pads. Returns false if
// the implementation cannot run the given shape.
bool gemm_i8(int M, int N, int K,
             const int8_t* dA, const int8_t* dB, int32_t* dC,
             I8GemmImpl impl);

// Device memory helpers. Thin wrappers so the benchmark does not have to
// include cuda_runtime.h itself.
void* device_alloc(size_t bytes);
void  device_free(void* p);
void  upload(void* dst, const void* src, size_t bytes);
void  download(void* dst, const void* src, size_t bytes);
void  device_sync();

// Wall time of `iters` back-to-back launches, in milliseconds, measured with
// CUDA events. Excludes the warmup launch.
double time_gemm_i8(int M, int N, int K,
                    const int8_t* dA, const int8_t* dB, int32_t* dC,
                    I8GemmImpl impl, int iters);


// ---------------------------------------------------------------- conv2d
// fp32 NCHW convolution with optional fused bias and ReLU. The CPU engine runs
// bias and activation as a separate pass over the output tensor; on a GPU that
// pass is pure memory traffic, so fusing it is worth measuring directly.
enum class ConvImplCuda {
  Direct,       // conv only, one thread per output element
  DirectFused,  // + bias and ReLU folded into the same store
  SmemFused,    // + weights staged in shared memory
  CudnnUnfused,  // cudnnConvolutionForward, then bias, then ReLU
  CudnnFused,    // cudnnConvolutionBiasActivationForward
  CudnnFusedF32, // the same, forced to true fp32 instead of TF32
};

const char* conv_impl_name(ConvImplCuda impl);

struct ConvShape {
  int N, C, H, W;   // input
  int K, R, S;      // output channels, kernel height/width
  int stride, pad;
  bool depthwise;   // groups == C
  int out_h() const { return (H + 2 * pad - R) / stride + 1; }
  int out_w() const { return (W + 2 * pad - S) / stride + 1; }
  size_t out_elems() const {
    return size_t(N) * (depthwise ? C : K) * out_h() * out_w();
  }
  size_t in_elems() const { return size_t(N) * C * H * W; }
  size_t w_elems() const {
    return depthwise ? size_t(C) * R * S : size_t(K) * C * R * S;
  }
};

// Runs the convolution. Returns false if the implementation declines the shape.
bool conv2d_f32(const ConvShape& s, const float* dX, const float* dW,
                const float* dBias, float* dY, bool relu, ConvImplCuda impl);

double time_conv2d_f32(const ConvShape& s, const float* dX, const float* dW,
                       const float* dBias, float* dY, bool relu,
                       ConvImplCuda impl, int iters);

bool cudnn_available();

struct DeviceInfo {
  char name[256];
  int major, minor, sms;
  int clock_khz;
  double mem_bandwidth_gbps;
};
DeviceInfo device_info();

}  // namespace ni::cuda

#endif  // NI_WITH_CUDA

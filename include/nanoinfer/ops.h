// Kernels. Each op has a reference implementation that is obviously correct
// and, where it matters, a faster path selected at runtime. The reference
// version is never deleted: it is what the fast paths are tested against.
#pragma once

#include "nanoinfer/tensor.h"

namespace ni {

struct ConvParams {
  int stride_h = 1, stride_w = 1;
  int pad_h = 0, pad_w = 0;
  int groups = 1;
  bool relu = false;  // fused activation
};

// Which conv implementation to use. Selected per node, benchmarked separately.
enum class ConvImpl { Naive, Im2colGemm, DepthwiseNeon };

// out = conv2d(x, w) + b. x is NCHW, w is OIHW.
void conv2d(const Tensor& x, const Tensor& w, const Tensor* b, Tensor& out,
            const ConvParams& p, ConvImpl impl, float* workspace,
            size_t workspace_floats);

size_t conv2d_workspace_floats(const Tensor& x, const Tensor& w, const Tensor& out,
                               const ConvParams& p, ConvImpl impl);

// General matrix multiply: C[MxN] = A[MxK] * B[KxN], row-major.
// Uses Accelerate/BLAS when available, otherwise a blocked fallback.
void gemm(int M, int N, int K, const float* A, const float* B, float* C);
void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C);

// `w_t` is the [K, N] transpose of w, precomputed at load time. Transposing on
// every call costs O(K*N) copies per inference, which on a small model is a
// bigger line item than the matmul itself.
void linear(const Tensor& x, const Tensor& w, const Tensor* b, Tensor& out, bool relu,
            const float* w_t);
void relu(Tensor& x);
void add(const Tensor& a, const Tensor& b, Tensor& out, bool relu);
void maxpool2d(const Tensor& x, Tensor& out, int k, int stride, int pad);
void avgpool_global(const Tensor& x, Tensor& out);
void adaptive_maxpool1d_like(const Tensor& x, Tensor& out);
void softmax(const Tensor& x, Tensor& out);
void flatten_copy(const Tensor& x, Tensor& out);

// int8 quantized paths
void quantize(const Tensor& x, Tensor& out);
void dequantize(const Tensor& x, Tensor& out);
// int8 conv with per-output-channel weight scales, int32 accumulation.
void conv2d_i8(const Tensor& x, const Tensor& w, const Tensor* bias_f32, Tensor& out,
               const ConvParams& p, const float* w_scales, float* workspace,
               size_t workspace_floats);
void linear_i8(const Tensor& x, const Tensor& w, const Tensor* bias_f32, Tensor& out,
               const float* w_scales, bool relu);

int threads_used();
void set_threads(int n);

}  // namespace ni

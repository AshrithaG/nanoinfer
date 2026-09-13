// The int8 GEMM kernels and cuBLAS, callable from Python, for tools/triton_gemm.py.
//
// ni_bench_cuda times these kernels in its own process. Setting its numbers next
// to Triton's from a Python process would compare different runs, timers and
// launch paths, so this library lets one process time everything on the same
// stream under the same timer.
//
// It includes gemm_i8.cu rather than linking against it for one reason: the
// public gemm_i8() launches on the default stream, and CUDA graph capture needs
// launches on the capturing stream. Including the source reaches the same
// kernels with a stream argument and changes nothing inside them.

#include "../src/cuda/gemm_i8.cu"

#include <cstddef>

namespace {

void* g_workspace = nullptr;
size_t g_workspace_bytes = 0;

// Return codes: 0 ran, 1 declined the shape, 2 CUDA or cuBLAS error. Nothing here
// aborts; the caller is a Python process that should record a failure and go on.
int launched() { return cudaGetLastError() == cudaSuccess ? 0 : 2; }

}  // namespace

extern "C" {

// impl: 0 tiled+dp4a, 1 wmma, 2 wmma+smem. Same launch geometry as gemm_i8().
int ni_gemm_i8(int impl, int M, int N, int K, const void* A, const void* B, void* C,
               void* stream) {
  using namespace ni::cuda;
  const auto* a = static_cast<const int8_t*>(A);
  const auto* b = static_cast<const int8_t*>(B);
  auto* c = static_cast<int32_t*>(C);
  const auto s = static_cast<cudaStream_t>(stream);

  switch (impl) {
    case 0: {
      if (K % 4) return 1;
      const dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
      k_tiled_dp4a<<<grid, dim3(kTile, kTile), 0, s>>>(M, N, K, a, b, c);
      return launched();
    }
    case 1: {
      if (M % 16 || N % 16 || K % 16) return 1;
      k_wmma<<<dim3((N + 31) / 32, (M + 31) / 32), dim3(128), 0, s>>>(M, N, K, a, b, c);
      return launched();
    }
    case 2: {
      if (M % kBM || N % kBN || K % kBK) return 1;
      k_wmma_smem<<<dim3(N / kBN, M / kBM), dim3(128), 0, s>>>(M, N, K, a, b, c);
      return launched();
    }
  }
  return 1;
}

// layout 0: B is B[K,N] row-major. This is the exact call gemm_i8.cu makes (NN).
// layout 1: B is passed as Bt[N,K] row-major, so both operands are K-contiguous
//           (TN), the layout int8 tensor-core GEMMs are built around.
// Both compute C^T = B^T A^T in cuBLAS's column-major view, which is row-major C.
int ni_cublas_i8(int layout, int M, int N, int K, const void* A, const void* B, void* C,
                 void* stream) {
  if ((N % 4) != 0 || (K % 4) != 0) return 1;  // same restriction gemm_i8.cu applies
  cublasHandle_t h = ni::cuda::cublas();
  if (cublasSetStream(h, static_cast<cudaStream_t>(stream)) != CUBLAS_STATUS_SUCCESS) return 2;
  // cublasSetStream resets the workspace to cuBLAS's own pool, and allocating from
  // that during graph capture is not allowed, so point it back at ours every call.
  if (g_workspace &&
      cublasSetWorkspace(h, g_workspace, g_workspace_bytes) != CUBLAS_STATUS_SUCCESS) {
    return 2;
  }
  const int32_t alpha = 1, beta = 0;
  const cublasOperation_t opB = layout == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int ldb = layout == 0 ? N : K;
  const cublasStatus_t st = cublasGemmEx(h, opB, CUBLAS_OP_N, N, M, K, &alpha,
                                         B, CUDA_R_8I, ldb,
                                         A, CUDA_R_8I, K, &beta,
                                         C, CUDA_R_32I, N,
                                         CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  if (st == CUBLAS_STATUS_NOT_SUPPORTED) return 1;
  return st == CUBLAS_STATUS_SUCCESS ? 0 : 2;
}

// Device buffer cuBLAS uses instead of allocating its own. Owned by the caller.
int ni_cublas_set_workspace(void* ptr, size_t bytes) {
  g_workspace = ptr;
  g_workspace_bytes = bytes;
  return 0;
}

}  // extern "C"

// int8 GEMM on CUDA.
//
// int8 is the case where quantization actually pays off on a GPU: sm_61 and
// later expose __dp4a, which does four int8 multiply-accumulates into an int32
// in one instruction. On the CPU side of this repo the int8 path wins mostly on
// memory traffic; here it should win on arithmetic too.
//
// Three implementations, kept side by side so the ladder is measured rather
// than remembered, plus cuBLAS as the thing to lose to honestly.

#include "nanoinfer/cuda_ops.h"

#ifdef NI_WITH_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>

namespace ni::cuda {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "cuda error in %s: %s\n", what, cudaGetErrorString(e));
    std::abort();
  }
}

void check(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublas error in %s: %d\n", what, static_cast<int>(s));
    std::abort();
  }
}

cublasHandle_t cublas() {
  static cublasHandle_t h = nullptr;
  if (!h) check(cublasCreate(&h), "cublasCreate");
  return h;
}

// ---------------------------------------------------------------- naive
// One thread per output element. Every thread walks all of K straight out of
// global memory, so each A row is re-read N times and each B column M times.
// This is the baseline the other two have to beat.
__global__ void k_naive(int M, int N, int K,
                        const int8_t* __restrict__ A,
                        const int8_t* __restrict__ B,
                        int32_t* __restrict__ C) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  int32_t acc = 0;
  for (int k = 0; k < K; ++k) {
    acc += static_cast<int32_t>(A[row * K + k]) * static_cast<int32_t>(B[k * N + col]);
  }
  C[row * N + col] = acc;
}

// ---------------------------------------------------------------- tiled
// Classic shared-memory tiling: each block cooperatively stages a TILE x TILE
// square of A and of B, so every element loaded from global memory is reused
// TILE times instead of once.
constexpr int kTile = 32;

__global__ void k_tiled(int M, int N, int K,
                        const int8_t* __restrict__ A,
                        const int8_t* __restrict__ B,
                        int32_t* __restrict__ C) {
  __shared__ int8_t sA[kTile][kTile];
  __shared__ int8_t sB[kTile][kTile];

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * kTile + ty;
  const int col = blockIdx.x * kTile + tx;

  int32_t acc = 0;
  for (int k0 = 0; k0 < K; k0 += kTile) {
    // Staging. Out-of-range reads become zeros, which contribute nothing to the
    // dot product, so the edge blocks need no separate code path.
    sA[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : int8_t{0};
    sB[ty][tx] = (col < N && k0 + ty < K) ? B[(k0 + ty) * N + col] : int8_t{0};
    __syncthreads();

#pragma unroll
    for (int k = 0; k < kTile; ++k) {
      acc += static_cast<int32_t>(sA[ty][k]) * static_cast<int32_t>(sB[k][tx]);
    }
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// ------------------------------------------------------------ tiled + dp4a
// Same tiling, but the inner loop reads four int8 values at a time as a packed
// int32 and uses __dp4a. That is one instruction for four MACs.
//
// A is packed along K, which is contiguous, so a thread can just reinterpret
// four consecutive bytes. B is not: consecutive k values are N apart. So B is
// staged transposed into shared memory, which makes its k axis contiguous there
// and lets the same trick work.
__global__ void k_tiled_dp4a(int M, int N, int K,
                             const int8_t* __restrict__ A,
                             const int8_t* __restrict__ B,
                             int32_t* __restrict__ C) {
  __shared__ int8_t sA[kTile][kTile];
  __shared__ int8_t sBT[kTile][kTile];  // [n][k], transposed on the way in

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * kTile + ty;
  const int col = blockIdx.x * kTile + tx;

  int32_t acc = 0;
  for (int k0 = 0; k0 < K; k0 += kTile) {
    sA[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : int8_t{0};
    // thread (tx,ty) stages B[k0+ty][blockCol+tx] into sBT[tx][ty]
    sBT[tx][ty] = (col < N && k0 + ty < K) ? B[(k0 + ty) * N + col] : int8_t{0};
    __syncthreads();

#pragma unroll
    for (int k = 0; k < kTile; k += 4) {
      // Both rows are 32 bytes and shared memory is 4-byte aligned per row,
      // so these reinterprets are safe for k a multiple of 4.
      const int32_t a4 = *reinterpret_cast<const int32_t*>(&sA[ty][k]);
      const int32_t b4 = *reinterpret_cast<const int32_t*>(&sBT[tx][k]);
      acc = __dp4a(a4, b4, acc);
    }
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}


// ---------------------------------------------------------------- wmma
// The previous three kernels all run on the CUDA cores. cuBLAS does not: it
// issues IMMA, the integer tensor-core instruction, which is a different and
// much wider unit of the chip. __dp4a does 4 MACs per instruction per thread;
// one IMMA does a 16x16x16 matrix multiply per warp. No amount of tuning on
// the CUDA cores closes that, so the honest response is to use the same unit.
//
// Each warp owns one 16x16 output tile. A block is four warps in a 2x2
// arrangement, so a block covers 32x32.
//
// The restriction this buys: wmma's int8 loads require the leading dimension to
// be a multiple of 16 bytes, so M, N and K must all be multiples of 16. Like
// cuBLAS, this kernel declines ragged shapes that the dp4a kernel accepts.
__global__ void k_wmma(int M, int N, int K,
                       const int8_t* __restrict__ A,
                       const int8_t* __restrict__ B,
                       int32_t* __restrict__ C) {
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int tileM = blockIdx.y * 32 + (warp / 2) * 16;
  const int tileN = blockIdx.x * 32 + (warp % 2) * 16;
  if (tileM >= M || tileN >= N) return;

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int32_t> acc;
  nvcuda::wmma::fill_fragment(acc, 0);

  for (int k = 0; k < K; k += 16) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, int8_t,
                           nvcuda::wmma::row_major> a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, int8_t,
                           nvcuda::wmma::row_major> b;
    nvcuda::wmma::load_matrix_sync(a, A + tileM * K + k, K);
    nvcuda::wmma::load_matrix_sync(b, B + k * N + tileN, N);
    nvcuda::wmma::mma_sync(acc, a, b, acc);
  }
  nvcuda::wmma::store_matrix_sync(C + tileM * N + tileN, acc, N,
                                  nvcuda::wmma::mem_row_major);
}


// ------------------------------------------------------------ wmma + shared
// k_wmma loads every fragment straight from global memory, so at 4096 each A
// row is re-read by every block along N and each B column by every block along
// M. Measured traffic works out to roughly 8.6 GB against a 1008 GB/s bus,
// which is the entire runtime: the kernel is bandwidth-bound, not compute-bound,
// even though it is issuing tensor-core instructions.
//
// This version stages tiles through shared memory so each byte fetched from
// global is reused across the whole block, and gives each warp a 32x32 output
// tile (four accumulator fragments) so one staged tile feeds more math.
//
// Block covers 64x64 of C. 4 warps in a 2x2 grid, each owning 32x32.
constexpr int kBM = 64, kBN = 64, kBK = 32;

__global__ void k_wmma_smem(int M, int N, int K,
                            const int8_t* __restrict__ A,
                            const int8_t* __restrict__ B,
                            int32_t* __restrict__ C) {
  __shared__ int8_t As[kBM][kBK];
  __shared__ int8_t Bs[kBK][kBN];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int warpRow = (warp / 2) * 32;   // 0 or 32, within the 64-row block tile
  const int warpCol = (warp % 2) * 32;

  const int blockRow = blockIdx.y * kBM;
  const int blockCol = blockIdx.x * kBN;

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int32_t> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) nvcuda::wmma::fill_fragment(acc[i][j], 0);

  // 128 threads move 2048 bytes per stage, 16 bytes each, as one int4.
  const int aRow = tid / 2, aCol = (tid % 2) * 16;   // into As[64][32]
  const int bRow = tid / 4, bCol = (tid % 4) * 16;   // into Bs[32][64]

  for (int k0 = 0; k0 < K; k0 += kBK) {
    *reinterpret_cast<int4*>(&As[aRow][aCol]) =
        *reinterpret_cast<const int4*>(&A[(blockRow + aRow) * K + k0 + aCol]);
    *reinterpret_cast<int4*>(&Bs[bRow][bCol]) =
        *reinterpret_cast<const int4*>(&B[(k0 + bRow) * N + blockCol + bCol]);
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, int8_t,
                             nvcuda::wmma::row_major> a[2];
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, int8_t,
                             nvcuda::wmma::row_major> b[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        nvcuda::wmma::load_matrix_sync(a[i], &As[warpRow + i * 16][kk], kBK);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        nvcuda::wmma::load_matrix_sync(b[j], &Bs[kk][warpCol + j * 16], kBN);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) nvcuda::wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      nvcuda::wmma::store_matrix_sync(
          &C[(blockRow + warpRow + i * 16) * N + blockCol + warpCol + j * 16],
          acc[i][j], N, nvcuda::wmma::mem_row_major);
    }
}

}  // namespace

const char* i8_gemm_impl_name(I8GemmImpl impl) {
  switch (impl) {
    case I8GemmImpl::Naive: return "naive";
    case I8GemmImpl::Tiled: return "tiled";
    case I8GemmImpl::TiledDp4a: return "tiled+dp4a";
    case I8GemmImpl::Wmma: return "wmma/IMMA";
    case I8GemmImpl::WmmaSmem: return "wmma+smem";
    case I8GemmImpl::Cublas: return "cuBLAS";
  }
  return "?";
}

bool gemm_i8(int M, int N, int K,
             const int8_t* dA, const int8_t* dB, int32_t* dC,
             I8GemmImpl impl) {
  if (impl == I8GemmImpl::TiledDp4a && (K % 4) != 0) return false;
  if (impl == I8GemmImpl::Wmma &&
      ((M % 16) != 0 || (N % 16) != 0 || (K % 16) != 0)) return false;

  if (impl == I8GemmImpl::WmmaSmem &&
      ((M % kBM) != 0 || (N % kBN) != 0 || (K % kBK) != 0)) return false;

  if (impl == I8GemmImpl::WmmaSmem) {
    const dim3 block(128);  // 4 warps, each owning 32x32 of the 64x64 block tile
    const dim3 grid(N / kBN, M / kBM);
    k_wmma_smem<<<grid, block>>>(M, N, K, dA, dB, dC);
    check(cudaGetLastError(), "wmma_smem launch");
    return true;
  }

  if (impl == I8GemmImpl::Wmma) {
    const dim3 block(128);  // 4 warps
    const dim3 grid((N + 31) / 32, (M + 31) / 32);
    k_wmma<<<grid, block>>>(M, N, K, dA, dB, dC);
    check(cudaGetLastError(), "wmma launch");
    return true;
  }

  if (impl == I8GemmImpl::Cublas) {
    // cuBLAS's int8 path requires every leading dimension to be a multiple of
    // 4, because the IMMA instructions underneath it load four bytes at a time.
    // Our hand-written kernels have no such restriction, which is worth stating
    // plainly: part of "beating cuBLAS" on an odd shape is that cuBLAS simply
    // declines to run it.
    if ((N % 4) != 0 || (K % 4) != 0) return false;

    // cuBLAS is column-major. Computing C^T = B^T * A^T in its layout gives us
    // the row-major C we want, with no explicit transposes.
    const int32_t alpha = 1, beta = 0;
    const cublasStatus_t st = cublasGemmEx(cublas(), CUBLAS_OP_N, CUBLAS_OP_N,
                                           N, M, K,
                                           &alpha,
                                           dB, CUDA_R_8I, N,
                                           dA, CUDA_R_8I, K,
                                           &beta,
                                           dC, CUDA_R_32I, N,
                                           CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_NOT_SUPPORTED) return false;  // shape it will not take
    check(st, "cublasGemmEx");
    return true;
  }

  const dim3 block(kTile, kTile);
  const dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  switch (impl) {
    case I8GemmImpl::Naive:     k_naive<<<grid, block>>>(M, N, K, dA, dB, dC); break;
    case I8GemmImpl::Tiled:     k_tiled<<<grid, block>>>(M, N, K, dA, dB, dC); break;
    case I8GemmImpl::TiledDp4a: k_tiled_dp4a<<<grid, block>>>(M, N, K, dA, dB, dC); break;
    default: return false;
  }
  check(cudaGetLastError(), "kernel launch");
  return true;
}

double time_gemm_i8(int M, int N, int K,
                    const int8_t* dA, const int8_t* dB, int32_t* dC,
                    I8GemmImpl impl, int iters) {
  if (!gemm_i8(M, N, K, dA, dB, dC, impl)) return -1.0;  // warmup
  check(cudaDeviceSynchronize(), "warmup sync");

  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "eventCreate");
  check(cudaEventCreate(&stop), "eventCreate");
  check(cudaEventRecord(start), "eventRecord");
  for (int i = 0; i < iters; ++i) gemm_i8(M, N, K, dA, dB, dC, impl);
  check(cudaEventRecord(stop), "eventRecord");
  check(cudaEventSynchronize(stop), "eventSync");

  float ms = 0.f;
  check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return static_cast<double>(ms) / iters;
}

void* device_alloc(size_t bytes) {
  void* p = nullptr;
  check(cudaMalloc(&p, bytes), "cudaMalloc");
  return p;
}
void device_free(void* p) { cudaFree(p); }
void upload(void* dst, const void* src, size_t bytes) {
  check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "H2D");
}
void download(void* dst, const void* src, size_t bytes) {
  check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "D2H");
}
void device_sync() { check(cudaDeviceSynchronize(), "sync"); }

DeviceInfo device_info() {
  cudaDeviceProp p{};
  check(cudaGetDeviceProperties(&p, 0), "getDeviceProperties");
  DeviceInfo d{};
  std::snprintf(d.name, sizeof(d.name), "%s", p.name);
  d.major = p.major;
  d.minor = p.minor;
  d.sms = p.multiProcessorCount;
  d.clock_khz = p.clockRate;
  // effective bytes/s = bus width (bits) / 8 * memory clock (kHz) * 1000 * 2 (DDR)
  d.mem_bandwidth_gbps =
      2.0 * p.memoryBusWidth / 8.0 * p.memoryClockRate * 1e3 / 1e9;
  return d;
}

}  // namespace ni::cuda

#endif  // NI_WITH_CUDA

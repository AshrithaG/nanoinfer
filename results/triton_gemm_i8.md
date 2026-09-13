# int8 GEMM: hand-written CUDA, Triton and cuBLAS on NVIDIA GeForce RTX 4090 (sm_89)

torch 2.11.0+cu130 (CUDA 13.0), Triton 3.6.0, driver 580.178.04. Hand-written kernels: loaded.

C[n,n] int32 = A[n,n] int8 x B[n,n] int8. Milliseconds per GEMM, median of 5 timing windows. Every number passed an exact CPU check at its own size first. Unless a column says eager, times are CUDA-graph replay, which leaves host launch cost out.

## Ladder, B row-major (NN)

| n | wmma+smem | same tiling as wmma+smem | + pipelining, 3 stages | + 128x128 tile, 8 warps | + grouped tile order | Triton best | cuBLAS | Triton best vs cuBLAS |
|---|---|---|---|---|---|---|---|---|
| 256 | 0.005 | 0.004 | 0.004 | 0.006 | 0.006 | 0.003 (64x64x64 w4 s4 g8) | 0.008 | 2.99x |
| 512 | 0.008 | 0.007 | 0.006 | 0.009 | 0.009 | 0.004 (64x64x64 w4 s4 g8) | 0.008 | 1.95x |
| 1024 | 0.021 | 0.016 | 0.014 | 0.016 | 0.016 | 0.009 (64x128x128 w4 s4 g8) | 0.024 | 2.63x |
| 2048 | 0.145 | 0.095 | 0.098 | 0.056 | 0.056 | 0.045 (128x128x64 w4 s3 g8) | 0.105 | 2.33x |
| 4096 | 1.125 | 0.699 | 0.689 | 0.414 | 0.416 | 0.333 (128x128x64 w4 s3 g8) | 0.858 | 2.57x |

## Layout: B row-major (NN) or K-contiguous (TN)

| n | cuBLAS NN | cuBLAS TN | torch._int_mm NN | torch._int_mm TN | Triton best NN | Triton best TN | fastest | TOPS |
|---|---|---|---|---|---|---|---|---|
| 256 | 0.008 | 0.002 | 0.003 | 0.003 | 0.003 | 0.002 | Triton best TN | 17.0 |
| 512 | 0.008 | 0.003 | 0.004 | 0.004 | 0.004 | 0.003 | Triton best TN | 95.0 |
| 1024 | 0.024 | 0.009 | 0.018 | 0.008 | 0.009 | 0.008 | Triton best TN | 273.8 |
| 2048 | 0.105 | 0.057 | 0.116 | 0.050 | 0.045 | 0.035 | Triton best TN | 497.9 |
| 4096 | 0.858 | 0.250 | 0.955 | 0.251 | 0.333 | 0.254 | cuBLAS TN | 550.1 |

## Back-to-back launches (ni_bench_cuda's method) against CUDA-graph replay

| n | wmma+smem eager | graph | cuBLAS eager | graph | torch._int_mm eager | graph | Triton: same tiling as wmma+smem eager | graph |
|---|---|---|---|---|---|---|---|---|
| 256 | 0.008 | 0.005 | 0.014 | 0.008 | 0.006 | 0.003 | 0.018 | 0.004 |
| 512 | 0.009 | 0.008 | 0.013 | 0.008 | 0.006 | 0.004 | 0.018 | 0.007 |
| 1024 | 0.022 | 0.021 | 0.026 | 0.024 | 0.019 | 0.018 | 0.019 | 0.016 |
| 2048 | 0.146 | 0.145 | 0.106 | 0.105 | 0.117 | 0.116 | 0.097 | 0.095 |
| 4096 | 1.123 | 1.125 | 0.804 | 0.858 | 0.936 | 0.955 | 0.720 | 0.699 |

## What each kernel compiled to

Static counts from cuobjdump -sass: which instructions exist, not how often they run. Triton kernels as compiled for B row-major. The hand-written kernels stage through plain __shared__ arrays, so their shared memory is unswizzled by construction.

| kernel | regs | local B | smem B | swizzled | mma (PTX) | IMMA | LDSM | LDGSTS | LDS | STS | LDG | BAR | PRMT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tiled+dp4a | 40 | 0 | 2048 | False | n/a | 0 | 0 | 0 | 12 | 6 | 6 | 6 | 6 |
| wmma | 40 | 0 | 0 | False | n/a | 10 | 0 | 0 | 0 | 0 | 50 | 0 | 30 |
| wmma+smem | 70 | 0 | 4096 | False | n/a | 48 | 12 | 0 | 96 | 6 | 6 | 6 | 72 |
| Triton: same tiling as wmma+smem | 71 | 0 | 8192 | True | m16n8k32 | 8 | 2 | 0 | 40 | 11 | 2 | 6 | 25 |
| Triton: + pipelining, 3 stages | 72 | 0 | 8192 | True | m16n8k32 | 8 | 2 | 9 | 46 | 8 | 0 | 7 | 24 |
| Triton: + 128x128 tile, 8 warps | 128 | 0 | 16384 | True | m16n8k32 | 16 | 4 | 9 | 54 | 16 | 0 | 11 | 24 |
| Triton: + grouped tile order | 128 | 0 | 16384 | True | m16n8k32 | 16 | 4 | 9 | 54 | 16 | 0 | 11 | 24 |
| Triton best at 4096, NN: 128x128x64 w4 s3 g8 | 230 | 0 | 32768 | True | m16n8k32 | 64 | 8 | 36 | 166 | 32 | 0 | 11 | 96 |
| Triton best at 4096, TN: 128x128x64 w4 s3 g8 | 216 | 0 | 32768 | True | m16n8k32 | 64 | 16 | 24 | 38 | 32 | 0 | 11 | 0 |

## Correctness

At M x N x K = 256x512x768: exact 43.


Other processes on the GPU during the run: none.

# int8 GEMM on RTX 4090 (sm_89), median ms per launch

C[n,n] int32 = A[n,n] int8 * B[n,n] int8. All implementations verified exact
against a CPU int32 reference on 61x76x132, 64x80x128 and 128x128x256 first.

| n | naive | tiled | tiled+dp4a | wmma | wmma+smem | cuBLAS | best TOPS | vs cuBLAS |
|---|---|---|---|---|---|---|---|---|
| 256 | 0.015 | 0.017 | 0.011 | 0.004 | 0.006 | 0.009 | 7.70 | 2.09x |
| 512 | 0.055 | 0.062 | 0.039 | 0.008 | 0.010 | 0.010 | 33.14 | 1.26x |
| 1024 | 0.423 | 0.485 | 0.303 | 0.044 | 0.024 | 0.029 | 90.86 | 1.24x |
| 2048 | 3.251 | 3.552 | 2.208 | 0.329 | 0.145 | 0.106 | 118.74 | 0.73x |
| 4096 | 24.746 | 30.464 | 18.344 | 5.142 | 1.119 | 0.797 | 122.81 | 0.71x |

Reproduce: `cmake -B build-cuda -DNI_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 && ./build-cuda/ni_bench_cuda`

## Kernel resources and theoretical occupancy

From cudaFuncGetAttributes and cudaOccupancyMaxActiveBlocksPerMultiprocessor.
No profiler permissions required.

| kernel | regs/thread | shared B | block | blocks/SM | occupancy |
|---|---|---|---|---|---|
| gemm naive | 38 | 0 | 1024 | 1 | 67% |
| gemm tiled | 37 | 2048 | 1024 | 1 | 67% |
| gemm tiled+dp4a | 37 | 2048 | 1024 | 1 | 67% |
| gemm wmma | 40 | 0 | 128 | 12 | 100% |
| gemm wmma+smem | 70 | 4096 | 128 | 7 | 58% |
| conv direct | 38 | 0 | 256 | 6 | 100% |
| conv direct+fused | 38 | 0 | 256 | 6 | 100% |
| conv smem+fused | 40 | 4608 | 256 | 6 | 100% |

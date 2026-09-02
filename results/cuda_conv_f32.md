# fp32 NCHW conv + bias + ReLU on RTX 4090 (sm_89), median ms of 50 launches

Batch 1. All implementations verified against the direct kernel's output.
cuDNN "fused f32" forces CUDNN_FMA_MATH; the other cuDNN rows use the default,
which promotes to TF32 on this hardware.

| layer | direct | +fused | +smem | cuDNN unfused | cuDNN fused | cuDNN fused f32 | mine vs cuDNN f32 |
|---|---|---|---|---|---|---|---|
| first 3x3 (3->32, 96x96) | 0.0158 | 0.0106 | 0.0096 | 0.0312 | 0.0305 | 0.0302 | 3.15x |
| mid 3x3 (64->64, 56x56) | 0.1283 | 0.1189 | 0.1062 | 0.0586 | 0.0635 | 0.0355 | 0.33x |
| pointwise (128->128, 28x28) | 0.0542 | 0.0490 | 0.0353 | 0.0421 | 0.0435 | 0.0431 | 1.22x |
| depthwise (128, 28x28) | 0.0072 | 0.0035 | declined | 0.0219 | 0.0215 | 0.0211 | 6.02x |

TF32 relative error against the fp32 reference: 7.76e-03 (mid 3x3),
4.58e-03 (pointwise unfused).

Reproduce: `cmake -B build-cuda -DNI_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DNI_CUDNN_ROOT=<cudnn> && ./build-cuda/ni_bench_conv`

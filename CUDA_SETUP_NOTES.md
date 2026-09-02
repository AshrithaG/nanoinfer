# CUDA environment on cerlab23 (audited 2026-09-01)

Local notes, not committed. Move B (CUDA backend) can run entirely on this box.
No rented GPU needed, with one caveat below.

## Hardware
RTX 4090, 24 GB, sm_89, 128 SMs. Driver 580.173.02. GPU was idle at audit time.
/home has 1.6 TB free.

## Toolchain: use 12.4, avoid 13.x

| path | version | status |
|---|---|---|
| `/usr/bin/nvcc` | 12.4.131 | **works.** Compiles and runs sm_89 kernels with system g++ 15.2 |
| `/usr/local/cuda-13.1/bin/nvcc` | 13.1.115 | **broken.** `rsqrt` exception-spec clash between glibc math headers and CUDA `crt/math_functions.h` |
| `/usr/local/cuda-13/bin/nvcc` | 13.1.115 | same failure |

`/usr/local/cuda` symlinks to the broken 13.1, so do not follow it in CMake.
Point explicitly at `/usr/bin/nvcc`.

The 12.4 smoke test only included `<cstdio>`. A C++20 codebase exercises far more
host headers, so if g++ 15.2 trips nvcc later, use `-allow-unsupported-compiler`
or `-ccbin` pointed at an older g++.

## Libraries

- **cuBLAS**: system-wide, `/usr/include/cublas_v2.h`. Ready.
- **cuDNN 9.25.1.1** (cu12, matches nvcc 12.4), installed into a user venv:
  - headers `~/cuda-env/lib/python3.14/site-packages/nvidia/cudnn/include`
  - libs    `~/cuda-env/lib/python3.14/site-packages/nvidia/cudnn/lib`
- **TensorRT 11.2.1.2**: Python bindings only, no `NvInfer.h`, and the wheel is
  the cu13 flavor. Benchmark TRT through Python rather than linking it from C++.

pip into system Python is blocked by PEP 668. Everything goes in `~/cuda-env`.

## The one blocker: Nsight profiling is denied

`ncu` and `nsys` are installed at `/usr/bin/`, but profiling fails with
`ERR_NVGPUCTRPERM`. GPU performance counters are admin-only on this machine.

Fix requires root and is a request to whoever owns cerlab23:
`/etc/modprobe.d/nvidia-profiler.conf` containing
`options nvidia NVreg_RestrictProfilingToAdminUsers=0`, then reboot or reload
the nvidia module. See https://developer.nvidia.com/ERR_NVGPUCTRPERM

Fallbacks if refused: compute achieved bandwidth and occupancy from timings and
launch geometry instead of counters, or rent a RunPod/Vast hour where profiling
permissions come with the box.

## What move B looks like here

Kernels in C++ vs cuBLAS and cuDNN, TensorRT compared via Python, all on the
same models and batch sweep. Correctness oracle already exists: 25 reference
checks and PyTorch parity at 5.2e-7.

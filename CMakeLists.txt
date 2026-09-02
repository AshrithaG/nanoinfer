cmake_minimum_required(VERSION 3.20)
project(nanoinfer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

add_library(nanoinfer src/tensor.cpp src/ops.cpp src/graph.cpp src/pool.cpp)
target_include_directories(nanoinfer PUBLIC include)

# Accelerate gives us a tuned sgemm on Apple platforms. Without it the blocked
# fallback in ops.cpp is used, which is the honest baseline for the comparison.
option(NI_WITH_ACCELERATE "Use Apple Accelerate for sgemm when available" ON)
if(APPLE AND NI_WITH_ACCELERATE)
  find_library(ACCELERATE Accelerate)
  if(ACCELERATE)
    target_link_libraries(nanoinfer PUBLIC ${ACCELERATE})
    # the ILP64-capable headers; without these the legacy cblas entry points
    # are deprecated and warn on every build
    target_compile_definitions(nanoinfer PUBLIC
      NI_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(nanoinfer PUBLIC Threads::Threads)

# ARM dot-product instructions (SDOT) turn 16 int8 multiply-accumulates into one
# instruction. Present on every Apple M-series and most recent ARM server parts,
# but not guaranteed, so it is probed rather than assumed.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-march=armv8.2-a+dotprod")
check_cxx_source_compiles("
#include <arm_neon.h>
int main() {
  int32x4_t a = vdupq_n_s32(0);
  int8x16_t b = vdupq_n_s8(1);
  return vaddvq_s32(vdotq_s32(a, b, b)) == 16 ? 0 : 1;
}" NI_HAVE_DOTPROD)
unset(CMAKE_REQUIRED_FLAGS)
if(NI_HAVE_DOTPROD)
  target_compile_options(nanoinfer PRIVATE -march=armv8.2-a+dotprod)
  target_compile_definitions(nanoinfer PRIVATE NI_USE_DOTPROD)
endif()

# ---------------------------------------------------------------- CUDA backend
# Off by default so the CPU build stays dependency-free. Enable with
#   cmake -B build-cuda -DNI_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
# On machines where /usr/local/cuda points at a toolkit whose headers clash with
# the system glibc, pass -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc explicitly.
option(NI_WITH_CUDA "Build the CUDA backend and its benchmark" OFF)
if(NI_WITH_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)  # Ada, RTX 4090
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  add_library(nanoinfer_cuda src/cuda/gemm_i8.cu src/cuda/conv_fused.cu)
  target_include_directories(nanoinfer_cuda PUBLIC include)
  target_compile_definitions(nanoinfer_cuda PUBLIC NI_WITH_CUDA)
  set_target_properties(nanoinfer_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(nanoinfer_cuda PUBLIC CUDA::cublas CUDA::cudart)

  # cuDNN is not part of the CUDA toolkit. Point NI_CUDNN_ROOT at a directory
  # containing include/cudnn.h and lib/libcudnn.so; a pip install of
  # nvidia-cudnn-cu12 inside a venv works and needs no root.
  set(NI_CUDNN_ROOT "" CACHE PATH "Root of a cuDNN install (include/ and lib/)")
  if(NI_CUDNN_ROOT AND EXISTS "${NI_CUDNN_ROOT}/include/cudnn.h")
    # pip's cuDNN wheels ship libcudnn.so.9 with no unversioned symlink, which
    # plain -lcudnn cannot find, so locate the actual file.
    find_library(NI_CUDNN_LIB NAMES cudnn PATHS "${NI_CUDNN_ROOT}/lib" NO_DEFAULT_PATH)
    if(NOT NI_CUDNN_LIB)
      file(GLOB _ni_cudnn_so "${NI_CUDNN_ROOT}/lib/libcudnn.so*")
      if(_ni_cudnn_so)
        list(GET _ni_cudnn_so 0 NI_CUDNN_LIB)
      endif()
    endif()
  endif()
  if(NI_CUDNN_LIB)
    message(STATUS "cuDNN: ${NI_CUDNN_LIB}")
    target_include_directories(nanoinfer_cuda PUBLIC "${NI_CUDNN_ROOT}/include")
    target_link_libraries(nanoinfer_cuda PUBLIC "${NI_CUDNN_LIB}")
    target_compile_definitions(nanoinfer_cuda PUBLIC NI_WITH_CUDNN)
    set_target_properties(nanoinfer_cuda PROPERTIES
      BUILD_RPATH "${NI_CUDNN_ROOT}/lib" INSTALL_RPATH "${NI_CUDNN_ROOT}/lib")
  else()
    message(STATUS "cuDNN: not found, conv benchmark will skip the vendor rows")
  endif()

  add_executable(ni_bench_cuda bench/bench_cuda.cu)
  target_link_libraries(ni_bench_cuda PRIVATE nanoinfer_cuda)

  add_executable(ni_bench_conv bench/bench_conv.cu)
  target_link_libraries(ni_bench_conv PRIVATE nanoinfer_cuda)
  if(NI_CUDNN_LIB)
    set_target_properties(ni_bench_conv PROPERTIES BUILD_RPATH "${NI_CUDNN_ROOT}/lib")
  endif()
endif()

add_executable(ni_test tests/test_main.cpp)
target_link_libraries(ni_test PRIVATE nanoinfer)

add_executable(ni_bench bench/bench.cpp)
target_link_libraries(ni_bench PRIVATE nanoinfer)

add_executable(ni_run tools/run.cpp)
target_link_libraries(ni_run PRIVATE nanoinfer)

enable_testing()
add_test(NAME unit COMMAND ni_test)

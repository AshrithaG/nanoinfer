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

add_executable(ni_test tests/test_main.cpp)
target_link_libraries(ni_test PRIVATE nanoinfer)

add_executable(ni_bench bench/bench.cpp)
target_link_libraries(ni_bench PRIVATE nanoinfer)

add_executable(ni_run tools/run.cpp)
target_link_libraries(ni_run PRIVATE nanoinfer)

enable_testing()
add_test(NAME unit COMMAND ni_test)

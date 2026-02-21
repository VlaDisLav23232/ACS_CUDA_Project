cmake_minimum_required(VERSION 3.18)
project(heat_stencil LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# prevent nvcc from optimizing away Kahan compensation
# -fmad=false stops fused multiply-add which can break error tracking
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(heat_stencil
    src/main.cpp
    src/heat2d_cpu.cpp
    src/heat2d_cuda.cu
    src/heat2d_cuda_fp16.cu
    src/heat3d_cpu.cpp
    src/heat3d_cuda.cu
    src/heat3d_cuda_fp16.cu
    src/metrics.cpp
)

target_include_directories(heat_stencil PRIVATE include)

# results directory created at config time
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results)

cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 86)

project(wave_stencil LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(wave_stencil
	src/main.cpp
	src/wave2d_cpu.cpp
	src/wave2d_cuda.cu
	src/wave2d_cuda_fp16.cu
	src/wave3d_cpu.cpp
	src/wave3d_cuda.cu
	src/wave3d_cuda_fp16.cu
	src/metrics.cpp
)

target_include_directories(wave_stencil PRIVATE include)

add_executable(visualize_vtk
	src/visualize_3d_vtk.cu
)

target_include_directories(visualize_vtk PRIVATE include)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

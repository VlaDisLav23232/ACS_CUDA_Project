#include "stencil.h"
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// block size for 2D kernel - 16x16 = 256 threads is a good default for stencils
// fits nicely in SM resources on CC 7.5 (max 1024 threads/block, 48KB shared mem)
static const int BLOCK_X = 16;
static const int BLOCK_Y = 16;

__global__ void heat2d_fp32_kernel(const float* __restrict__ u,
                                    float* __restrict__ u_next,
                                    int N, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
        float center = u[j * N + i];
        float left   = u[j * N + (i - 1)];
        float right  = u[j * N + (i + 1)];
        float up     = u[(j - 1) * N + i];
        float down   = u[(j + 1) * N + i];
        u_next[j * N + i] = center + r * (left + right + up + down - 4.0f * center);
    }
}

__global__ void apply_neumann_bc(float* u, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        u[0 * N + idx] = u[1 * N + idx];
        u[(N-1) * N + idx] = u[(N-2) * N + idx];
        u[idx * N + 0] = u[idx * N + 1];
        u[idx * N + (N-1)] = u[idx * N + (N-2)];
    }
}

StencilResult run_cuda_fp32(const StencilConfig& cfg) {
    int N = cfg.nx;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t grid_bytes = N * N * sizeof(float);

    // prepare initial condition on host (same as CPU)
    std::vector<float> h_u(N * N, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++)
        for (int i = src_start; i < src_start + src_size; i++)
            h_u[j * N + i] = cfg.temp_source;

    float *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, grid_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);
    int bc_threads = 256;
    int bc_blocks = (N + bc_threads - 1) / bc_threads;

    // timing with CUDA events (more accurate than wall clock for GPU work)
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat2d_fp32_kernel<<<grid, block>>>(d_u, d_u_next, N, r);
        apply_neumann_bc<<<bc_blocks, bc_threads>>>(d_u_next, N);
        // swap pointers instead of copying
        float* tmp = d_u;
        d_u = d_u_next;
        d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // copy result back
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, grid_bytes, cudaMemcpyDeviceToHost));

    // each timestep reads 5 floats and writes 1 float per interior point
    // (center + 4 neighbors read, 1 write) = 6 * 4 bytes per point
    double bytes_per_step = (double)(N - 2) * (N - 2) * 6 * sizeof(float);
    double total_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp32";
    res.grid_size = N;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes = 2 * grid_bytes;
    res.final_grid = h_u;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return res;
}

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

static const int BX = 8;
static const int BY = 8;
static const int BZ = 4;

__constant__ float d_coeffs_3d[MAX_REACH + 1];
__constant__ int   d_reach_3d;

__global__ void heat3d_fp32_kernel(const float* __restrict__ u,
                                    float* __restrict__ u_next,
                                    int N, float r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_reach_3d;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = (size_t)z * N * N + y * N + x;
        float center = u[idx];
        float lap = 3.0f * d_coeffs_3d[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_3d[m] * (u[idx - m] + u[idx + m]
                                   + u[idx - (size_t)m * N] + u[idx + (size_t)m * N]
                                   + u[idx - (size_t)m * N * N] + u[idx + (size_t)m * N * N]);
        }
        u_next[idx] = center + r * lap;
    }
}

__global__ void apply_neumann_bc_3d(float* u, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;

    for (int b = R - 1; b >= 0; b--) {
        // x faces
        u[(size_t)a * N * N + b_idx * N + b]       = u[(size_t)a * N * N + b_idx * N + (b + 1)];
        u[(size_t)a * N * N + b_idx * N + (N-1-b)] = u[(size_t)a * N * N + b_idx * N + (N-2-b)];
        // y faces
        u[(size_t)a * N * N + b * N + b_idx]       = u[(size_t)a * N * N + (b + 1) * N + b_idx];
        u[(size_t)a * N * N + (N-1-b) * N + b_idx] = u[(size_t)a * N * N + (N-2-b) * N + b_idx];
        // z faces
        u[(size_t)b * N * N + a * N + b_idx]       = u[(size_t)(b + 1) * N * N + a * N + b_idx];
        u[(size_t)(N-1-b) * N * N + a * N + b_idx] = u[(size_t)(N-2-b) * N * N + a * N + b_idx];
    }
}

// Kernel to enforce constant temperature heat source at center
__global__ void apply_heat_source_3d(float* u, int N, int src_start, int src_size, float temp_source) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= src_start && x < src_start + src_size &&
        y >= src_start && y < src_start + src_size &&
        z >= src_start && z < src_start + src_size) {
        size_t idx = (size_t)z * N * N + y * N + x;
        u[idx] = temp_source;
    }
}

StencilResult run_cuda_fp32_3d(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total = (size_t)N * N * N;
    size_t grid_bytes = total * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_3d, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_3d, &R, sizeof(int)));

    std::vector<float> h_u(total, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int z = src_start; z < src_start + src_size; z++)
        for (int y = src_start; y < src_start + src_size; y++)
            for (int x = src_start; x < src_start + src_size; x++)
                h_u[(size_t)z * N * N + y * N + x] = cfg.temp_source;

    float *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, grid_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));

    dim3 block(BX, BY, BZ);
    dim3 grid3((N + BX - 1) / BX, (N + BY - 1) / BY, (N + BZ - 1) / BZ);
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);
    
    // Heat source grid - reuse src_size and src_start from initialization
    dim3 src_block(8, 8, 8);
    dim3 src_grid((src_size + 7) / 8, (src_size + 7) / 8, (src_size + 7) / 8);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat3d_fp32_kernel<<<grid3, block>>>(d_u, d_u_next, N, r);
        apply_neumann_bc_3d<<<bc_grid, bc_block>>>(d_u_next, N, R);
        // IMPORTANT: Reapply constant heat source after each timestep
        apply_heat_source_3d<<<src_grid, src_block>>>(d_u_next, N, src_start, src_size, cfg.temp_source);
        float* tmp = d_u; d_u = d_u_next; d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, grid_bytes, cudaMemcpyDeviceToHost));

    // minimum DRAM bandwidth: each point read once, written once
    double bytes_per_step = 2.0 * (double)N * N * N * sizeof(float);
    double total_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp32_3d";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
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

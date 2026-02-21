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

static const int BLOCK_X = 16;
static const int BLOCK_Y = 16;

__constant__ float d_coeffs_fp32[MAX_REACH + 1];
__constant__ int   d_reach_fp32;

__global__ void heat2d_fp32_kernel(const float* __restrict__ u,
                                    float* __restrict__ u_next,
                                    int N, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int R = d_reach_fp32;

    if (i >= R && i < N - R && j >= R && j < N - R) {
        float center = u[j * N + i];
        float lap = 2.0f * d_coeffs_fp32[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_fp32[m] * (u[j * N + (i - m)] + u[j * N + (i + m)]
                                     + u[(j - m) * N + i] + u[(j + m) * N + i]);
        }
        u_next[j * N + i] = center + r * lap;
    }
}

__global__ void apply_neumann_bc(float* u, int N, int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int b = R - 1; b >= 0; b--) {
            u[b * N + idx]       = u[(b + 1) * N + idx];
            u[(N-1-b) * N + idx] = u[(N-2-b) * N + idx];
            u[idx * N + b]       = u[idx * N + (b + 1)];
            u[idx * N + (N-1-b)] = u[idx * N + (N-2-b)];
        }
    }
}

StencilResult run_cuda_fp32(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t grid_bytes = N * N * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_fp32, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_fp32, &R, sizeof(int)));

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

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat2d_fp32_kernel<<<grid, block>>>(d_u, d_u_next, N, r);
        apply_neumann_bc<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        float* tmp = d_u;
        d_u = d_u_next;
        d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, grid_bytes, cudaMemcpyDeviceToHost));

    int interior = N - 2 * R;
    double reads_per_point = (2 * 2 * R + 1);
    double bytes_per_step = (double)interior * interior * (reads_per_point + 1) * sizeof(float);
    double total_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp32";
    res.grid_size = N;
    res.dim = cfg.dim;
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

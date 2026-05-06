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
__constant__ int d_reach_fp32;

__global__ void first_step_2d_kernel(const float* __restrict__ u0,
                                     float* __restrict__ u1,
                                     int N, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int R = d_reach_fp32;

    if (i >= R && i < N - R && j >= R && j < N - R) {
        int idx = j * N + i;
        float center = u0[idx];
        float lap = 2.0f * d_coeffs_fp32[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_fp32[m] * (u0[j * N + (i - m)] + u0[j * N + (i + m)]
                                     + u0[(j - m) * N + i] + u0[(j + m) * N + i]);
        }
        u1[idx] = center + 0.5f * s * lap;
    }
}

__global__ void wave2d_fp32_kernel(const float* __restrict__ u_prev,
                                   const float* __restrict__ u,
                                   float* __restrict__ u_next,
                                   int N, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int R = d_reach_fp32;

    if (i >= R && i < N - R && j >= R && j < N - R) {
        int idx = j * N + i;
        float center = u[idx];
        float lap = 2.0f * d_coeffs_fp32[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_fp32[m] * (u[j * N + (i - m)] + u[j * N + (i + m)]
                                     + u[(j - m) * N + i] + u[(j + m) * N + i]);
        }
        u_next[idx] = 2.0f * center - u_prev[idx] + s * lap;
    }
}

__global__ void apply_dirichlet_2d(float* u, int N, int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int b = 0; b < R; b++) {
            u[b * N + idx] = 0.0f;
            u[(N - 1 - b) * N + idx] = 0.0f;
            u[idx * N + b] = 0.0f;
            u[idx * N + (N - 1 - b)] = 0.0f;
        }
    }
}

static void init_wave_field_2d(std::vector<float>& u, int N, float amp, float sigma) {
    const float s2 = sigma * sigma;
    const float src[2][2] = {{-0.35f, -0.35f}, {0.35f, 0.35f}};
    for (int j = 0; j < N; j++) {
        float y = -1.0f + 2.0f * j / (N - 1.0f);
        for (int i = 0; i < N; i++) {
            float x = -1.0f + 2.0f * i / (N - 1.0f);
            float v = 0.0f;
            for (int k = 0; k < 2; k++) {
                float dx = x - src[k][0];
                float dy = y - src[k][1];
                v += amp * expf(-(dx * dx + dy * dy) / s2);
            }
            u[j * N + i] = v;
        }
    }
}

StencilResult run_cuda_fp32(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float s = (cfg.c * cfg.dt / cfg.dx) * (cfg.c * cfg.dt / cfg.dx);
    size_t elems = static_cast<size_t>(N) * N;
    size_t bytes = elems * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_fp32, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_fp32, &R, sizeof(int)));

    std::vector<float> h_u0(elems, cfg.disp_initial);
    init_wave_field_2d(h_u0, N, cfg.source_amplitude, cfg.source_sigma);

    float *d_u_prev, *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, bytes));

    CUDA_CHECK(cudaMemcpy(d_u_prev, h_u0.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u, 0, bytes));
    CUDA_CHECK(cudaMemset(d_u_next, 0, bytes));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);
    int bc_threads = 256;
    int bc_blocks = (N + bc_threads - 1) / bc_threads;

    first_step_2d_kernel<<<grid, block>>>(d_u_prev, d_u, N, s);
    apply_dirichlet_2d<<<bc_blocks, bc_threads>>>(d_u_prev, N, R);
    apply_dirichlet_2d<<<bc_blocks, bc_threads>>>(d_u, N, R);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 1; t < cfg.timesteps; t++) {
        wave2d_fp32_kernel<<<grid, block>>>(d_u_prev, d_u, d_u_next, N, s);
        apply_dirichlet_2d<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        float* tmp = d_u_prev;
        d_u_prev = d_u;
        d_u = d_u_next;
        d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::vector<float> h_u(elems);
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, bytes, cudaMemcpyDeviceToHost));

    int interior = N - 2 * R;
    double reads = (2.0 * 2.0 * R + 2.0);
    double writes = 1.0;
    double bytes_per_step = static_cast<double>(interior) * interior * (reads + writes) * sizeof(float);
    double total_bytes = bytes_per_step * (cfg.timesteps - 1);
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp32";
    res.grid_size = N;
    res.dim = cfg.dim;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes = 3 * bytes;
    res.final_grid = h_u;

    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

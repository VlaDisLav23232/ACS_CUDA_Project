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
__constant__ int d_reach_3d;

__global__ void first_step_3d_kernel(const float* __restrict__ u0,
                                     float* __restrict__ u1,
                                     int N, float s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_reach_3d;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = static_cast<size_t>(z) * N * N + y * N + x;
        float center = u0[idx];
        float lap = 3.0f * d_coeffs_3d[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_3d[m] * (u0[idx - m] + u0[idx + m]
                                   + u0[idx - static_cast<size_t>(m) * N] + u0[idx + static_cast<size_t>(m) * N]
                                   + u0[idx - static_cast<size_t>(m) * N * N] + u0[idx + static_cast<size_t>(m) * N * N]);
        }
        u1[idx] = center + 0.5f * s * lap;
    }
}

__global__ void wave3d_fp32_kernel(const float* __restrict__ u_prev,
                                   const float* __restrict__ u,
                                   float* __restrict__ u_next,
                                   int N, float s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_reach_3d;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = static_cast<size_t>(z) * N * N + y * N + x;
        float center = u[idx];
        float lap = 3.0f * d_coeffs_3d[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_3d[m] * (u[idx - m] + u[idx + m]
                                   + u[idx - static_cast<size_t>(m) * N] + u[idx + static_cast<size_t>(m) * N]
                                   + u[idx - static_cast<size_t>(m) * N * N] + u[idx + static_cast<size_t>(m) * N * N]);
        }
        u_next[idx] = 2.0f * center - u_prev[idx] + s * lap;
    }
}

__global__ void apply_dirichlet_3d(float* u, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b >= N) return;

    for (int r = 0; r < R; r++) {
        u[static_cast<size_t>(a) * N * N + b * N + r] = 0.0f;
        u[static_cast<size_t>(a) * N * N + b * N + (N - 1 - r)] = 0.0f;
        u[static_cast<size_t>(a) * N * N + r * N + b] = 0.0f;
        u[static_cast<size_t>(a) * N * N + (N - 1 - r) * N + b] = 0.0f;
        u[static_cast<size_t>(r) * N * N + a * N + b] = 0.0f;
        u[static_cast<size_t>(N - 1 - r) * N * N + a * N + b] = 0.0f;
    }
}

static void init_wave_field_3d(std::vector<float>& u, int N, float amp, float sigma) {
    const float s2 = sigma * sigma;
    const float src[2][3] = {{-0.30f, -0.30f, -0.30f}, {0.30f, 0.30f, 0.30f}};
    for (int z = 0; z < N; z++) {
        float zz = -1.0f + 2.0f * z / (N - 1.0f);
        for (int y = 0; y < N; y++) {
            float yy = -1.0f + 2.0f * y / (N - 1.0f);
            for (int x = 0; x < N; x++) {
                float xx = -1.0f + 2.0f * x / (N - 1.0f);
                float v = 0.0f;
                for (int k = 0; k < 2; k++) {
                    float dx = xx - src[k][0];
                    float dy = yy - src[k][1];
                    float dz = zz - src[k][2];
                    v += amp * expf(-(dx * dx + dy * dy + dz * dz) / s2);
                }
                u[static_cast<size_t>(z) * N * N + y * N + x] = v;
            }
        }
    }
}

StencilResult run_cuda_fp32_3d(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float s = (cfg.c * cfg.dt / cfg.dx) * (cfg.c * cfg.dt / cfg.dx);
    size_t total = static_cast<size_t>(N) * N * N;
    size_t bytes = total * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_3d, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_3d, &R, sizeof(int)));

    std::vector<float> h_u0(total, cfg.disp_initial);
    init_wave_field_3d(h_u0, N, cfg.source_amplitude, cfg.source_sigma);

    float *d_u_prev, *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, bytes));
    CUDA_CHECK(cudaMemcpy(d_u_prev, h_u0.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u, 0, bytes));
    CUDA_CHECK(cudaMemset(d_u_next, 0, bytes));

    dim3 block(BX, BY, BZ);
    dim3 grid((N + BX - 1) / BX, (N + BY - 1) / BY, (N + BZ - 1) / BZ);
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    first_step_3d_kernel<<<grid, block>>>(d_u_prev, d_u, N, s);
    apply_dirichlet_3d<<<bc_grid, bc_block>>>(d_u_prev, N, R);
    apply_dirichlet_3d<<<bc_grid, bc_block>>>(d_u, N, R);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 1; t < cfg.timesteps; t++) {
        wave3d_fp32_kernel<<<grid, block>>>(d_u_prev, d_u, d_u_next, N, s);
        apply_dirichlet_3d<<<bc_grid, bc_block>>>(d_u_next, N, R);
        float* tmp = d_u_prev;
        d_u_prev = d_u;
        d_u = d_u_next;
        d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::vector<float> h_u(total);
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, bytes, cudaMemcpyDeviceToHost));

    int interior = N - 2 * R;
    double reads = (2.0 * 3.0 * R + 2.0);
    double writes = 1.0;
    double bytes_per_step = static_cast<double>(interior) * interior * interior * (reads + writes) * sizeof(float);
    double bw = bytes_per_step * (cfg.timesteps - 1) / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp32_3d";
    res.grid_size = N;
    res.dim = 3;
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

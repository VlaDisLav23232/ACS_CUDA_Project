#include "stencil.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static const int BLOCK_X = 16;
static const int BLOCK_Y = 16;

__constant__ float d_coeffs_fp16[MAX_REACH + 1];
__constant__ int d_reach_fp16;

__global__ void first_step_fp16_naive(const __half* __restrict__ u0,
                                      __half* __restrict__ u1,
                                      int N, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int R = d_reach_fp16;

    if (i >= R && i < N - R && j >= R && j < N - R) {
        int idx = j * N + i;
        float center = __half2float(u0[idx]);
        float lap = 2.0f * d_coeffs_fp16[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_fp16[m] * (__half2float(u0[j * N + (i - m)]) + __half2float(u0[j * N + (i + m)])
                                     + __half2float(u0[(j - m) * N + i]) + __half2float(u0[(j + m) * N + i]));
        }
        u1[idx] = __float2half(center + 0.5f * s * lap);
    }
}

__global__ void wave2d_fp16_naive_kernel(const __half* __restrict__ u_prev,
                                         const __half* __restrict__ u,
                                         __half* __restrict__ u_next,
                                         int N, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int R = d_reach_fp16;

    if (i >= R && i < N - R && j >= R && j < N - R) {
        int idx = j * N + i;
        float center = __half2float(u[idx]);
        float lap = 2.0f * d_coeffs_fp16[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_fp16[m] * (__half2float(u[j * N + (i - m)]) + __half2float(u[j * N + (i + m)])
                                     + __half2float(u[(j - m) * N + i]) + __half2float(u[(j + m) * N + i]));
        }
        float next = 2.0f * center - __half2float(u_prev[idx]) + s * lap;
        u_next[idx] = __float2half(next);
    }
}

__global__ void first_step_fp16_kahan(const __half* __restrict__ u0,
                                      __half* __restrict__ u1,
                                      float* __restrict__ c1,
                                      int N, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int R = d_reach_fp16;

    if (i >= R && i < N - R && j >= R && j < N - R) {
        int idx = j * N + i;
        float center = __half2float(u0[idx]);
        float lap = 2.0f * d_coeffs_fp16[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_fp16[m] * (__half2float(u0[j * N + (i - m)]) + __half2float(u0[j * N + (i + m)])
                                     + __half2float(u0[(j - m) * N + i]) + __half2float(u0[(j + m) * N + i]));
        }
        float exact = center + 0.5f * s * lap;
        __half stored = __float2half(exact);
        u1[idx] = stored;
        c1[idx] = exact - __half2float(stored);
    }
}

__global__ void wave2d_fp16_kahan_kernel(const __half* __restrict__ u_prev,
                                         const __half* __restrict__ u,
                                         __half* __restrict__ u_next,
                                         const float* __restrict__ c_prev,
                                         const float* __restrict__ c,
                                         float* __restrict__ c_next,
                                         int N, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int R = d_reach_fp16;

    if (i >= R && i < N - R && j >= R && j < N - R) {
        int idx = j * N + i;
        float center = __half2float(u[idx]) + c[idx];
        float lap = 2.0f * d_coeffs_fp16[0] * center;
        for (int m = 1; m <= R; m++) {
            float xm = __half2float(u[j * N + (i - m)]) + c[j * N + (i - m)];
            float xp = __half2float(u[j * N + (i + m)]) + c[j * N + (i + m)];
            float ym = __half2float(u[(j - m) * N + i]) + c[(j - m) * N + i];
            float yp = __half2float(u[(j + m) * N + i]) + c[(j + m) * N + i];
            lap += d_coeffs_fp16[m] * (xm + xp + ym + yp);
        }
        float prev = __half2float(u_prev[idx]) + c_prev[idx];
        float exact = 2.0f * center - prev + s * lap;
        __half stored = __float2half(exact);
        u_next[idx] = stored;
        c_next[idx] = exact - __half2float(stored);
    }
}

__global__ void apply_dirichlet_fp16(__half* u, int N, int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        __half z = __float2half(0.0f);
        for (int b = 0; b < R; b++) {
            u[b * N + idx] = z;
            u[(N - 1 - b) * N + idx] = z;
            u[idx * N + b] = z;
            u[idx * N + (N - 1 - b)] = z;
        }
    }
}

__global__ void apply_dirichlet_float(float* c, int N, int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int b = 0; b < R; b++) {
            c[b * N + idx] = 0.0f;
            c[(N - 1 - b) * N + idx] = 0.0f;
            c[idx * N + b] = 0.0f;
            c[idx * N + (N - 1 - b)] = 0.0f;
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

static std::vector<__half> float_to_half(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); i++) h[i] = __float2half(f[i]);
    return h;
}

StencilResult run_cuda_fp16_naive(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float s = (cfg.c * cfg.dt / cfg.dx) * (cfg.c * cfg.dt / cfg.dx);
    size_t elems = static_cast<size_t>(N) * N;
    size_t half_bytes = elems * sizeof(__half);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_fp16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_fp16, &R, sizeof(int)));

    std::vector<float> h_u0(elems, cfg.disp_initial);
    init_wave_field_2d(h_u0, N, cfg.source_amplitude, cfg.source_sigma);
    auto h_half = float_to_half(h_u0);

    __half *d_u_prev, *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u_prev, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMemcpy(d_u_prev, h_half.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u, 0, half_bytes));
    CUDA_CHECK(cudaMemset(d_u_next, 0, half_bytes));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);
    int bc_threads = 256;
    int bc_blocks = (N + bc_threads - 1) / bc_threads;

    first_step_fp16_naive<<<grid, block>>>(d_u_prev, d_u, N, s);
    apply_dirichlet_fp16<<<bc_blocks, bc_threads>>>(d_u_prev, N, R);
    apply_dirichlet_fp16<<<bc_blocks, bc_threads>>>(d_u, N, R);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 1; t < cfg.timesteps; t++) {
        wave2d_fp16_naive_kernel<<<grid, block>>>(d_u_prev, d_u, d_u_next, N, s);
        apply_dirichlet_fp16<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        __half* tmp = d_u_prev;
        d_u_prev = d_u;
        d_u = d_u_next;
        d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_half.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> result_f(elems);
    for (size_t i = 0; i < elems; i++) result_f[i] = __half2float(h_half[i]);

    int interior = N - 2 * R;
    double reads = (2.0 * 2.0 * R + 2.0);
    double writes = 1.0;
    double bytes_per_step = static_cast<double>(interior) * interior * (reads + writes) * sizeof(__half);
    double bw = bytes_per_step * (cfg.timesteps - 1) / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp16_naive";
    res.grid_size = N;
    res.dim = cfg.dim;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes = 3 * half_bytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

StencilResult run_cuda_fp16_kahan(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float s = (cfg.c * cfg.dt / cfg.dx) * (cfg.c * cfg.dt / cfg.dx);
    size_t elems = static_cast<size_t>(N) * N;
    size_t half_bytes = elems * sizeof(__half);
    size_t float_bytes = elems * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_fp16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_fp16, &R, sizeof(int)));

    std::vector<float> h_u0(elems, cfg.disp_initial);
    init_wave_field_2d(h_u0, N, cfg.source_amplitude, cfg.source_sigma);
    auto h_half = float_to_half(h_u0);
    std::vector<float> h_comp(elems, 0.0f);

    __half *d_u_prev, *d_u, *d_u_next;
    float *d_c_prev, *d_c, *d_c_next;
    CUDA_CHECK(cudaMalloc(&d_u_prev, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_prev, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_next, float_bytes));

    CUDA_CHECK(cudaMemcpy(d_u_prev, h_half.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u, 0, half_bytes));
    CUDA_CHECK(cudaMemset(d_u_next, 0, half_bytes));
    CUDA_CHECK(cudaMemset(d_c_prev, 0, float_bytes));
    CUDA_CHECK(cudaMemset(d_c, 0, float_bytes));
    CUDA_CHECK(cudaMemset(d_c_next, 0, float_bytes));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);
    int bc_threads = 256;
    int bc_blocks = (N + bc_threads - 1) / bc_threads;

    first_step_fp16_kahan<<<grid, block>>>(d_u_prev, d_u, d_c, N, s);
    apply_dirichlet_fp16<<<bc_blocks, bc_threads>>>(d_u_prev, N, R);
    apply_dirichlet_fp16<<<bc_blocks, bc_threads>>>(d_u, N, R);
    apply_dirichlet_float<<<bc_blocks, bc_threads>>>(d_c_prev, N, R);
    apply_dirichlet_float<<<bc_blocks, bc_threads>>>(d_c, N, R);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 1; t < cfg.timesteps; t++) {
        wave2d_fp16_kahan_kernel<<<grid, block>>>(d_u_prev, d_u, d_u_next, d_c_prev, d_c, d_c_next, N, s);
        apply_dirichlet_fp16<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        apply_dirichlet_float<<<bc_blocks, bc_threads>>>(d_c_next, N, R);

        __half* tmp_h = d_u_prev;
        d_u_prev = d_u;
        d_u = d_u_next;
        d_u_next = tmp_h;

        float* tmp_c = d_c_prev;
        d_c_prev = d_c;
        d_c = d_c_next;
        d_c_next = tmp_c;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_half.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(elems);
    for (size_t i = 0; i < elems; i++) result_f[i] = __half2float(h_half[i]) + h_comp[i];

    int interior = N - 2 * R;
    double reads = (2.0 * 2.0 * R + 2.0);
    double writes = 1.0;
    double half_rw = (reads + writes) * sizeof(__half);
    double comp_rw = (reads + writes) * sizeof(float);
    double bytes_per_step = static_cast<double>(interior) * interior * (half_rw + comp_rw);
    double bw = bytes_per_step * (cfg.timesteps - 1) / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp16_kahan";
    res.grid_size = N;
    res.dim = cfg.dim;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes = 3 * half_bytes + 3 * float_bytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_c_prev));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_c_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

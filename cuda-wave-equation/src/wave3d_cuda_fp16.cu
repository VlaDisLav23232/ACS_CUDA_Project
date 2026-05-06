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

static const int BX = 8;
static const int BY = 8;
static const int BZ = 4;

__constant__ float d_coeffs_3d16[MAX_REACH + 1];
__constant__ int d_reach_3d16;

__global__ void first_step_3d_fp16_naive(const __half* __restrict__ u0,
                                         __half* __restrict__ u1,
                                         int N, float s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_reach_3d16;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = static_cast<size_t>(z) * N * N + y * N + x;
        float center = __half2float(u0[idx]);
        float lap = 3.0f * d_coeffs_3d16[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_3d16[m] * (
                __half2float(u0[idx - m]) + __half2float(u0[idx + m])
              + __half2float(u0[idx - static_cast<size_t>(m) * N]) + __half2float(u0[idx + static_cast<size_t>(m) * N])
              + __half2float(u0[idx - static_cast<size_t>(m) * N * N]) + __half2float(u0[idx + static_cast<size_t>(m) * N * N]));
        }
        u1[idx] = __float2half(center + 0.5f * s * lap);
    }
}

__global__ void wave3d_fp16_naive_kernel(const __half* __restrict__ u_prev,
                                         const __half* __restrict__ u,
                                         __half* __restrict__ u_next,
                                         int N, float s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_reach_3d16;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = static_cast<size_t>(z) * N * N + y * N + x;
        float center = __half2float(u[idx]);
        float lap = 3.0f * d_coeffs_3d16[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_3d16[m] * (
                __half2float(u[idx - m]) + __half2float(u[idx + m])
              + __half2float(u[idx - static_cast<size_t>(m) * N]) + __half2float(u[idx + static_cast<size_t>(m) * N])
              + __half2float(u[idx - static_cast<size_t>(m) * N * N]) + __half2float(u[idx + static_cast<size_t>(m) * N * N]));
        }
        float prev = __half2float(u_prev[idx]);
        u_next[idx] = __float2half(2.0f * center - prev + s * lap);
    }
}

__global__ void first_step_3d_fp16_kahan(const __half* __restrict__ u0,
                                         __half* __restrict__ u1,
                                         float* __restrict__ c1,
                                         int N, float s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_reach_3d16;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = static_cast<size_t>(z) * N * N + y * N + x;
        float center = __half2float(u0[idx]);
        float lap = 3.0f * d_coeffs_3d16[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_3d16[m] * (
                __half2float(u0[idx - m]) + __half2float(u0[idx + m])
              + __half2float(u0[idx - static_cast<size_t>(m) * N]) + __half2float(u0[idx + static_cast<size_t>(m) * N])
              + __half2float(u0[idx - static_cast<size_t>(m) * N * N]) + __half2float(u0[idx + static_cast<size_t>(m) * N * N]));
        }
        float exact = center + 0.5f * s * lap;
        __half stored = __float2half(exact);
        u1[idx] = stored;
        c1[idx] = exact - __half2float(stored);
    }
}

__global__ void wave3d_fp16_kahan_kernel(const __half* __restrict__ u_prev,
                                         const __half* __restrict__ u,
                                         __half* __restrict__ u_next,
                                         const float* __restrict__ c_prev,
                                         const float* __restrict__ c,
                                         float* __restrict__ c_next,
                                         int N, float s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_reach_3d16;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = static_cast<size_t>(z) * N * N + y * N + x;
        float center = __half2float(u[idx]) + c[idx];
        float lap = 3.0f * d_coeffs_3d16[0] * center;
        for (int m = 1; m <= R; m++) {
            size_t ox = static_cast<size_t>(m);
            size_t oy = static_cast<size_t>(m) * N;
            size_t oz = static_cast<size_t>(m) * N * N;
            float xm = __half2float(u[idx - ox]) + c[idx - ox];
            float xp = __half2float(u[idx + ox]) + c[idx + ox];
            float ym = __half2float(u[idx - oy]) + c[idx - oy];
            float yp = __half2float(u[idx + oy]) + c[idx + oy];
            float zm = __half2float(u[idx - oz]) + c[idx - oz];
            float zp = __half2float(u[idx + oz]) + c[idx + oz];
            lap += d_coeffs_3d16[m] * (xm + xp + ym + yp + zm + zp);
        }
        float prev = __half2float(u_prev[idx]) + c_prev[idx];
        float exact = 2.0f * center - prev + s * lap;
        __half stored = __float2half(exact);
        u_next[idx] = stored;
        c_next[idx] = exact - __half2float(stored);
    }
}

__global__ void apply_dirichlet_3d_fp16(__half* u, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b >= N) return;

    __half z = __float2half(0.0f);
    for (int r = 0; r < R; r++) {
        u[static_cast<size_t>(a) * N * N + b * N + r] = z;
        u[static_cast<size_t>(a) * N * N + b * N + (N - 1 - r)] = z;
        u[static_cast<size_t>(a) * N * N + r * N + b] = z;
        u[static_cast<size_t>(a) * N * N + (N - 1 - r) * N + b] = z;
        u[static_cast<size_t>(r) * N * N + a * N + b] = z;
        u[static_cast<size_t>(N - 1 - r) * N * N + a * N + b] = z;
    }
}

__global__ void apply_dirichlet_3d_float(float* c, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b >= N) return;

    for (int r = 0; r < R; r++) {
        c[static_cast<size_t>(a) * N * N + b * N + r] = 0.0f;
        c[static_cast<size_t>(a) * N * N + b * N + (N - 1 - r)] = 0.0f;
        c[static_cast<size_t>(a) * N * N + r * N + b] = 0.0f;
        c[static_cast<size_t>(a) * N * N + (N - 1 - r) * N + b] = 0.0f;
        c[static_cast<size_t>(r) * N * N + a * N + b] = 0.0f;
        c[static_cast<size_t>(N - 1 - r) * N * N + a * N + b] = 0.0f;
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

static std::vector<__half> float_to_half_3d(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); i++) h[i] = __float2half(f[i]);
    return h;
}

StencilResult run_cuda_fp16_naive_3d(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float s = (cfg.c * cfg.dt / cfg.dx) * (cfg.c * cfg.dt / cfg.dx);
    size_t total = static_cast<size_t>(N) * N * N;
    size_t half_bytes = total * sizeof(__half);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_3d16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_3d16, &R, sizeof(int)));

    std::vector<float> h_u0(total, cfg.disp_initial);
    init_wave_field_3d(h_u0, N, cfg.source_amplitude, cfg.source_sigma);
    auto h_half = float_to_half_3d(h_u0);

    __half *d_u_prev, *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u_prev, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMemcpy(d_u_prev, h_half.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u, 0, half_bytes));
    CUDA_CHECK(cudaMemset(d_u_next, 0, half_bytes));

    dim3 block(BX, BY, BZ);
    dim3 grid((N + BX - 1) / BX, (N + BY - 1) / BY, (N + BZ - 1) / BZ);
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    first_step_3d_fp16_naive<<<grid, block>>>(d_u_prev, d_u, N, s);
    apply_dirichlet_3d_fp16<<<bc_grid, bc_block>>>(d_u_prev, N, R);
    apply_dirichlet_3d_fp16<<<bc_grid, bc_block>>>(d_u, N, R);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 1; t < cfg.timesteps; t++) {
        wave3d_fp16_naive_kernel<<<grid, block>>>(d_u_prev, d_u, d_u_next, N, s);
        apply_dirichlet_3d_fp16<<<bc_grid, bc_block>>>(d_u_next, N, R);
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
    std::vector<float> result(total);
    for (size_t i = 0; i < total; i++) result[i] = __half2float(h_half[i]);

    int interior = N - 2 * R;
    double reads = (2.0 * 3.0 * R + 2.0);
    double writes = 1.0;
    double bytes_per_step = static_cast<double>(interior) * interior * interior * (reads + writes) * sizeof(__half);
    double bw = bytes_per_step * (cfg.timesteps - 1) / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp16_naive_3d";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes = 3 * half_bytes;
    res.final_grid = result;

    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

StencilResult run_cuda_fp16_kahan_3d(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float s = (cfg.c * cfg.dt / cfg.dx) * (cfg.c * cfg.dt / cfg.dx);
    size_t total = static_cast<size_t>(N) * N * N;
    size_t half_bytes = total * sizeof(__half);
    size_t float_bytes = total * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_3d16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_3d16, &R, sizeof(int)));

    std::vector<float> h_u0(total, cfg.disp_initial);
    init_wave_field_3d(h_u0, N, cfg.source_amplitude, cfg.source_sigma);
    auto h_half = float_to_half_3d(h_u0);
    std::vector<float> h_comp(total, 0.0f);

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

    dim3 block(BX, BY, BZ);
    dim3 grid((N + BX - 1) / BX, (N + BY - 1) / BY, (N + BZ - 1) / BZ);
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    first_step_3d_fp16_kahan<<<grid, block>>>(d_u_prev, d_u, d_c, N, s);
    apply_dirichlet_3d_fp16<<<bc_grid, bc_block>>>(d_u_prev, N, R);
    apply_dirichlet_3d_fp16<<<bc_grid, bc_block>>>(d_u, N, R);
    apply_dirichlet_3d_float<<<bc_grid, bc_block>>>(d_c_prev, N, R);
    apply_dirichlet_3d_float<<<bc_grid, bc_block>>>(d_c, N, R);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 1; t < cfg.timesteps; t++) {
        wave3d_fp16_kahan_kernel<<<grid, block>>>(d_u_prev, d_u, d_u_next, d_c_prev, d_c, d_c_next, N, s);
        apply_dirichlet_3d_fp16<<<bc_grid, bc_block>>>(d_u_next, N, R);
        apply_dirichlet_3d_float<<<bc_grid, bc_block>>>(d_c_next, N, R);

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

    std::vector<float> result(total);
    for (size_t i = 0; i < total; i++) result[i] = __half2float(h_half[i]) + h_comp[i];

    int interior = N - 2 * R;
    double reads = (2.0 * 3.0 * R + 2.0);
    double writes = 1.0;
    double half_rw = (reads + writes) * sizeof(__half);
    double comp_rw = (reads + writes) * sizeof(float);
    double bytes_per_step = static_cast<double>(interior) * interior * interior * (half_rw + comp_rw);
    double bw = bytes_per_step * (cfg.timesteps - 1) / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp16_kahan_3d";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes = 3 * half_bytes + 3 * float_bytes;
    res.final_grid = result;

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

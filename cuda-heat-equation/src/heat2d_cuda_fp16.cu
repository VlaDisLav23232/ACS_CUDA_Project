#include "stencil.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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
__constant__ int   d_reach_fp16;

__global__ void heat2d_fp16_naive_kernel(const __half* __restrict__ u,
                                          __half* __restrict__ u_next,
                                          int N, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int R = d_reach_fp16;

    if (i >= R && i < N - R && j >= R && j < N - R) {
        float center = __half2float(u[j * N + i]);
        float lap = 2.0f * d_coeffs_fp16[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_fp16[m] * (__half2float(u[j * N + (i - m)]) + __half2float(u[j * N + (i + m)])
                                     + __half2float(u[(j - m) * N + i]) + __half2float(u[(j + m) * N + i]));
        }
        u_next[j * N + i] = __float2half(center + r * lap);
    }
}

__global__ void heat2d_fp16_kahan_kernel(const __half* __restrict__ u,
                                          __half* __restrict__ u_next,
                                          const float* __restrict__ c,
                                          float* __restrict__ c_next,
                                          int N, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int R = d_reach_fp16;

    if (i >= R && i < N - R && j >= R && j < N - R) {
        int idx = j * N + i;
        float center = __half2float(u[idx]) + c[idx];
        float lap = 2.0f * d_coeffs_fp16[0] * center;
        for (int m = 1; m <= R; m++) {
            int xm_idx = j * N + (i - m);
            int xp_idx = j * N + (i + m);
            int ym_idx = (j - m) * N + i;
            int yp_idx = (j + m) * N + i;
            float xm = __half2float(u[xm_idx]) + c[xm_idx];
            float xp = __half2float(u[xp_idx]) + c[xp_idx];
            float ym = __half2float(u[ym_idx]) + c[ym_idx];
            float yp = __half2float(u[yp_idx]) + c[yp_idx];
            lap += d_coeffs_fp16[m] * (xm + xp + ym + yp);
        }
        float exact_result = center + r * lap;
        __half stored = __float2half(exact_result);
        u_next[idx] = stored;
        volatile float stored_back = __half2float(stored);
        c_next[idx] = exact_result - stored_back;
    }
}

__global__ void apply_neumann_bc_fp16(__half* u, int N, int R) {
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

__global__ void apply_neumann_bc_comp(float* c, int N, int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int b = R - 1; b >= 0; b--) {
            c[b * N + idx]       = c[(b + 1) * N + idx];
            c[(N-1-b) * N + idx] = c[(N-2-b) * N + idx];
            c[idx * N + b]       = c[idx * N + (b + 1)];
            c[idx * N + (N-1-b)] = c[idx * N + (N-2-b)];
        }
    }
}

static std::vector<__half> float_to_half(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); i++)
        h[i] = __float2half(f[i]);
    return h;
}

StencilResult run_cuda_fp16_naive(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t n_elems = N * N;
    size_t half_bytes = n_elems * sizeof(__half);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_fp16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_fp16, &R, sizeof(int)));

    std::vector<float> h_f(n_elems, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++)
        for (int i = src_start; i < src_start + src_size; i++)
            h_f[j * N + i] = cfg.temp_source;

    auto h_data = float_to_half(h_f);

    __half *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);
    int bc_threads = 256;
    int bc_blocks = (N + bc_threads - 1) / bc_threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat2d_fp16_naive_kernel<<<grid, block>>>(d_u, d_u_next, N, r);
        apply_neumann_bc_fp16<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        __half* tmp = d_u; d_u = d_u_next; d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(n_elems);
    for (size_t i = 0; i < n_elems; i++)
        result_f[i] = __half2float(h_data[i]);

    // minimum DRAM bandwidth: each point read once, written once
    double bytes_per_step = 2.0 * (double)N * N * sizeof(__half);
    double total_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp16_naive";
    res.grid_size = N;
    res.dim = cfg.dim;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes = 2 * half_bytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

StencilResult run_cuda_fp16_kahan(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t n_elems = N * N;
    size_t half_bytes = n_elems * sizeof(__half);
    size_t float_bytes = n_elems * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_fp16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_fp16, &R, sizeof(int)));

    std::vector<float> h_f(n_elems, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++)
        for (int i = src_start; i < src_start + src_size; i++)
            h_f[j * N + i] = cfg.temp_source;

    auto h_data = float_to_half(h_f);
    std::vector<float> h_comp(n_elems, 0.0f);

    __half *d_u, *d_u_next;
    float *d_c, *d_c_next;
    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_next, float_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_next, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);
    int bc_threads = 256;
    int bc_blocks = (N + bc_threads - 1) / bc_threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat2d_fp16_kahan_kernel<<<grid, block>>>(d_u, d_u_next, d_c, d_c_next, N, r);
        apply_neumann_bc_fp16<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        apply_neumann_bc_comp<<<bc_blocks, bc_threads>>>(d_c_next, N, R);
        __half* tmp_h = d_u; d_u = d_u_next; d_u_next = tmp_h;
        float* tmp_c = d_c; d_c = d_c_next; d_c_next = tmp_c;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(n_elems);
    for (size_t i = 0; i < n_elems; i++)
        result_f[i] = __half2float(h_data[i]) + h_comp[i];

    // minimum DRAM bandwidth: half grid (r+w) + comp grid (r+w)
    double bytes_per_step = 2.0 * (double)N * N * sizeof(__half) + 2.0 * (double)N * N * sizeof(float);
    double total_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp16_kahan";
    res.grid_size = N;
    res.dim = cfg.dim;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes = 2 * half_bytes + float_bytes;  // single compensation buffer
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_c_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

// Fused time-loop kernel: center comp stays in register across timesteps
__global__ void heat2d_fp16_kahan_reg_kernel(
    __half* u_a, __half* u_b, float* c_a, float* c_b,
    int N, float r, int T)
{
    cg::grid_group grid = cg::this_grid();
    int R = d_reach_fp16;
    int total = N * N;

    __half* u_src = u_a;
    __half* u_dst = u_b;
    float* c_src = c_a;
    float* c_dst = c_b;

    for (int t = 0; t < T; t++) {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < total;
             idx += gridDim.x * blockDim.x)
        {
            int i = idx % N;
            int j = idx / N;
            if (i >= R && i < N - R && j >= R && j < N - R) {
                float center = __half2float(u_src[idx]) + c_src[idx];
                float lap = 2.0f * d_coeffs_fp16[0] * center;
                for (int m = 1; m <= R; m++) {
                    int xm = j * N + (i - m);
                    int xp = j * N + (i + m);
                    int ym = (j - m) * N + i;
                    int yp = (j + m) * N + i;
                    lap += d_coeffs_fp16[m] * (
                        (__half2float(u_src[xm]) + c_src[xm]) +
                        (__half2float(u_src[xp]) + c_src[xp]) +
                        (__half2float(u_src[ym]) + c_src[ym]) +
                        (__half2float(u_src[yp]) + c_src[yp]));
                }
                float exact = center + r * lap;
                __half stored = __float2half(exact);
                u_dst[idx] = stored;
                float sb = __half2float(stored);
                c_dst[idx] = exact - sb;
            }
        }
        grid.sync();
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += gridDim.x * blockDim.x)
        {
            for (int b = R - 1; b >= 0; b--) {
                u_dst[b * N + idx]       = u_dst[(b + 1) * N + idx];
                u_dst[(N-1-b) * N + idx] = u_dst[(N-2-b) * N + idx];
                u_dst[idx * N + b]       = u_dst[idx * N + (b + 1)];
                u_dst[idx * N + (N-1-b)] = u_dst[idx * N + (N-2-b)];
                c_dst[b * N + idx]       = c_dst[(b + 1) * N + idx];
                c_dst[(N-1-b) * N + idx] = c_dst[(N-2-b) * N + idx];
                c_dst[idx * N + b]       = c_dst[idx * N + (b + 1)];
                c_dst[idx * N + (N-1-b)] = c_dst[idx * N + (N-2-b)];
            }
        }
        grid.sync();
        __half* tmp_u = u_src; u_src = u_dst; u_dst = tmp_u;
        float*  tmp_c = c_src; c_src = c_dst; c_dst = tmp_c;
    }
}

StencilResult run_cuda_fp16_kahan_reg(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t n_elems = (size_t)N * N;
    size_t half_bytes = n_elems * sizeof(__half);
    size_t float_bytes = n_elems * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_fp16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_fp16, &R, sizeof(int)));

    std::vector<float> h_f(n_elems, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++)
        for (int i = src_start; i < src_start + src_size; i++)
            h_f[j * N + i] = cfg.temp_source;

    auto h_data = float_to_half(h_f);
    std::vector<float> h_comp(n_elems, 0.0f);

    __half *d_u_a, *d_u_b;
    float *d_c_a, *d_c_b;
    CUDA_CHECK(cudaMalloc(&d_u_a, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_b, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_a, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_b, float_bytes));
    CUDA_CHECK(cudaMemcpy(d_u_a, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_b, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_a, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_b, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));

    int block_size = 256;
    int max_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, heat2d_fp16_kahan_reg_kernel, block_size, 0));
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int num_blocks = max_blocks_per_sm * prop.multiProcessorCount;

    int T = cfg.timesteps;
    void* args[] = { &d_u_a, &d_u_b, &d_c_a, &d_c_b, &N, &r, &T };

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)heat2d_fp16_kahan_reg_kernel,
        dim3(num_blocks), dim3(block_size), args));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    __half* d_final_u = (T % 2 == 0) ? d_u_a : d_u_b;
    float*  d_final_c = (T % 2 == 0) ? d_c_a : d_c_b;

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_final_u, half_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_final_c, float_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(n_elems);
    for (size_t i = 0; i < n_elems; i++)
        result_f[i] = __half2float(h_data[i]) + h_comp[i];

    double bytes_per_step = 2.0 * n_elems * sizeof(__half) + 2.0 * n_elems * sizeof(float);
    double total_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp16_kahan_reg";
    res.grid_size = N;
    res.dim = cfg.dim;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes = 2 * half_bytes + 2 * float_bytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u_a));
    CUDA_CHECK(cudaFree(d_u_b));
    CUDA_CHECK(cudaFree(d_c_a));
    CUDA_CHECK(cudaFree(d_c_b));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

#include "stencil.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>

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
static const int TILE_X = 16;
static const int TILE_Y = 16;

__constant__ float d_coeffs_async[MAX_REACH + 1];
__constant__ int d_reach_async;

// Plain 3D async variant (no 2.5D tiling):
// we asynchronously stage the fp32 compensation term for each thread in shared memory.
__global__ void heat3d_fp16_kahan_async_kernel(const __half* __restrict__ u,
                                               __half* __restrict__ u_next,
                                               const float* __restrict__ c,
                                               float* __restrict__ c_next,
                                               int N, float r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_reach_async;

    int t = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    extern __shared__ float smem_c[];

    if (x < N && y < N && z < N) {
        size_t idx = (size_t)z * N * N + y * N + x;
        cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
        pipe.producer_acquire();
        cuda::memcpy_async(&smem_c[t], &c[idx], sizeof(float), pipe);
        pipe.producer_commit();
        cuda::pipeline_consumer_wait_prior<0>(pipe);
    }
    __syncthreads();

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = (size_t)z * N * N + y * N + x;
        float center = __half2float(u[idx]) + smem_c[t];
        float lap = 3.0f * d_coeffs_async[0] * center;

        for (int m = 1; m <= R; m++) {
            size_t ox = (size_t)m;
            size_t oy = (size_t)m * N;
            size_t oz = (size_t)m * N * N;
            float xm = __half2float(u[idx - ox]) + c[idx - ox];
            float xp = __half2float(u[idx + ox]) + c[idx + ox];
            float ym = __half2float(u[idx - oy]) + c[idx - oy];
            float yp = __half2float(u[idx + oy]) + c[idx + oy];
            float zm = __half2float(u[idx - oz]) + c[idx - oz];
            float zp = __half2float(u[idx + oz]) + c[idx + oz];
            lap += d_coeffs_async[m] * (xm + xp + ym + yp + zm + zp);
        }

        float exact_result = center + r * lap;
        __half stored = __float2half(exact_result);
        u_next[idx] = stored;
        volatile float stored_back = __half2float(stored);
        c_next[idx] = exact_result - stored_back;
    }
}

__global__ void heat3d_fp16_kahan_25d_async_kernel(const __half* __restrict__ u,
                                                   __half* __restrict__ u_next,
                                                   const float* __restrict__ c,
                                                   float* __restrict__ c_next,
                                                   int N, float r) {
    int R = d_reach_async;
    int tile_x = blockIdx.x * TILE_X + threadIdx.x;
    int tile_y = blockIdx.y * TILE_Y + threadIdx.y;
    int z = blockIdx.z;

    int smem_w = TILE_X + 2 * R;
    int smem_size = smem_w * (TILE_Y + 2 * R);

    extern __shared__ char smem_raw[];
    __half* smem_u = (__half*)smem_raw;
    float* smem_c = (float*)(smem_raw + smem_size * sizeof(__half));

    int tid = threadIdx.y * TILE_X + threadIdx.x;
    int n_threads = TILE_X * TILE_Y;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    pipe.producer_acquire();
    for (int idx = tid; idx < smem_size; idx += n_threads) {
        int sy = idx / smem_w;
        int sx = idx % smem_w;
        int gx = blockIdx.x * TILE_X + sx - R;
        int gy = blockIdx.y * TILE_Y + sy - R;
        gx = max(0, min(gx, N - 1));
        gy = max(0, min(gy, N - 1));
        size_t gidx = (size_t)z * N * N + gy * N + gx;
        smem_u[idx] = u[gidx];
        cuda::memcpy_async(&smem_c[idx], &c[gidx], sizeof(float), pipe);
    }
    pipe.producer_commit();
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    if (tile_x >= R && tile_x < N - R &&
        tile_y >= R && tile_y < N - R &&
        z >= R && z < N - R) {

        int lx = threadIdx.x + R;
        int ly = threadIdx.y + R;

        float center = __half2float(smem_u[ly * smem_w + lx]) + smem_c[ly * smem_w + lx];
        float lap = 3.0f * d_coeffs_async[0] * center;

        for (int m = 1; m <= R; m++) {
            float xm = __half2float(smem_u[ly * smem_w + (lx - m)]) + smem_c[ly * smem_w + (lx - m)];
            float xp = __half2float(smem_u[ly * smem_w + (lx + m)]) + smem_c[ly * smem_w + (lx + m)];
            float ym = __half2float(smem_u[(ly - m) * smem_w + lx]) + smem_c[(ly - m) * smem_w + lx];
            float yp = __half2float(smem_u[(ly + m) * smem_w + lx]) + smem_c[(ly + m) * smem_w + lx];

            size_t idx_center = (size_t)z * N * N + tile_y * N + tile_x;
            float zm = __half2float(u[idx_center - (size_t)m * N * N]) + c[idx_center - (size_t)m * N * N];
            float zp = __half2float(u[idx_center + (size_t)m * N * N]) + c[idx_center + (size_t)m * N * N];
            lap += d_coeffs_async[m] * (xm + xp + ym + yp + zm + zp);
        }

        float exact_result = center + r * lap;
        __half stored = __float2half(exact_result);
        size_t out_idx = (size_t)z * N * N + tile_y * N + tile_x;
        u_next[out_idx] = stored;
        volatile float stored_back = __half2float(stored);
        c_next[out_idx] = exact_result - stored_back;
    }
}

__global__ void apply_neumann_bc_3d_async_fp16(__half* u, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;

    for (int b = R - 1; b >= 0; b--) {
        u[(size_t)a * N * N + b_idx * N + b]       = u[(size_t)a * N * N + b_idx * N + (b + 1)];
        u[(size_t)a * N * N + b_idx * N + (N - 1 - b)] = u[(size_t)a * N * N + b_idx * N + (N - 2 - b)];
        u[(size_t)a * N * N + b * N + b_idx]       = u[(size_t)a * N * N + (b + 1) * N + b_idx];
        u[(size_t)a * N * N + (N - 1 - b) * N + b_idx] = u[(size_t)a * N * N + (N - 2 - b) * N + b_idx];
        u[(size_t)b * N * N + a * N + b_idx]       = u[(size_t)(b + 1) * N * N + a * N + b_idx];
        u[(size_t)(N - 1 - b) * N * N + a * N + b_idx] = u[(size_t)(N - 2 - b) * N * N + a * N + b_idx];
    }
}

__global__ void apply_neumann_bc_3d_async_comp(float* c, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;

    for (int b = R - 1; b >= 0; b--) {
        c[(size_t)a * N * N + b_idx * N + b]       = c[(size_t)a * N * N + b_idx * N + (b + 1)];
        c[(size_t)a * N * N + b_idx * N + (N - 1 - b)] = c[(size_t)a * N * N + b_idx * N + (N - 2 - b)];
        c[(size_t)a * N * N + b * N + b_idx]       = c[(size_t)a * N * N + (b + 1) * N + b_idx];
        c[(size_t)a * N * N + (N - 1 - b) * N + b_idx] = c[(size_t)a * N * N + (N - 2 - b) * N + b_idx];
        c[(size_t)b * N * N + a * N + b_idx]       = c[(size_t)(b + 1) * N * N + a * N + b_idx];
        c[(size_t)(N - 1 - b) * N * N + a * N + b_idx] = c[(size_t)(N - 2 - b) * N * N + a * N + b_idx];
    }
}

static std::vector<__half> float_to_half_async(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); i++) {
        h[i] = __float2half(f[i]);
    }
    return h;
}

StencilResult run_cuda_fp16_kahan_3d_async(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total = (size_t)N * N * N;
    size_t half_bytes = total * sizeof(__half);
    size_t float_bytes = total * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_async, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_async, &R, sizeof(int)));

    std::vector<float> h_f(total, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int z = src_start; z < src_start + src_size; z++)
        for (int y = src_start; y < src_start + src_size; y++)
            for (int x = src_start; x < src_start + src_size; x++)
                h_f[(size_t)z * N * N + y * N + x] = cfg.temp_source;

    auto h_data = float_to_half_async(h_f);
    std::vector<float> h_comp(total, 0.0f);

    __half* d_u = nullptr;
    __half* d_u_next = nullptr;
    float* d_c = nullptr;
    float* d_c_next = nullptr;

    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_next, float_bytes));

    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_next, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));

    dim3 block(BX, BY, BZ);
    dim3 grid((N + BX - 1) / BX, (N + BY - 1) / BY, (N + BZ - 1) / BZ);

    size_t smem_bytes = BX * BY * BZ * sizeof(float);

    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; ++t) {
        heat3d_fp16_kahan_async_kernel<<<grid, block, smem_bytes>>>(d_u, d_u_next, d_c, d_c_next, N, r);
        apply_neumann_bc_3d_async_fp16<<<bc_grid, bc_block>>>(d_u_next, N, R);
        apply_neumann_bc_3d_async_comp<<<bc_grid, bc_block>>>(d_c_next, N, R);

        __half* tmp_h = d_u;
        d_u = d_u_next;
        d_u_next = tmp_h;

        float* tmp_c = d_c;
        d_c = d_c_next;
        d_c_next = tmp_c;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(total);
    for (size_t i = 0; i < total; i++) {
        result_f[i] = __half2float(h_data[i]) + h_comp[i];
    }

    double bytes_per_step = 2.0 * (double)N * N * N * sizeof(__half)
                          + 2.0 * (double)N * N * N * sizeof(float);
    double total_bw_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bw_bytes / (elapsed_ms / 1000.0) / 1e9;
    double mpts = ((double)N * N * N * cfg.timesteps) / (elapsed_ms / 1000.0) / 1e6;

    StencilResult res;
    res.variant_name = "cuda_fp16_kahan_3d_async";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * half_bytes + 2 * float_bytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_c_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

StencilResult run_cuda_fp16_kahan_3d_25d_async(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total = (size_t)N * N * N;
    size_t half_bytes = total * sizeof(__half);
    size_t float_bytes = total * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_async, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_async, &R, sizeof(int)));

    std::vector<float> h_f(total, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int z = src_start; z < src_start + src_size; z++)
        for (int y = src_start; y < src_start + src_size; y++)
            for (int x = src_start; x < src_start + src_size; x++)
                h_f[(size_t)z * N * N + y * N + x] = cfg.temp_source;

    auto h_data = float_to_half_async(h_f);
    std::vector<float> h_comp(total, 0.0f);

    __half* d_u = nullptr;
    __half* d_u_next = nullptr;
    float* d_c = nullptr;
    float* d_c_next = nullptr;

    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_next, float_bytes));

    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_next, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_X, TILE_Y);
    dim3 grid25((N + TILE_X - 1) / TILE_X, (N + TILE_Y - 1) / TILE_Y, N);

    int smem_tile = (TILE_X + 2 * R) * (TILE_Y + 2 * R);
    size_t smem_bytes = smem_tile * sizeof(__half) + smem_tile * sizeof(float);

    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat3d_fp16_kahan_25d_async_kernel<<<grid25, block, smem_bytes>>>(d_u, d_u_next, d_c, d_c_next, N, r);
        apply_neumann_bc_3d_async_fp16<<<bc_grid, bc_block>>>(d_u_next, N, R);
        apply_neumann_bc_3d_async_comp<<<bc_grid, bc_block>>>(d_c_next, N, R);

        __half* tmp_h = d_u;
        d_u = d_u_next;
        d_u_next = tmp_h;
        float* tmp_c = d_c;
        d_c = d_c_next;
        d_c_next = tmp_c;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(total);
    for (size_t i = 0; i < total; i++)
        result_f[i] = __half2float(h_data[i]) + h_comp[i];

    double bytes_per_step = 2.0 * (double)N * N * N * sizeof(__half)
                          + 2.0 * (double)N * N * N * sizeof(float);
    double total_bw_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bw_bytes / (elapsed_ms / 1000.0) / 1e9;
    double mpts = ((double)N * N * N * cfg.timesteps) / (elapsed_ms / 1000.0) / 1e6;

    StencilResult res;
    res.variant_name = "cuda_fp16_kahan_3d_25d_async";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * half_bytes + 2 * float_bytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_c_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

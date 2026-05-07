#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "stencil.h"
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

using cfp16_t = uint16_t;

__constant__ float d_coeffs_cfp16[MAX_REACH + 1];
__constant__ int   d_reach_cfp16;

__host__ __device__ inline uint32_t float_as_uint(float x) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = x;
    return bits.u;
}

__host__ __device__ inline float uint_as_float(uint32_t x) {
    union {
        uint32_t u;
        float f;
    } bits;
    bits.u = x;
    return bits.f;
}

__host__ __device__ inline float cfp16_to_float(cfp16_t x) {
    const uint32_t e = (static_cast<uint32_t>(x) & 0x7800u) >> 11;
    const uint32_t m = (static_cast<uint32_t>(x) & 0x07FFu) << 12;
    const uint32_t v = float_as_uint(static_cast<float>(m)) >> 23;
    return uint_as_float(
        (static_cast<uint32_t>(x & 0x8000u)) << 16
        | (e != 0u) * (((e + 112u) << 23) | m)
        | ((e == 0u) & (m != 0u)) * (((v - 37u) << 23) | ((m << (150u - v)) & 0x007FF000u))
    );
}

__host__ __device__ inline cfp16_t float_to_cfp16(float x) {
    const uint32_t b = float_as_uint(x) + 0x00000800u;
    const uint32_t e = (b & 0x7F800000u) >> 23;
    const uint32_t m = b & 0x007FFFFFu;
    return static_cast<cfp16_t>(
        ((b & 0x80000000u) >> 16)
        | (e > 112u) * ((((e - 112u) << 11) & 0x7800u) | (m >> 12))
        | ((e < 113u) & (e > 100u)) * (((((0x7FF800u + m) >> (124u - e)) + 1u) >> 1))
    );
}

__global__ void heat2d_cfp16_naive_kernel(const cfp16_t* __restrict__ u,
                                          cfp16_t* __restrict__ u_next,
                                          int N, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int R = d_reach_cfp16;

    if (i >= R && i < N - R && j >= R && j < N - R) {
        float center = cfp16_to_float(u[j * N + i]);
        float lap = 2.0f * d_coeffs_cfp16[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_cfp16[m] * (cfp16_to_float(u[j * N + (i - m)]) + cfp16_to_float(u[j * N + (i + m)])
                                      + cfp16_to_float(u[(j - m) * N + i]) + cfp16_to_float(u[(j + m) * N + i]));
        }
        u_next[j * N + i] = float_to_cfp16(center + r * lap);
    }
}

__global__ void heat2d_cfp16_kahan_kernel(const cfp16_t* __restrict__ u,
                                          cfp16_t* __restrict__ u_next,
                                          const float* __restrict__ c,
                                          float* __restrict__ c_next,
                                          int N, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int R = d_reach_cfp16;

    if (i >= R && i < N - R && j >= R && j < N - R) {
        int idx = j * N + i;
        float center = cfp16_to_float(u[idx]) + c[idx];
        float lap = 2.0f * d_coeffs_cfp16[0] * center;
        for (int m = 1; m <= R; m++) {
            int xm_idx = j * N + (i - m);
            int xp_idx = j * N + (i + m);
            int ym_idx = (j - m) * N + i;
            int yp_idx = (j + m) * N + i;
            float xm = cfp16_to_float(u[xm_idx]) + c[xm_idx];
            float xp = cfp16_to_float(u[xp_idx]) + c[xp_idx];
            float ym = cfp16_to_float(u[ym_idx]) + c[ym_idx];
            float yp = cfp16_to_float(u[yp_idx]) + c[yp_idx];
            lap += d_coeffs_cfp16[m] * (xm + xp + ym + yp);
        }
        float exact_result = center + r * lap;
        cfp16_t stored = float_to_cfp16(exact_result);
        u_next[idx] = stored;
        volatile float stored_back = cfp16_to_float(stored);
        c_next[idx] = exact_result - stored_back;
    }
}

__global__ void heat2d_cfp16_kahan_tiled_kernel(const cfp16_t* __restrict__ u,
                                                cfp16_t* __restrict__ u_next,
                                                const float* __restrict__ c,
                                                float* __restrict__ c_next,
                                                int N, float r) {
    __shared__ float s_u[BLOCK_Y + 2 * MAX_REACH][BLOCK_X + 2 * MAX_REACH];
    __shared__ float s_c[BLOCK_Y + 2 * MAX_REACH][BLOCK_X + 2 * MAX_REACH];

    int R = d_reach_cfp16;
    int tile_w = blockDim.x + 2 * R;
    int tile_h = blockDim.y + 2 * R;
    int tile_elems = tile_w * tile_h;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int base_i = blockIdx.x * blockDim.x - R;
    int base_j = blockIdx.y * blockDim.y - R;

    for (int linear = tid; linear < tile_elems; linear += blockDim.x * blockDim.y) {
        int local_x = linear % tile_w;
        int local_y = linear / tile_w;
        int global_x = base_i + local_x;
        int global_y = base_j + local_y;
        global_x = max(0, min(global_x, N - 1));
        global_y = max(0, min(global_y, N - 1));
        int global_idx = global_y * N + global_x;
        s_u[local_y][local_x] = cfp16_to_float(u[global_idx]);
        s_c[local_y][local_x] = c[global_idx];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= R && i < N - R && j >= R && j < N - R) {
        int idx = j * N + i;
        int li = threadIdx.x + R;
        int lj = threadIdx.y + R;
        float center = s_u[lj][li] + s_c[lj][li];
        float lap = 2.0f * d_coeffs_cfp16[0] * center;
        for (int m = 1; m <= R; m++) {
            float xm = s_u[lj][li - m] + s_c[lj][li - m];
            float xp = s_u[lj][li + m] + s_c[lj][li + m];
            float ym = s_u[lj - m][li] + s_c[lj - m][li];
            float yp = s_u[lj + m][li] + s_c[lj + m][li];
            lap += d_coeffs_cfp16[m] * (xm + xp + ym + yp);
        }
        float exact_result = center + r * lap;
        cfp16_t stored = float_to_cfp16(exact_result);
        u_next[idx] = stored;
        float stored_back = cfp16_to_float(stored);
        c_next[idx] = exact_result - stored_back;
    }
}

__global__ void apply_neumann_bc_cfp16(cfp16_t* u, int N, int R) {
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

__global__ void apply_neumann_bc_comp_cfp16(float* c, int N, int R) {
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

static std::vector<cfp16_t> float_to_cfp16_vector(const std::vector<float>& f) {
    std::vector<cfp16_t> h(f.size());
    for (size_t i = 0; i < f.size(); i++)
        h[i] = float_to_cfp16(f[i]);
    return h;
}

static float cfp16_normalization_scale(const StencilConfig& cfg) {
    float max_abs_temp = fmaxf(fabsf(cfg.temp_initial), fabsf(cfg.temp_source));
    return (max_abs_temp > 1.0f) ? max_abs_temp : 1.0f;
}

static std::vector<float> make_normalized_initial_grid(const StencilConfig& cfg, float scale) {
    int N = cfg.nx;
    size_t n_elems = static_cast<size_t>(N) * N;
    std::vector<float> h_f(n_elems, cfg.temp_initial / scale);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++)
        for (int i = src_start; i < src_start + src_size; i++)
            h_f[j * N + i] = cfg.temp_source / scale;
    return h_f;
}

static void denormalize_result(std::vector<float>& result, float scale) {
    if (scale == 1.0f) {
        return;
    }
    for (float& value : result) {
        value *= scale;
    }
}

StencilResult run_cuda_cfp16_naive(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t n_elems = static_cast<size_t>(N) * N;
    size_t cfp16_bytes = n_elems * sizeof(cfp16_t);
    float normalization_scale = cfp16_normalization_scale(cfg);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_cfp16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_cfp16, &R, sizeof(int)));

    std::vector<float> h_f = make_normalized_initial_grid(cfg, normalization_scale);
    auto h_data = float_to_cfp16_vector(h_f);

    cfp16_t *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u, cfp16_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, cfp16_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), cfp16_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), cfp16_bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);
    int bc_threads = 256;
    int bc_blocks = (N + bc_threads - 1) / bc_threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat2d_cfp16_naive_kernel<<<grid, block>>>(d_u, d_u_next, N, r);
        apply_neumann_bc_cfp16<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        cfp16_t* tmp = d_u; d_u = d_u_next; d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, cfp16_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(n_elems);
    for (size_t i = 0; i < n_elems; i++)
        result_f[i] = cfp16_to_float(h_data[i]);
    denormalize_result(result_f, normalization_scale);

    double bytes_per_step = 2.0 * static_cast<double>(N) * N * sizeof(cfp16_t);
    double total_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;
    double mpts = (static_cast<double>(N) * N * cfg.timesteps) / (elapsed_ms / 1000.0) / 1e6;

    StencilResult res;
    res.variant_name = "cuda_cfp16_naive";
    res.grid_size = N;
    res.dim = cfg.dim;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * cfp16_bytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

StencilResult run_cuda_cfp16_kahan(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t n_elems = static_cast<size_t>(N) * N;
    size_t cfp16_bytes = n_elems * sizeof(cfp16_t);
    size_t float_bytes = n_elems * sizeof(float);
    float normalization_scale = cfp16_normalization_scale(cfg);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_cfp16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_cfp16, &R, sizeof(int)));

    std::vector<float> h_f = make_normalized_initial_grid(cfg, normalization_scale);
    auto h_data = float_to_cfp16_vector(h_f);
    std::vector<float> h_comp(n_elems, 0.0f);

    cfp16_t *d_u, *d_u_next;
    float *d_c, *d_c_next;
    CUDA_CHECK(cudaMalloc(&d_u, cfp16_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, cfp16_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_next, float_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), cfp16_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), cfp16_bytes, cudaMemcpyHostToDevice));
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
        heat2d_cfp16_kahan_kernel<<<grid, block>>>(d_u, d_u_next, d_c, d_c_next, N, r);
        apply_neumann_bc_cfp16<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        apply_neumann_bc_comp_cfp16<<<bc_blocks, bc_threads>>>(d_c_next, N, R);
        cfp16_t* tmp_h = d_u; d_u = d_u_next; d_u_next = tmp_h;
        float* tmp_c = d_c; d_c = d_c_next; d_c_next = tmp_c;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, cfp16_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(n_elems);
    for (size_t i = 0; i < n_elems; i++)
        result_f[i] = (cfp16_to_float(h_data[i]) + h_comp[i]) * normalization_scale;

    double bytes_per_step = 2.0 * static_cast<double>(N) * N * sizeof(cfp16_t)
                          + 2.0 * static_cast<double>(N) * N * sizeof(float);
    double total_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;
    double mpts = (static_cast<double>(N) * N * cfg.timesteps) / (elapsed_ms / 1000.0) / 1e6;

    StencilResult res;
    res.variant_name = "cuda_cfp16_kahan";
    res.grid_size = N;
    res.dim = cfg.dim;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * cfp16_bytes + float_bytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_c_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

StencilResult run_cuda_cfp16_kahan_tiled(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t n_elems = static_cast<size_t>(N) * N;
    size_t cfp16_bytes = n_elems * sizeof(cfp16_t);
    size_t float_bytes = n_elems * sizeof(float);
    float normalization_scale = cfp16_normalization_scale(cfg);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_cfp16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_cfp16, &R, sizeof(int)));

    std::vector<float> h_f = make_normalized_initial_grid(cfg, normalization_scale);
    auto h_data = float_to_cfp16_vector(h_f);
    std::vector<float> h_comp(n_elems, 0.0f);

    cfp16_t *d_u, *d_u_next;
    float *d_c, *d_c_next;
    CUDA_CHECK(cudaMalloc(&d_u, cfp16_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, cfp16_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_next, float_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), cfp16_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), cfp16_bytes, cudaMemcpyHostToDevice));
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
        heat2d_cfp16_kahan_tiled_kernel<<<grid, block>>>(d_u, d_u_next, d_c, d_c_next, N, r);
        apply_neumann_bc_cfp16<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        apply_neumann_bc_comp_cfp16<<<bc_blocks, bc_threads>>>(d_c_next, N, R);
        cfp16_t* tmp_h = d_u; d_u = d_u_next; d_u_next = tmp_h;
        float* tmp_c = d_c; d_c = d_c_next; d_c_next = tmp_c;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, cfp16_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(n_elems);
    for (size_t i = 0; i < n_elems; i++)
        result_f[i] = (cfp16_to_float(h_data[i]) + h_comp[i]) * normalization_scale;

    double bytes_per_step = 2.0 * static_cast<double>(N) * N * sizeof(cfp16_t)
                          + 2.0 * static_cast<double>(N) * N * sizeof(float);
    double total_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;
    double mpts = (static_cast<double>(N) * N * cfg.timesteps) / (elapsed_ms / 1000.0) / 1e6;

    StencilResult res;
    res.variant_name = "cuda_cfp16_kahan_tiled";
    res.grid_size = N;
    res.dim = cfg.dim;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * cfp16_bytes + float_bytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_c_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

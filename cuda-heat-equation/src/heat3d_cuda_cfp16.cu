#include "stencil.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cfp16.cu>

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err));                  \
        exit(1);                                                               \
    }                                                                          \
} while (0)


// Block dimensions
static const int BX = 8;
static const int BY = 8;
static const int BZ = 4;

__constant__ float d_coeffs_3d16[MAX_REACH + 1];
__constant__ int   d_reach_3d16;

static float cfp16_normalization_scale_3d(const StencilConfig& cfg) {
    float max_abs_temp = fmaxf(fabsf(cfg.temp_initial), fabsf(cfg.temp_source));
    // If already within cfp16 range with headroom, no scaling needed.
    return (max_abs_temp >= 1.0f) ? max_abs_temp : 1.0f;
}

static std::vector<float> make_normalized_initial_grid_3d(
        const StencilConfig& cfg, float scale) {
    int    N       = cfg.nx;
    size_t n_elems = (size_t)N * N * N;

    std::vector<float> h_f(n_elems, cfg.temp_initial / scale);

    int src_size  = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int z = src_start; z < src_start + src_size; z++)
        for (int y = src_start; y < src_start + src_size; y++)
            for (int x = src_start; x < src_start + src_size; x++)
                h_f[(size_t)z * N * N + y * N + x] = cfg.temp_source / scale;

    return h_f;
}

static void denormalize_result_3d(std::vector<float>& result, float scale) {
    if (scale == 1.0f) return;
    for (float& v : result) v *= scale;
}

static std::vector<cfp16_t> float_to_cfp16_vec(const std::vector<float>& f) {
    std::vector<cfp16_t> out(f.size());
    for (size_t i = 0; i < f.size(); i++)
        out[i] = float_to_cfp16(f[i]);
    return out;
}
__global__ void heat3d_cfp16_naive_kernel(
        const cfp16_t* __restrict__ u,
        cfp16_t*       __restrict__ u_next,
        int N, float r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_reach_3d16;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = (size_t)z * N * N + y * N + x;

        float center = cfp16_to_float(u[idx]);
        float lap    = 3.0f * d_coeffs_3d16[0] * center;

        for (int m = 1; m <= R; m++) {
            size_t ox = (size_t)m;
            size_t oy = (size_t)m * N;
            size_t oz = (size_t)m * N * N;
            lap += d_coeffs_3d16[m] * (
                cfp16_to_float(u[idx - ox]) + cfp16_to_float(u[idx + ox]) +
                cfp16_to_float(u[idx - oy]) + cfp16_to_float(u[idx + oy]) +
                cfp16_to_float(u[idx - oz]) + cfp16_to_float(u[idx + oz]));
        }
        u_next[idx] = float_to_cfp16(center + r * lap);
    }
}

__global__ void heat3d_cfp16_kahan_kernel(
        const cfp16_t* __restrict__ u,
        cfp16_t*       __restrict__ u_next,
        const float*   __restrict__ c,
        float*         __restrict__ c_next,
        int N, float r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_reach_3d16;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = (size_t)z * N * N + y * N + x;

        float center = cfp16_to_float(u[idx]) + c[idx];
        float lap    = 3.0f * d_coeffs_3d16[0] * center;

        for (int m = 1; m <= R; m++) {
            size_t ox = (size_t)m;
            size_t oy = (size_t)m * N;
            size_t oz = (size_t)m * N * N;

            float xm = cfp16_to_float(u[idx - ox]) + c[idx - ox];
            float xp = cfp16_to_float(u[idx + ox]) + c[idx + ox];
            float ym = cfp16_to_float(u[idx - oy]) + c[idx - oy];
            float yp = cfp16_to_float(u[idx + oy]) + c[idx + oy];
            float zm = cfp16_to_float(u[idx - oz]) + c[idx - oz];
            float zp = cfp16_to_float(u[idx + oz]) + c[idx + oz];

            lap += d_coeffs_3d16[m] * (xm + xp + ym + yp + zm + zp);
        }

        float   exact_result = center + r * lap;
        cfp16_t stored       = float_to_cfp16(exact_result);
        u_next[idx]          = stored;
        c_next[idx]          = exact_result - cfp16_to_float(stored);
    }
}

__global__ void heat3d_cfp16_kahan_tiled_kernel(
        const cfp16_t* __restrict__ u,
        cfp16_t*       __restrict__ u_next,
        const float*   __restrict__ c,
        float*         __restrict__ c_next,
        int N, float r)
{
    extern __shared__ float shared_mem[];

    int R      = d_reach_3d16;
    int tile_x = blockDim.x + 2 * R;
    int tile_y = blockDim.y + 2 * R;
    int tile_z = blockDim.z + 2 * R;
    int tile_elems = tile_x * tile_y * tile_z;

    float* s_u = shared_mem;
    float* s_c = shared_mem + tile_elems;

    int base_x = blockIdx.x * blockDim.x - R;
    int base_y = blockIdx.y * blockDim.y - R;
    int base_z = blockIdx.z * blockDim.z - R;

    int tid        = threadIdx.z * blockDim.y * blockDim.x
                   + threadIdx.y * blockDim.x
                   + threadIdx.x;
    int block_size = blockDim.x * blockDim.y * blockDim.z;

    // Cooperative halo load: clamp out-of-bounds indices to boundary.
    // Neumann BC kernel corrects the ghost layer afterwards.
    for (int linear = tid; linear < tile_elems; linear += block_size) {
        int local_x = linear % tile_x;
        int local_y = (linear / tile_x) % tile_y;
        int local_z = linear / (tile_x * tile_y);

        int gx = max(0, min(base_x + local_x, N - 1));
        int gy = max(0, min(base_y + local_y, N - 1));
        int gz = max(0, min(base_z + local_z, N - 1));

        size_t g_idx = (size_t)gz * N * N + gy * N + gx;
        size_t t_idx = (size_t)local_z * tile_y * tile_x
                     + (size_t)local_y * tile_x
                     + local_x;

        s_u[t_idx] = cfp16_to_float(u[g_idx]);
        s_c[t_idx] = c[g_idx];
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t g_idx = (size_t)z * N * N + y * N + x;

        // Local tile indices with halo offset
        int lx = threadIdx.x + R;
        int ly = threadIdx.y + R;
        int lz = threadIdx.z + R;

        // Inline tile index helper
        auto tidx = [&](int dz, int dy, int dx) -> size_t {
            return (size_t)(lz + dz) * tile_y * tile_x
                 + (size_t)(ly + dy) * tile_x
                 + (lx + dx);
        };

        float center = s_u[tidx(0, 0, 0)] + s_c[tidx(0, 0, 0)];
        float lap    = 3.0f * d_coeffs_3d16[0] * center;

        for (int m = 1; m <= R; m++) {
            float fxm = s_u[tidx( 0,  0, -m)] + s_c[tidx( 0,  0, -m)];
            float fxp = s_u[tidx( 0,  0, +m)] + s_c[tidx( 0,  0, +m)];
            float fym = s_u[tidx( 0, -m,  0)] + s_c[tidx( 0, -m,  0)];
            float fyp = s_u[tidx( 0, +m,  0)] + s_c[tidx( 0, +m,  0)];
            float fzm = s_u[tidx(-m,  0,  0)] + s_c[tidx(-m,  0,  0)];
            float fzp = s_u[tidx(+m,  0,  0)] + s_c[tidx(+m,  0,  0)];
            lap += d_coeffs_3d16[m] * (fxm + fxp + fym + fyp + fzm + fzp);
        }

        float   exact_result = center + r * lap;
        cfp16_t stored       = float_to_cfp16(exact_result);
        u_next[g_idx]        = stored;
        c_next[g_idx]        = exact_result - cfp16_to_float(stored);
    }
}

__global__ void apply_neumann_bc_3d_cfp16(cfp16_t* u, int N, int R) {
    int a     = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;

    for (int b = R - 1; b >= 0; b--) {
        // X faces
        u[(size_t)a * N * N + b_idx * N + b]       = u[(size_t)a * N * N + b_idx * N + (b + 1)];
        u[(size_t)a * N * N + b_idx * N + (N-1-b)] = u[(size_t)a * N * N + b_idx * N + (N-2-b)];
        // Y faces
        u[(size_t)a * N * N + b * N + b_idx]       = u[(size_t)a * N * N + (b + 1) * N + b_idx];
        u[(size_t)a * N * N + (N-1-b) * N + b_idx] = u[(size_t)a * N * N + (N-2-b) * N + b_idx];
        // Z faces
        u[(size_t)b * N * N + a * N + b_idx]       = u[(size_t)(b + 1) * N * N + a * N + b_idx];
        u[(size_t)(N-1-b) * N * N + a * N + b_idx] = u[(size_t)(N-2-b) * N * N + a * N + b_idx];
    }
}

__global__ void apply_neumann_bc_3d_comp(float* c, int N, int R) {
    int a     = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;

    for (int b = R - 1; b >= 0; b--) {
        c[(size_t)a * N * N + b_idx * N + b]       = c[(size_t)a * N * N + b_idx * N + (b + 1)];
        c[(size_t)a * N * N + b_idx * N + (N-1-b)] = c[(size_t)a * N * N + b_idx * N + (N-2-b)];
        c[(size_t)a * N * N + b * N + b_idx]       = c[(size_t)a * N * N + (b + 1) * N + b_idx];
        c[(size_t)a * N * N + (N-1-b) * N + b_idx] = c[(size_t)a * N * N + (N-2-b) * N + b_idx];
        c[(size_t)b * N * N + a * N + b_idx]       = c[(size_t)(b + 1) * N * N + a * N + b_idx];
        c[(size_t)(N-1-b) * N * N + a * N + b_idx] = c[(size_t)(N-2-b) * N * N + a * N + b_idx];
    }
}

StencilResult run_cuda_cfp16_naive_3d(const StencilConfig& cfg) {
    int    N           = cfg.nx;
    int    R           = cfg.stencil_reach;
    float  r           = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total       = (size_t)N * N * N;
    size_t cfp16_bytes = total * sizeof(cfp16_t);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_3d16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_3d16,  &R,            sizeof(int)));

    // Scale into cfp16 range, encode, upload
    float scale  = cfp16_normalization_scale_3d(cfg);
    auto  h_f    = make_normalized_initial_grid_3d(cfg, scale);
    auto  h_data = float_to_cfp16_vec(h_f);

    cfp16_t *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u,      cfp16_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, cfp16_bytes));
    CUDA_CHECK(cudaMemcpy(d_u,      h_data.data(), cfp16_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), cfp16_bytes, cudaMemcpyHostToDevice));

    dim3 block(BX, BY, BZ);
    dim3 grid3((N + BX - 1) / BX, (N + BY - 1) / BY, (N + BZ - 1) / BZ);
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat3d_cfp16_naive_kernel<<<grid3, block>>>(d_u, d_u_next, N, r);
        apply_neumann_bc_3d_cfp16<<<bc_grid, bc_block>>>(d_u_next, N, R);
        cfp16_t* tmp = d_u; d_u = d_u_next; d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    // Download, decode, scale back to physical temperatures
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, cfp16_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> result_f(total);
    for (size_t i = 0; i < total; i++)
        result_f[i] = cfp16_to_float(h_data[i]);
    denormalize_result_3d(result_f, scale);

    // Minimum DRAM bandwidth: read u + write u_next each step
    double bytes_per_step = 2.0 * (double)total * sizeof(cfp16_t);
    double bw = (bytes_per_step * cfg.timesteps) / (elapsed_ms * 1e-3) / 1e9;

    StencilResult res;
    res.variant_name     = "cuda_cfp16_naive_3d";
    res.grid_size        = N;
    res.dim              = 3;
    res.stencil_reach    = R;
    res.timesteps        = cfg.timesteps;
    res.elapsed_ms       = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes     = 2 * cfp16_bytes;
    res.final_grid       = std::move(result_f);

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return res;
}

StencilResult run_cuda_cfp16_kahan_3d(const StencilConfig& cfg) {
    int    N           = cfg.nx;
    int    R           = cfg.stencil_reach;
    float  r           = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total       = (size_t)N * N * N;
    size_t cfp16_bytes = total * sizeof(cfp16_t);
    size_t float_bytes = total * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_3d16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_3d16,  &R,            sizeof(int)));

    // Scale into cfp16 range, encode, upload
    float scale  = cfp16_normalization_scale_3d(cfg);
    auto  h_f    = make_normalized_initial_grid_3d(cfg, scale);
    auto  h_data = float_to_cfp16_vec(h_f);
    std::vector<float> h_comp(total, 0.0f);   // compensation starts at zero

    cfp16_t *d_u, *d_u_next;
    float   *d_c, *d_c_next;
    CUDA_CHECK(cudaMalloc(&d_u,      cfp16_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, cfp16_bytes));
    CUDA_CHECK(cudaMalloc(&d_c,      float_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_next, float_bytes));
    CUDA_CHECK(cudaMemcpy(d_u,      h_data.data(), cfp16_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), cfp16_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c,      h_comp.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_next, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));

    dim3 block(BX, BY, BZ);
    dim3 grid3((N + BX - 1) / BX, (N + BY - 1) / BY, (N + BZ - 1) / BZ);
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    // Shared memory: two float arrays each of size tile_x * tile_y * tile_z
    int    tile_x     = BX + 2 * R;
    int    tile_y     = BY + 2 * R;
    int    tile_z     = BZ + 2 * R;
    size_t smem_bytes = 2 * (size_t)tile_x * tile_y * tile_z * sizeof(float);

    if (smem_bytes > 49152) {
        fprintf(stderr,
            "WARNING: shared memory (%zu B) may exceed 48 KB device limit for R=%d.\n",
            smem_bytes, R);
    }

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat3d_cfp16_kahan_tiled_kernel<<<grid3, block, smem_bytes>>>(
            d_u, d_u_next, d_c, d_c_next, N, r);
        apply_neumann_bc_3d_cfp16<<<bc_grid, bc_block>>>(d_u_next, N, R);
        apply_neumann_bc_3d_comp <<<bc_grid, bc_block>>>(d_c_next, N, R);
        cfp16_t* tmp_h = d_u; d_u = d_u_next; d_u_next = tmp_h;
        float*   tmp_c = d_c; d_c = d_c_next; d_c_next = tmp_c;
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    // Download, decode, add compensation, scale back to physical temperatures
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, cfp16_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> result_f(total);
    for (size_t i = 0; i < total; i++)
        result_f[i] = cfp16_to_float(h_data[i]) + h_comp[i];  // both in scaled space
    denormalize_result_3d(result_f, scale);

    // DRAM bandwidth: cfp16 grid r+w  +  float compensation grid r+w
    double bytes_per_step = 2.0 * (double)total * sizeof(cfp16_t)
                          + 2.0 * (double)total * sizeof(float);
    double bw = (bytes_per_step * cfg.timesteps) / (elapsed_ms * 1e-3) / 1e9;

    StencilResult res;
    res.variant_name     = "cuda_cfp16_kahan_3d";
    res.grid_size        = N;
    res.dim              = 3;
    res.stencil_reach    = R;
    res.timesteps        = cfg.timesteps;
    res.elapsed_ms       = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes     = 2 * cfp16_bytes + 2 * float_bytes;
    res.final_grid       = std::move(result_f);

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_c_next));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return res;
}

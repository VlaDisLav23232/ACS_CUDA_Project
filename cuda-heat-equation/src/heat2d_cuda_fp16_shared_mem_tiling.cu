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

__constant__ float d_coeffs_smem2d16[MAX_REACH + 1];
__constant__ int   d_reach_smem2d16;

/* ---- GPU auto-detection ---- */

struct TileConfig2D16 {
    int tile_x, tile_y;
    size_t smem_bytes;
};

static TileConfig2D16 query_tile_2d16(int R, size_t elem_bytes) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    size_t smem_avail = prop.sharedMemPerBlock;
    int max_threads   = prop.maxThreadsPerBlock;

    static const int cands[][2] = {
        {32,32},{32,16},{16,32},{32,8},{16,16},{16,8},{8,16},{8,8},{4,4}
    };
    int n = sizeof(cands) / sizeof(cands[0]);

    for (int i = 0; i < n; i++) {
        int tx = cands[i][0], ty = cands[i][1];
        if (tx * ty > max_threads) continue;
        size_t smem = (size_t)(tx + 2*R) * (ty + 2*R) * elem_bytes;
        if (smem <= smem_avail) {
            printf("  2D fp16 tile: %dx%d, smem: %zu bytes (%.1fKB)\n",
                   tx, ty, smem, smem / 1024.0);
            return {tx, ty, smem};
        }
    }
    size_t fb = (size_t)(4+2*R)*(4+2*R)*elem_bytes;
    return {4, 4, fb};
}

/* ---- fp16 naive shared memory kernel ---- */

__global__ void heat2d_fp16_naive_smem_kernel(const __half* __restrict__ u,
                                               __half* __restrict__ u_next,
                                               int N, float r,
                                               int tile_x, int tile_y) {
    int R = d_reach_smem2d16;
    int smem_w = tile_x + 2 * R;
    int smem_h = tile_y + 2 * R;

    extern __shared__ __half smem_h16[];

    int gi = blockIdx.x * tile_x + threadIdx.x;
    int gj = blockIdx.y * tile_y + threadIdx.y;
    int base_i = blockIdx.x * tile_x - R;
    int base_j = blockIdx.y * tile_y - R;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_sz = blockDim.x * blockDim.y;
    int total = smem_w * smem_h;

    for (int idx = tid; idx < total; idx += block_sz) {
        int sy = idx / smem_w;
        int sx = idx % smem_w;
        int ci = max(0, min(N - 1, base_i + sx));
        int cj = max(0, min(N - 1, base_j + sy));
        smem_h16[idx] = u[cj * N + ci];
    }
    __syncthreads();

    if (gi >= R && gi < N - R && gj >= R && gj < N - R) {
        int si = threadIdx.x + R;
        int sj = threadIdx.y + R;

        float center = __half2float(smem_h16[sj * smem_w + si]);
        float lap = 2.0f * d_coeffs_smem2d16[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_smem2d16[m] * (
                __half2float(smem_h16[sj * smem_w + (si - m)])
              + __half2float(smem_h16[sj * smem_w + (si + m)])
              + __half2float(smem_h16[(sj - m) * smem_w + si])
              + __half2float(smem_h16[(sj + m) * smem_w + si]));
        }
        u_next[gj * N + gi] = __float2half(center + r * lap);
    }
}

/* ---- fp16 Kahan shared memory kernel ---- */

__global__ void heat2d_fp16_kahan_smem_kernel(const __half* __restrict__ u,
                                               __half* __restrict__ u_next,
                                               float* __restrict__ c,
                                               float* __restrict__ c_next,
                                               int N, float r,
                                               int tile_x, int tile_y) {
    int R = d_reach_smem2d16;
    int smem_w = tile_x + 2 * R;
    int smem_h = tile_y + 2 * R;
    int n_elem = smem_w * smem_h;

    extern __shared__ char smem_raw[];
    __half* s_u = (__half*)smem_raw;
    /* align float array to 4-byte boundary */
    size_t half_bytes = (size_t)n_elem * sizeof(__half);
    size_t aligned    = (half_bytes + 3) & ~(size_t)3;
    float* s_c = (float*)(smem_raw + aligned);

    int gi = blockIdx.x * tile_x + threadIdx.x;
    int gj = blockIdx.y * tile_y + threadIdx.y;
    int base_i = blockIdx.x * tile_x - R;
    int base_j = blockIdx.y * tile_y - R;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_sz = blockDim.x * blockDim.y;

    /* cooperative loading of both half and compensation tiles */
    for (int idx = tid; idx < n_elem; idx += block_sz) {
        int sy = idx / smem_w;
        int sx = idx % smem_w;
        int ci = max(0, min(N - 1, base_i + sx));
        int cj = max(0, min(N - 1, base_j + sy));
        int gidx = cj * N + ci;
        s_u[idx] = u[gidx];
        s_c[idx] = c[gidx];
    }
    __syncthreads();

    if (gi >= R && gi < N - R && gj >= R && gj < N - R) {
        int si = threadIdx.x + R;
        int sj = threadIdx.y + R;
        int sidx = sj * smem_w + si;

        float center = __half2float(s_u[sidx]) + s_c[sidx];
        float lap = 2.0f * d_coeffs_smem2d16[0] * center;
        for (int m = 1; m <= R; m++) {
            int s_xm = sj * smem_w + (si - m);
            int s_xp = sj * smem_w + (si + m);
            int s_ym = (sj - m) * smem_w + si;
            int s_yp = (sj + m) * smem_w + si;
            float xm = __half2float(s_u[s_xm]) + s_c[s_xm];
            float xp = __half2float(s_u[s_xp]) + s_c[s_xp];
            float ym = __half2float(s_u[s_ym]) + s_c[s_ym];
            float yp = __half2float(s_u[s_yp]) + s_c[s_yp];
            lap += d_coeffs_smem2d16[m] * (xm + xp + ym + yp);
        }
        float exact_result = center + r * lap;
        __half stored = __float2half(exact_result);
        int gidx = gj * N + gi;
        u_next[gidx] = stored;
        volatile float stored_back = __half2float(stored);
        c_next[gidx] = exact_result - stored_back;
    }
}

/* ---- boundary condition kernels ---- */

__global__ void apply_neumann_bc_smem2d_fp16(__half* u, int N, int R) {
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

__global__ void apply_neumann_bc_smem2d_comp(float* c, int N, int R) {
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

/* ---- helpers ---- */

static std::vector<__half> float_to_half_smem(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); i++) h[i] = __float2half(f[i]);
    return h;
}

/* ---- fp16 naive host entry ---- */

StencilResult run_cuda_fp16_naive_smem(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t n_elems   = N * N;
    size_t half_bytes = n_elems * sizeof(__half);

    printf("  [smem] auto-detecting optimal tile...\n");
    TileConfig2D16 tile = query_tile_2d16(R, sizeof(__half));

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_smem2d16, cfg.fd_coeffs, (R+1)*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_smem2d16, &R, sizeof(int)));

    std::vector<float> h_f(n_elems, cfg.temp_initial);
    int src_size  = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++)
        for (int i = src_start; i < src_start + src_size; i++)
            h_f[j * N + i] = cfg.temp_source;

    auto h_data = float_to_half_smem(h_f);

    __half *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));

    dim3 block(tile.tile_x, tile.tile_y);
    dim3 grid((N + tile.tile_x - 1) / tile.tile_x,
              (N + tile.tile_y - 1) / tile.tile_y);
    int bc_threads = 256;
    int bc_blocks  = (N + bc_threads - 1) / bc_threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat2d_fp16_naive_smem_kernel<<<grid, block, tile.smem_bytes>>>(
            d_u, d_u_next, N, r, tile.tile_x, tile.tile_y);
        apply_neumann_bc_smem2d_fp16<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        __half* tmp = d_u; d_u = d_u_next; d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> result_f(n_elems);
    for (size_t i = 0; i < n_elems; i++) result_f[i] = __half2float(h_data[i]);

    int interior = N - 2 * R;
    double reads_pp = (2 * 2 * R + 1);
    double bps      = (double)interior * interior * (reads_pp + 1) * sizeof(__half);
    double bw       = bps * cfg.timesteps / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name    = "cuda_fp16_naive_smem";
    res.grid_size       = N;
    res.dim             = cfg.dim;
    res.stencil_reach   = R;
    res.timesteps       = cfg.timesteps;
    res.elapsed_ms      = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes    = 2 * half_bytes;
    res.final_grid      = result_f;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

/* ---- fp16 Kahan host entry ---- */

StencilResult run_cuda_fp16_kahan_smem(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t n_elems    = N * N;
    size_t half_bytes  = n_elems * sizeof(__half);
    size_t float_bytes = n_elems * sizeof(float);

    printf("  [smem] auto-detecting optimal tile...\n");
    /* Kahan needs both __half + float in shared mem: 6 bytes/elem */
    size_t smem_per_elem = sizeof(__half) + sizeof(float);
    TileConfig2D16 tile = query_tile_2d16(R, smem_per_elem);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_smem2d16, cfg.fd_coeffs, (R+1)*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_smem2d16, &R, sizeof(int)));

    std::vector<float> h_f(n_elems, cfg.temp_initial);
    int src_size  = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++)
        for (int i = src_start; i < src_start + src_size; i++)
            h_f[j * N + i] = cfg.temp_source;

    auto h_data = float_to_half_smem(h_f);
    std::vector<float> h_comp(n_elems, 0.0f);

    __half *d_u, *d_u_next;
    float  *d_c, *d_c_next;
    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_next, float_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_next, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));

    dim3 block(tile.tile_x, tile.tile_y);
    dim3 grid((N + tile.tile_x - 1) / tile.tile_x,
              (N + tile.tile_y - 1) / tile.tile_y);

    /* actual smem: half tile (aligned) + float tile */
    int n_tile = (tile.tile_x + 2*R) * (tile.tile_y + 2*R);
    size_t h_tile  = (size_t)n_tile * sizeof(__half);
    size_t h_align = (h_tile + 3) & ~(size_t)3;
    size_t smem_actual = h_align + (size_t)n_tile * sizeof(float);

    int bc_threads = 256;
    int bc_blocks  = (N + bc_threads - 1) / bc_threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat2d_fp16_kahan_smem_kernel<<<grid, block, smem_actual>>>(
            d_u, d_u_next, d_c, d_c_next, N, r, tile.tile_x, tile.tile_y);
        apply_neumann_bc_smem2d_fp16<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        apply_neumann_bc_smem2d_comp<<<bc_blocks, bc_threads>>>(d_c_next, N, R);
        __half* tmp_h = d_u; d_u = d_u_next; d_u_next = tmp_h;
        float*  tmp_c = d_c; d_c = d_c_next; d_c_next = tmp_c;
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

    int interior = N - 2 * R;
    double reads_pp = (2 * 2 * R + 1);
    double half_rw  = (reads_pp + 1) * sizeof(__half);
    double comp_rw  = (reads_pp + 1) * sizeof(float);
    double bps      = (double)interior * interior * (half_rw + comp_rw);
    double bw       = bps * cfg.timesteps / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name    = "cuda_fp16_kahan_smem";
    res.grid_size       = N;
    res.dim             = cfg.dim;
    res.stencil_reach   = R;
    res.timesteps       = cfg.timesteps;
    res.elapsed_ms      = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes    = 2 * half_bytes + 2 * float_bytes;
    res.final_grid      = result_f;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_c_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

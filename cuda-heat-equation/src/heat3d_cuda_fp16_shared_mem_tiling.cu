/*
 * 3D heat stencil — fp16 (naive + Kahan) with 2.5D shared memory tiling
 *
 * Tiles only the XY plane in shared memory; Z neighbors read from global
 * memory (served efficiently by L2 cache).  This avoids the cubic halo
 * blowup and keeps tiles large even for R=8.
 */

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

__constant__ float d_coeffs_smem3d16[MAX_REACH + 1];
__constant__ int   d_reach_smem3d16;

/* ---- GPU auto-detection: 2D tile for 2.5D tiling ---- */

struct TileConfig25D16 {
    int tile_x, tile_y;
    size_t smem_bytes;
};

static TileConfig25D16 query_tile_25d16(int R, size_t elem_bytes) {
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
            printf("  3D fp16 2.5D tile: %dx%d (XY), smem: %zu bytes (%.1fKB)\n",
                   tx, ty, smem, smem / 1024.0);
            return {tx, ty, smem};
        }
    }
    size_t fb = (size_t)(4+2*R)*(4+2*R)*elem_bytes;
    printf("  warning: fallback 4x4 tile\n");
    return {4, 4, fb};
}

/* ---- fp16 naive 2.5D kernel ---- */

__global__ void heat3d_fp16_naive_smem25d_kernel(const __half* __restrict__ u,
                                                  __half* __restrict__ u_next,
                                                  int N, float r,
                                                  int tile_x, int tile_y) {
    int R = d_reach_smem3d16;
    int sw = tile_x + 2 * R;
    int sh = tile_y + 2 * R;

    extern __shared__ __half smem_h[];

    int gx = blockIdx.x * tile_x + threadIdx.x;
    int gy = blockIdx.y * tile_y + threadIdx.y;
    int gz = blockIdx.z + R;

    int bx = blockIdx.x * tile_x - R;
    int by = blockIdx.y * tile_y - R;

    int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    int block_sz = blockDim.x * blockDim.y;
    int total    = sw * sh;

    for (int idx = tid; idx < total; idx += block_sz) {
        int sy = idx / sw;
        int sx = idx % sw;
        int ci = max(0, min(N-1, bx + sx));
        int cj = max(0, min(N-1, by + sy));
        smem_h[idx] = u[(size_t)gz * N * N + cj * N + ci];
    }
    __syncthreads();

    if (gx >= R && gx < N-R && gy >= R && gy < N-R) {
        int si = threadIdx.x + R;
        int sj = threadIdx.y + R;

        float center = __half2float(smem_h[sj * sw + si]);
        float lap = 3.0f * d_coeffs_smem3d16[0] * center;

        for (int m = 1; m <= R; m++) {
            /* XY from shared memory */
            float xy = __half2float(smem_h[sj * sw + (si-m)])
                     + __half2float(smem_h[sj * sw + (si+m)])
                     + __half2float(smem_h[(sj-m) * sw + si])
                     + __half2float(smem_h[(sj+m) * sw + si]);
            /* Z from global memory */
            float zn = __half2float(u[(size_t)(gz-m) * N*N + gy*N + gx])
                     + __half2float(u[(size_t)(gz+m) * N*N + gy*N + gx]);
            lap += d_coeffs_smem3d16[m] * (xy + zn);
        }
        u_next[(size_t)gz * N * N + gy * N + gx] = __float2half(center + r * lap);
    }
}

/* ---- fp16 Kahan 2.5D kernel ---- */

__global__ void heat3d_fp16_kahan_smem25d_kernel(const __half* __restrict__ u,
                                                  __half* __restrict__ u_next,
                                                  float* __restrict__ c,
                                                  float* __restrict__ c_next,
                                                  int N, float r,
                                                  int tile_x, int tile_y) {
    int R = d_reach_smem3d16;
    int sw = tile_x + 2 * R;
    int sh = tile_y + 2 * R;
    int n_elem = sw * sh;

    extern __shared__ char smem_raw[];
    __half* s_u = (__half*)smem_raw;
    size_t half_bytes = (size_t)n_elem * sizeof(__half);
    size_t aligned    = (half_bytes + 3) & ~(size_t)3;
    float* s_c = (float*)(smem_raw + aligned);

    int gx = blockIdx.x * tile_x + threadIdx.x;
    int gy = blockIdx.y * tile_y + threadIdx.y;
    int gz = blockIdx.z + R;

    int bx = blockIdx.x * tile_x - R;
    int by = blockIdx.y * tile_y - R;

    int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    int block_sz = blockDim.x * blockDim.y;

    /* cooperatively load 2D XY tiles for both u and c at this Z level */
    for (int idx = tid; idx < n_elem; idx += block_sz) {
        int sy = idx / sw;
        int sx = idx % sw;
        int ci = max(0, min(N-1, bx + sx));
        int cj = max(0, min(N-1, by + sy));
        size_t gidx = (size_t)gz * N * N + cj * N + ci;
        s_u[idx] = u[gidx];
        s_c[idx] = c[gidx];
    }
    __syncthreads();

    if (gx >= R && gx < N-R && gy >= R && gy < N-R) {
        int si = threadIdx.x + R;
        int sj = threadIdx.y + R;
        int sidx = sj * sw + si;

        float center = __half2float(s_u[sidx]) + s_c[sidx];
        float lap = 3.0f * d_coeffs_smem3d16[0] * center;

        for (int m = 1; m <= R; m++) {
            /* XY neighbors from shared memory */
            int s_xm = sj * sw + (si - m);
            int s_xp = sj * sw + (si + m);
            int s_ym = (sj - m) * sw + si;
            int s_yp = (sj + m) * sw + si;
            float xm = __half2float(s_u[s_xm]) + s_c[s_xm];
            float xp = __half2float(s_u[s_xp]) + s_c[s_xp];
            float ym = __half2float(s_u[s_ym]) + s_c[s_ym];
            float yp = __half2float(s_u[s_yp]) + s_c[s_yp];

            /* Z neighbors from global memory */
            size_t gidx_zm = (size_t)(gz-m) * N*N + gy*N + gx;
            size_t gidx_zp = (size_t)(gz+m) * N*N + gy*N + gx;
            float zm = __half2float(u[gidx_zm]) + c[gidx_zm];
            float zp = __half2float(u[gidx_zp]) + c[gidx_zp];

            lap += d_coeffs_smem3d16[m] * (xm + xp + ym + yp + zm + zp);
        }

        float exact_result = center + r * lap;
        __half stored = __float2half(exact_result);
        size_t gidx = (size_t)gz * N * N + gy * N + gx;
        u_next[gidx] = stored;
        volatile float stored_back = __half2float(stored);
        c_next[gidx] = exact_result - stored_back;
    }
}

/* ---- boundary condition kernels ---- */

__global__ void apply_neumann_bc_smem3d_fp16(__half* u, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;
    for (int b = R - 1; b >= 0; b--) {
        u[(size_t)a*N*N + b_idx*N + b]       = u[(size_t)a*N*N + b_idx*N + (b+1)];
        u[(size_t)a*N*N + b_idx*N + (N-1-b)] = u[(size_t)a*N*N + b_idx*N + (N-2-b)];
        u[(size_t)a*N*N + b*N + b_idx]       = u[(size_t)a*N*N + (b+1)*N + b_idx];
        u[(size_t)a*N*N + (N-1-b)*N + b_idx] = u[(size_t)a*N*N + (N-2-b)*N + b_idx];
        u[(size_t)b*N*N + a*N + b_idx]       = u[(size_t)(b+1)*N*N + a*N + b_idx];
        u[(size_t)(N-1-b)*N*N + a*N + b_idx] = u[(size_t)(N-2-b)*N*N + a*N + b_idx];
    }
}

__global__ void apply_neumann_bc_smem3d_comp(float* c, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;
    for (int b = R - 1; b >= 0; b--) {
        c[(size_t)a*N*N + b_idx*N + b]       = c[(size_t)a*N*N + b_idx*N + (b+1)];
        c[(size_t)a*N*N + b_idx*N + (N-1-b)] = c[(size_t)a*N*N + b_idx*N + (N-2-b)];
        c[(size_t)a*N*N + b*N + b_idx]       = c[(size_t)a*N*N + (b+1)*N + b_idx];
        c[(size_t)a*N*N + (N-1-b)*N + b_idx] = c[(size_t)a*N*N + (N-2-b)*N + b_idx];
        c[(size_t)b*N*N + a*N + b_idx]       = c[(size_t)(b+1)*N*N + a*N + b_idx];
        c[(size_t)(N-1-b)*N*N + a*N + b_idx] = c[(size_t)(N-2-b)*N*N + a*N + b_idx];
    }
}

/* ---- helpers ---- */

static std::vector<__half> float_to_half_vec(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); i++) h[i] = __float2half(f[i]);
    return h;
}

/* ---- fp16 naive 3D host entry ---- */

StencilResult run_cuda_fp16_naive_smem_3d(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total_pts  = (size_t)N * N * N;
    size_t half_bytes = total_pts * sizeof(__half);

    printf("  [smem 2.5D] auto-detecting optimal tile...\n");
    TileConfig25D16 tile = query_tile_25d16(R, sizeof(__half));

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_smem3d16, cfg.fd_coeffs, (R+1)*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_smem3d16, &R, sizeof(int)));

    std::vector<float> h_f(total_pts, cfg.temp_initial);
    int src_size  = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int z = src_start; z < src_start + src_size; z++)
        for (int y = src_start; y < src_start + src_size; y++)
            for (int x = src_start; x < src_start + src_size; x++)
                h_f[(size_t)z*N*N + y*N + x] = cfg.temp_source;

    auto h_data = float_to_half_vec(h_f);

    __half *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));

    int interior_z = N - 2 * R;
    if (interior_z <= 0) {
        fprintf(stderr, "error: grid too small for reach %d in 3D\n", R);
        exit(1);
    }

    dim3 block(tile.tile_x, tile.tile_y);
    dim3 grid3((N + tile.tile_x - 1) / tile.tile_x,
               (N + tile.tile_y - 1) / tile.tile_y,
               interior_z);
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat3d_fp16_naive_smem25d_kernel<<<grid3, block, tile.smem_bytes>>>(
            d_u, d_u_next, N, r, tile.tile_x, tile.tile_y);
        CUDA_CHECK(cudaGetLastError());
        apply_neumann_bc_smem3d_fp16<<<bc_grid, bc_block>>>(d_u_next, N, R);
        __half* tmp = d_u; d_u = d_u_next; d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> result_f(total_pts);
    for (size_t i = 0; i < total_pts; i++) result_f[i] = __half2float(h_data[i]);

    int interior = N - 2 * R;
    double reads_pp = (2 * 3 * R + 1);
    double bps      = (double)interior * interior * interior * (reads_pp + 1) * sizeof(__half);
    double bw       = bps * cfg.timesteps / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name    = "cuda_fp16_naive_smem_3d";
    res.grid_size       = N;
    res.dim             = 3;
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

/* ---- fp16 Kahan 3D host entry ---- */

StencilResult run_cuda_fp16_kahan_smem_3d(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total_pts   = (size_t)N * N * N;
    size_t half_bytes  = total_pts * sizeof(__half);
    size_t float_bytes = total_pts * sizeof(float);

    printf("  [smem 2.5D] auto-detecting optimal Kahan tile...\n");
    /* Kahan needs __half + float per element = 6 bytes */
    size_t smem_per_elem = sizeof(__half) + sizeof(float);
    TileConfig25D16 tile = query_tile_25d16(R, smem_per_elem);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_smem3d16, cfg.fd_coeffs, (R+1)*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_smem3d16, &R, sizeof(int)));

    std::vector<float> h_f(total_pts, cfg.temp_initial);
    int src_size  = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int z = src_start; z < src_start + src_size; z++)
        for (int y = src_start; y < src_start + src_size; y++)
            for (int x = src_start; x < src_start + src_size; x++)
                h_f[(size_t)z*N*N + y*N + x] = cfg.temp_source;

    auto h_data = float_to_half_vec(h_f);
    std::vector<float> h_comp(total_pts, 0.0f);

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

    int interior_z = N - 2 * R;
    if (interior_z <= 0) {
        fprintf(stderr, "error: grid too small for reach %d in 3D\n", R);
        exit(1);
    }

    dim3 block(tile.tile_x, tile.tile_y);
    dim3 grid3((N + tile.tile_x - 1) / tile.tile_x,
               (N + tile.tile_y - 1) / tile.tile_y,
               interior_z);

    /* compute actual shared memory: __half array (aligned) + float array */
    int n_tile = (tile.tile_x + 2*R) * (tile.tile_y + 2*R);
    size_t h_tile  = (size_t)n_tile * sizeof(__half);
    size_t h_align = (h_tile + 3) & ~(size_t)3;
    size_t smem_actual = h_align + (size_t)n_tile * sizeof(float);

    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat3d_fp16_kahan_smem25d_kernel<<<grid3, block, smem_actual>>>(
            d_u, d_u_next, d_c, d_c_next, N, r,
            tile.tile_x, tile.tile_y);
        CUDA_CHECK(cudaGetLastError());
        apply_neumann_bc_smem3d_fp16<<<bc_grid, bc_block>>>(d_u_next, N, R);
        apply_neumann_bc_smem3d_comp<<<bc_grid, bc_block>>>(d_c_next, N, R);
        __half* tmp_h = d_u; d_u = d_u_next; d_u_next = tmp_h;
        float*  tmp_c = d_c; d_c = d_c_next; d_c_next = tmp_c;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(total_pts);
    for (size_t i = 0; i < total_pts; i++)
        result_f[i] = __half2float(h_data[i]) + h_comp[i];

    int interior = N - 2 * R;
    double reads_pp = (2 * 3 * R + 1);
    double half_rw  = (reads_pp + 1) * sizeof(__half);
    double comp_rw  = (reads_pp + 1) * sizeof(float);
    double bps      = (double)interior * interior * interior * (half_rw + comp_rw);
    double bw       = bps * cfg.timesteps / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name    = "cuda_fp16_kahan_smem_3d";
    res.grid_size       = N;
    res.dim             = 3;
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

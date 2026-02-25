/*
 * 3D heat stencil — fp32 with 2.5D shared memory tiling
 *
 * Strategy: tile only XY dimensions in shared memory, read Z neighbors
 * from global memory (served by L2 cache). This avoids the cubic halo
 * blowup that makes full 3D tiling impractical for large stencil reaches.
 *
 * Halo ratio comparison for R=8:
 *   Full 3D tile 8x8x4:  interior=256, halo=11520, ratio=45:1
 *   2.5D tile 16x16:     interior=256, halo=1024,  ratio=4:1
 */

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

__constant__ float d_coeffs_smem3d[MAX_REACH + 1];
__constant__ int   d_reach_smem3d;

/* ---- GPU auto-detection: 2D tile for 2.5D tiling ---- */

struct TileConfig2D_3D {
    int tile_x, tile_y;
    size_t smem_bytes;
};

static TileConfig2D_3D query_optimal_tile_25d(int R, size_t elem_bytes) {
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
            printf("  GPU: %s, smem/block: %zuKB\n", prop.name, smem_avail/1024);
            printf("  3D fp32 2.5D tile: %dx%d (XY), smem: %zu bytes (%.1fKB)\n",
                   tx, ty, smem, smem / 1024.0);
            return {tx, ty, smem};
        }
    }
    size_t fb = (size_t)(4+2*R)*(4+2*R)*elem_bytes;
    printf("  warning: fallback 4x4 tile\n");
    return {4, 4, fb};
}

/* ---- 2.5D shared-memory stencil kernel ----
 * Each block handles one XY tile at one Z coordinate.
 * XY neighbors come from shared memory, Z neighbors from global memory. */

__global__ void heat3d_fp32_smem25d_kernel(const float* __restrict__ u,
                                           float* __restrict__ u_next,
                                           int N, float r,
                                           int tile_x, int tile_y) {
    int R = d_reach_smem3d;
    int sw = tile_x + 2 * R;
    int sh = tile_y + 2 * R;

    extern __shared__ float smem[];

    int gx = blockIdx.x * tile_x + threadIdx.x;
    int gy = blockIdx.y * tile_y + threadIdx.y;
    int gz = blockIdx.z + R;  /* only interior Z slices get blocks */

    /* load 2D XY tile at this Z level into shared memory */
    int bx = blockIdx.x * tile_x - R;
    int by = blockIdx.y * tile_y - R;

    int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    int block_sz = blockDim.x * blockDim.y;
    int total    = sw * sh;

    for (int idx = tid; idx < total; idx += block_sz) {
        int sy = idx / sw;
        int sx = idx % sw;
        int ci = max(0, min(N - 1, bx + sx));
        int cj = max(0, min(N - 1, by + sy));
        smem[idx] = u[(size_t)gz * N * N + cj * N + ci];
    }
    __syncthreads();

    if (gx >= R && gx < N - R && gy >= R && gy < N - R) {
        int si = threadIdx.x + R;
        int sj = threadIdx.y + R;

        float center = smem[sj * sw + si];
        float lap = 3.0f * d_coeffs_smem3d[0] * center;

        for (int m = 1; m <= R; m++) {
            /* XY neighbors from shared memory */
            float xy = smem[sj * sw + (si - m)] + smem[sj * sw + (si + m)]
                     + smem[(sj - m) * sw + si] + smem[(sj + m) * sw + si];
            /* Z neighbors from global memory (L2 cached) */
            float zn = u[(size_t)(gz - m) * N * N + gy * N + gx]
                     + u[(size_t)(gz + m) * N * N + gy * N + gx];
            lap += d_coeffs_smem3d[m] * (xy + zn);
        }
        u_next[(size_t)gz * N * N + gy * N + gx] = center + r * lap;
    }
}

/* ---- 3D Neumann BC (unchanged) ---- */

__global__ void apply_neumann_bc_smem3d(float* u, int N, int R) {
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

/* ---- host entry point ---- */

StencilResult run_cuda_fp32_smem_3d(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total_pts  = (size_t)N * N * N;
    size_t grid_bytes = total_pts * sizeof(float);

    printf("  [smem 2.5D] auto-detecting optimal tile...\n");
    TileConfig2D_3D tile = query_optimal_tile_25d(R, sizeof(float));

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_smem3d, cfg.fd_coeffs, (R+1)*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_smem3d, &R, sizeof(int)));

    std::vector<float> h_u(total_pts, cfg.temp_initial);
    int src_size  = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int z = src_start; z < src_start + src_size; z++)
        for (int y = src_start; y < src_start + src_size; y++)
            for (int x = src_start; x < src_start + src_size; x++)
                h_u[(size_t)z*N*N + y*N + x] = cfg.temp_source;

    float *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, grid_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));

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
        heat3d_fp32_smem25d_kernel<<<grid3, block, tile.smem_bytes>>>(
            d_u, d_u_next, N, r, tile.tile_x, tile.tile_y);
        CUDA_CHECK(cudaGetLastError());
        apply_neumann_bc_smem3d<<<bc_grid, bc_block>>>(d_u_next, N, R);
        float* tmp = d_u; d_u = d_u_next; d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, grid_bytes, cudaMemcpyDeviceToHost));

    int interior = N - 2 * R;
    double reads_pp = (2 * 3 * R + 1);
    double bps      = (double)interior * interior * interior * (reads_pp + 1) * sizeof(float);
    double bw       = bps * cfg.timesteps / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name    = "cuda_fp32_smem_3d";
    res.grid_size       = N;
    res.dim             = 3;
    res.stencil_reach   = R;
    res.timesteps       = cfg.timesteps;
    res.elapsed_ms      = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes    = 2 * grid_bytes;
    res.final_grid      = h_u;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

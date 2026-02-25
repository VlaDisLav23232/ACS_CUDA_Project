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

__constant__ float d_coeffs_smem2d[MAX_REACH + 1];
__constant__ int   d_reach_smem2d;

/* ---- GPU auto-detection: pick largest tile that fits in shared memory ---- */

struct TileConfig2D {
    int tile_x, tile_y;
    size_t smem_bytes;
};

static TileConfig2D query_optimal_tile_2d(int R, size_t elem_bytes) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    size_t smem_avail = prop.sharedMemPerBlock;
    int max_threads   = prop.maxThreadsPerBlock;

    /* candidates ordered by thread count (largest first) */
    static const int cands[][2] = {
        {32,32},{32,16},{16,32},{32,8},{16,16},{16,8},{8,16},{8,8},{4,4}
    };
    int n = sizeof(cands) / sizeof(cands[0]);

    for (int i = 0; i < n; i++) {
        int tx = cands[i][0], ty = cands[i][1];
        if (tx * ty > max_threads) continue;
        size_t smem = (size_t)(tx + 2*R) * (ty + 2*R) * elem_bytes;
        if (smem <= smem_avail) {
            printf("  GPU: %s, smem/block: %zuKB, warp: %d\n",
                   prop.name, smem_avail / 1024, prop.warpSize);
            printf("  2D fp32 tile: %dx%d, smem: %zu bytes (%.1fKB)\n",
                   tx, ty, smem, smem / 1024.0);
            return {tx, ty, smem};
        }
    }
    size_t fb = (size_t)(4+2*R)*(4+2*R)*elem_bytes;
    printf("  warning: fallback 4x4 tile, smem %zu bytes\n", fb);
    return {4, 4, fb};
}

/* ---- shared-memory stencil kernel ---- */

__global__ void heat2d_fp32_smem_kernel(const float* __restrict__ u,
                                        float* __restrict__ u_next,
                                        int N, float r,
                                        int tile_x, int tile_y) {
    int R = d_reach_smem2d;
    int smem_w = tile_x + 2 * R;
    int smem_h = tile_y + 2 * R;

    extern __shared__ float smem[];

    /* global coords of this thread's output point */
    int gi = blockIdx.x * tile_x + threadIdx.x;
    int gj = blockIdx.y * tile_y + threadIdx.y;

    /* top-left corner of the halo region in global coords */
    int base_i = blockIdx.x * tile_x - R;
    int base_j = blockIdx.y * tile_y - R;

    /* cooperative loading: every thread loads multiple elements */
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_sz = blockDim.x * blockDim.y;
    int total = smem_w * smem_h;

    for (int idx = tid; idx < total; idx += block_sz) {
        int sy = idx / smem_w;
        int sx = idx % smem_w;
        int ci = base_i + sx;
        int cj = base_j + sy;
        ci = max(0, min(N - 1, ci));
        cj = max(0, min(N - 1, cj));
        smem[idx] = u[cj * N + ci];
    }

    __syncthreads();

    if (gi >= R && gi < N - R && gj >= R && gj < N - R) {
        int si = threadIdx.x + R;
        int sj = threadIdx.y + R;

        float center = smem[sj * smem_w + si];
        float lap = 2.0f * d_coeffs_smem2d[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_coeffs_smem2d[m] * (
                smem[sj * smem_w + (si - m)] + smem[sj * smem_w + (si + m)]
              + smem[(sj - m) * smem_w + si] + smem[(sj + m) * smem_w + si]);
        }
        u_next[gj * N + gi] = center + r * lap;
    }
}

/* ---- Neumann boundary condition (unchanged logic) ---- */

__global__ void apply_neumann_bc_smem2d(float* u, int N, int R) {
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

/* ---- host entry point ---- */

StencilResult run_cuda_fp32_smem(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t grid_bytes = N * N * sizeof(float);

    printf("  [smem] auto-detecting optimal tile...\n");
    TileConfig2D tile = query_optimal_tile_2d(R, sizeof(float));

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_smem2d, cfg.fd_coeffs, (R+1)*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_smem2d, &R, sizeof(int)));

    std::vector<float> h_u(N * N, cfg.temp_initial);
    int src_size  = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++)
        for (int i = src_start; i < src_start + src_size; i++)
            h_u[j * N + i] = cfg.temp_source;

    float *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, grid_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));

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
        heat2d_fp32_smem_kernel<<<grid, block, tile.smem_bytes>>>(
            d_u, d_u_next, N, r, tile.tile_x, tile.tile_y);
        apply_neumann_bc_smem2d<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        float* tmp = d_u; d_u = d_u_next; d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, grid_bytes, cudaMemcpyDeviceToHost));

    int interior = N - 2 * R;
    double reads_pp   = (2 * 2 * R + 1);
    double bps        = (double)interior * interior * (reads_pp + 1) * sizeof(float);
    double total_b    = bps * cfg.timesteps;
    double bw         = total_b / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name    = "cuda_fp32_smem";
    res.grid_size       = N;
    res.dim             = cfg.dim;
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

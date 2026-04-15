/*
 * 3D heat stencil — fp32 with Z-sliding-window 2.5D shared memory tiling
 *
 * Strategy: tile the XY plane in shared memory and slide a register-based
 * circular buffer through Z.  For each output Z-layer the block:
 *   1. Cooperatively loads the XY tile (core + R-halo) into shared memory.
 *   2. Reads XY neighbors from shared memory (zero global traffic).
 *   3. Reads Z  neighbors from thread-private registers  (zero global traffic).
 *   4. Slides the register window forward by one Z position, issuing only
 *      ONE new global load (the value at gz+R+1).
 *
 * Z-chunking: the interior Z range is split into chunks so that multiple
 * blocks can work on different Z-ranges in parallel, maintaining GPU
 * occupancy.  Each block handles one XY tile × one Z-chunk.
 *
 * Global-memory reads comparison (per output point, R=8):
 *   Baseline (no smem) : 6R+1 = 49
 *   Old 2.5D (Z global): ~3 (amortised smem) + 2R = 19
 *   Sliding window     : ~3 (amortised smem) + 1  =  4   ← this kernel
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

/* ---- GPU auto-detection: 2D tile for 2.5D sliding-window tiling ---- */

struct TileConfig2D_3D {
    int tile_x, tile_y;
    int z_chunk;        /* number of Z-layers per block */
    size_t smem_bytes;
};

static TileConfig2D_3D query_optimal_tile_sliding(int R, int N, size_t elem_bytes) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    size_t smem_avail = prop.sharedMemPerBlock;
    int max_threads   = prop.maxThreadsPerBlock;
    int max_regs      = prop.regsPerBlock;
    int sm_count      = prop.multiProcessorCount;

    /* Each thread uses (2R+1) registers for the Z sliding buffer plus
       ~32 registers for the rest of the kernel. */
    int regs_per_thread = (2 * R + 1) + 32;

    static const int cands[][2] = {
        {32,32},{32,16},{16,32},{32,8},{16,16},{16,8},{8,16},{8,8},{4,4}
    };
    int n = sizeof(cands) / sizeof(cands[0]);

    for (int i = 0; i < n; i++) {
        int tx = cands[i][0], ty = cands[i][1];
        int nthreads = tx * ty;
        if (nthreads > max_threads) continue;

        size_t smem = (size_t)(tx + 2*R) * (ty + 2*R) * elem_bytes;
        if (smem > smem_avail) continue;

        /* Check register pressure — we need at least 1 block to fit */
        if (nthreads * regs_per_thread > max_regs) continue;

        /* Compute Z-chunk size: want enough blocks to fill the GPU.
           XY blocks = ceil(N/tx) * ceil(N/ty).
           Total blocks needed for good occupancy ≈ sm_count * 4.
           z_chunks = max(1, ceil(target_blocks / xy_blocks)). */
        int xy_blocks = ((N + tx - 1) / tx) * ((N + ty - 1) / ty);
        int interior_z = N - 2 * R;
        if (interior_z <= 0) continue;

        int target_blocks = sm_count * 4;
        int z_chunks = max(1, (target_blocks + xy_blocks - 1) / xy_blocks);
        z_chunks = min(z_chunks, interior_z);
        int z_chunk = (interior_z + z_chunks - 1) / z_chunks;

        printf("  GPU: %s, smem/block: %zuKB, SMs: %d\n", prop.name, smem_avail/1024, sm_count);
        printf("  3D fp32 sliding tile: %dx%d (XY), z_chunk: %d, smem: %zu bytes (%.1fKB)\n",
               tx, ty, z_chunk, smem, smem / 1024.0);
        printf("  Regs/thread: ~%d (z_buf=%d + base~32), threads=%d\n",
               regs_per_thread, 2*R+1, nthreads);
        printf("  Total blocks: %d (XY: %d × Z-chunks: %d)\n",
               xy_blocks * z_chunks, xy_blocks, z_chunks);
        return {tx, ty, z_chunk, smem};
    }
    size_t fb = (size_t)(4+2*R)*(4+2*R)*elem_bytes;
    printf("  warning: fallback 4x4 tile\n");
    return {4, 4, 1, fb};
}

/* ---- Z-sliding-window shared-memory stencil kernel ----
 *
 * Each block processes ONE XY tile across a CHUNK of interior Z layers.
 * blockIdx.z selects the Z-chunk.
 *
 * Per-thread state:
 *   z_buf[0..2R] — shift register holding u(gx,gy,gz-R..gz+R)
 *
 * Per Z iteration the block:
 *   1. Cooperatively loads XY tile (core + halo) at current gz into smem.
 *   2. __syncthreads()
 *   3. Computes stencil: XY from smem, Z from z_buf.
 *   4. __syncthreads() (before smem is overwritten by next Z iter)
 *   5. Slides z_buf: shift left by 1, load u(gx,gy,gz+R+1) into z_buf[2R].
 */

__global__ void heat3d_fp32_smem_sliding_kernel(const float* __restrict__ u,
                                                 float* __restrict__ u_next,
                                                 int N, float r,
                                                 int tile_x, int tile_y,
                                                 int z_chunk) {
    const int R = d_reach_smem3d;
    const int DIAM = 2 * R + 1;   /* number of Z-planes in buffer */
    const int sw = tile_x + 2 * R;
    const int sh = tile_y + 2 * R;

    extern __shared__ float smem[];

    /* Global XY coordinates for this thread */
    const int gx = blockIdx.x * tile_x + threadIdx.x;
    const int gy = blockIdx.y * tile_y + threadIdx.y;

    /* Base corner of halo region in global XY */
    const int bx = blockIdx.x * tile_x - R;
    const int by = blockIdx.y * tile_y - R;

    /* Z-range for this block's chunk */
    const int gz_first = R + blockIdx.z * z_chunk;
    const int gz_last  = min(N - R - 1, gz_first + z_chunk - 1);
    if (gz_first > gz_last) return;

    /* Thread index within block */
    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_sz = blockDim.x * blockDim.y;
    const int total_xy = sw * sh;

    /* Determine if this thread computes an interior output point.
       Threads in the halo still participate in cooperative smem loads
       but skip the stencil computation. */
    const bool active = (gx >= R && gx < N - R && gy >= R && gy < N - R);

    /* ---- Phase 1: Prefill Z sliding buffer ----
       Load u(gx, gy, z) for z = gz_first-R .. gz_first+R */
    float z_buf[2 * MAX_REACH + 1];   /* MAX_REACH == 8 → 17 floats */

    {
        int cx = (gx < N) ? gx : N - 1;
        int cy = (gy < N) ? gy : N - 1;
        for (int m = 0; m < DIAM; m++) {
            int gz = gz_first - R + m;
            gz = max(0, min(N - 1, gz));
            z_buf[m] = u[(size_t)gz * N * N + (size_t)cy * N + cx];
        }
    }

    /* ---- Phase 2: Slide through this block's Z chunk ---- */
    for (int gz = gz_first; gz <= gz_last; gz++) {
        /* 2a. Cooperatively load the XY tile at this Z level into smem */
        for (int idx = tid; idx < total_xy; idx += block_sz) {
            int sy = idx / sw;
            int sx = idx % sw;
            int ci = max(0, min(N - 1, bx + sx));
            int cj = max(0, min(N - 1, by + sy));
            smem[idx] = u[(size_t)gz * N * N + (size_t)cj * N + ci];
        }
        __syncthreads();

        /* 2b. Compute stencil if this thread owns an interior point */
        if (active) {
            int si = threadIdx.x + R;
            int sj = threadIdx.y + R;

            float center = z_buf[R];   /* == u(gx,gy,gz) */
            float lap = 3.0f * d_coeffs_smem3d[0] * center;

            for (int m = 1; m <= R; m++) {
                /* XY neighbors from shared memory */
                float xy = smem[sj * sw + (si - m)]
                         + smem[sj * sw + (si + m)]
                         + smem[(sj - m) * sw + si]
                         + smem[(sj + m) * sw + si];
                /* Z neighbors from register buffer */
                float zn = z_buf[R - m] + z_buf[R + m];

                lap += d_coeffs_smem3d[m] * (xy + zn);
            }
            u_next[(size_t)gz * N * N + (size_t)gy * N + gx] = center + r * lap;
        }

        __syncthreads();  /* ensure smem not overwritten before all threads done */

        /* 2c. Slide the Z buffer: shift left by 1, load next Z value */
        for (int m = 0; m < DIAM - 1; m++) {
            z_buf[m] = z_buf[m + 1];
        }
        {
            int next_z = gz + R + 1;
            next_z = max(0, min(N - 1, next_z));
            int cx = (gx < N) ? gx : N - 1;
            int cy = (gy < N) ? gy : N - 1;
            z_buf[DIAM - 1] = u[(size_t)next_z * N * N + (size_t)cy * N + cx];
        }
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

    printf("  [smem sliding] auto-detecting optimal tile...\n");
    TileConfig2D_3D tile = query_optimal_tile_sliding(R, N, sizeof(float));

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
    int z_chunks = (interior_z + tile.z_chunk - 1) / tile.z_chunk;

    dim3 block(tile.tile_x, tile.tile_y);
    dim3 grid3((N + tile.tile_x - 1) / tile.tile_x,
               (N + tile.tile_y - 1) / tile.tile_y,
               z_chunks);

    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat3d_fp32_smem_sliding_kernel<<<grid3, block, tile.smem_bytes>>>(
            d_u, d_u_next, N, r, tile.tile_x, tile.tile_y, tile.z_chunk);
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

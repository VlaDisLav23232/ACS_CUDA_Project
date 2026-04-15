/*
 * 3D heat stencil — fp16 (naive + Kahan) with Z-sliding-window 2.5D tiling
 *
 * Same sliding-window + Z-chunking strategy as the fp32 kernel:
 *   - XY tile (core + R-halo) loaded cooperatively into shared memory
 *   - Z neighbors held in a per-thread register circular buffer
 *   - Each block slides through a CHUNK of Z-layers (blockIdx.z selects chunk)
 *   - Only 1 new global read per thread per Z iteration (amortised)
 *
 * For fp16 naive:  smem stores __half, z_buf stores float (promoted from __half)
 * For fp16 Kahan:  smem stores __half + float (compensation), z_buf stores float
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

/* ---- GPU auto-detection: 2D tile for sliding-window tiling ---- */

struct TileConfig25D16 {
    int tile_x, tile_y;
    int z_chunk;
    size_t smem_bytes;
};

static TileConfig25D16 query_tile_sliding16(int R, int N, size_t elem_bytes) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    size_t smem_avail = prop.sharedMemPerBlock;
    int max_threads   = prop.maxThreadsPerBlock;
    int max_regs      = prop.regsPerBlock;
    int sm_count      = prop.multiProcessorCount;

    /* Register estimate: (2R+1) for z_buf + ~32 base.
       For Kahan we also keep compensation in registers — add (2R+1). */
    int regs_per_thread = 2 * (2 * R + 1) + 32;  /* conservative for Kahan */

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

        if (nthreads * regs_per_thread > max_regs) continue;

        /* Compute Z-chunk for GPU occupancy */
        int xy_blocks = ((N + tx - 1) / tx) * ((N + ty - 1) / ty);
        int interior_z = N - 2 * R;
        if (interior_z <= 0) continue;

        int target_blocks = sm_count * 4;
        int z_chunks = max(1, (target_blocks + xy_blocks - 1) / xy_blocks);
        z_chunks = min(z_chunks, interior_z);
        int z_chunk = (interior_z + z_chunks - 1) / z_chunks;

        printf("  3D fp16 sliding tile: %dx%d (XY), z_chunk: %d, smem: %zu bytes (%.1fKB)\n",
               tx, ty, z_chunk, smem, smem / 1024.0);
        return {tx, ty, z_chunk, smem};
    }
    size_t fb = (size_t)(4+2*R)*(4+2*R)*elem_bytes;
    printf("  warning: fallback 4x4 tile\n");
    return {4, 4, 1, fb};
}

/* ---- fp16 naive Z-sliding kernel with Z-chunking ---- */

__global__ void heat3d_fp16_naive_sliding_kernel(const __half* __restrict__ u,
                                                  __half* __restrict__ u_next,
                                                  int N, float r,
                                                  int tile_x, int tile_y,
                                                  int z_chunk) {
    const int R = d_reach_smem3d16;
    const int DIAM = 2 * R + 1;
    const int sw = tile_x + 2 * R;
    const int sh = tile_y + 2 * R;

    extern __shared__ __half smem_h[];

    const int gx = blockIdx.x * tile_x + threadIdx.x;
    const int gy = blockIdx.y * tile_y + threadIdx.y;

    const int bx = blockIdx.x * tile_x - R;
    const int by = blockIdx.y * tile_y - R;

    /* Z-range for this block's chunk */
    const int gz_first = R + blockIdx.z * z_chunk;
    const int gz_last  = min(N - R - 1, gz_first + z_chunk - 1);
    if (gz_first > gz_last) return;

    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_sz = blockDim.x * blockDim.y;
    const int total_xy = sw * sh;

    const bool active = (gx >= R && gx < N - R && gy >= R && gy < N - R);

    /* ---- Phase 1: Prefill Z sliding buffer ---- */
    float z_buf[2 * MAX_REACH + 1];

    {
        int cx = (gx < N) ? gx : N - 1;
        int cy = (gy < N) ? gy : N - 1;
        for (int m = 0; m < DIAM; m++) {
            int gz = gz_first - R + m;
            gz = max(0, min(N - 1, gz));
            z_buf[m] = __half2float(u[(size_t)gz * N * N + (size_t)cy * N + cx]);
        }
    }

    /* ---- Phase 2: Slide through this block's Z chunk ---- */
    for (int gz = gz_first; gz <= gz_last; gz++) {
        /* Load XY tile at this Z into shared memory */
        for (int idx = tid; idx < total_xy; idx += block_sz) {
            int sy = idx / sw;
            int sx = idx % sw;
            int ci = max(0, min(N-1, bx + sx));
            int cj = max(0, min(N-1, by + sy));
            smem_h[idx] = u[(size_t)gz * N * N + (size_t)cj * N + ci];
        }
        __syncthreads();

        if (active) {
            int si = threadIdx.x + R;
            int sj = threadIdx.y + R;

            float center = z_buf[R];
            float lap = 3.0f * d_coeffs_smem3d16[0] * center;

            for (int m = 1; m <= R; m++) {
                float xy = __half2float(smem_h[sj * sw + (si-m)])
                         + __half2float(smem_h[sj * sw + (si+m)])
                         + __half2float(smem_h[(sj-m) * sw + si])
                         + __half2float(smem_h[(sj+m) * sw + si]);
                float zn = z_buf[R - m] + z_buf[R + m];
                lap += d_coeffs_smem3d16[m] * (xy + zn);
            }
            u_next[(size_t)gz * N * N + (size_t)gy * N + gx] = __float2half(center + r * lap);
        }

        __syncthreads();

        /* Slide the Z buffer */
        for (int m = 0; m < DIAM - 1; m++) {
            z_buf[m] = z_buf[m + 1];
        }
        {
            int next_z = max(0, min(N - 1, gz + R + 1));
            int cx = (gx < N) ? gx : N - 1;
            int cy = (gy < N) ? gy : N - 1;
            z_buf[DIAM - 1] = __half2float(u[(size_t)next_z * N * N + (size_t)cy * N + cx]);
        }
    }
}

/* ---- fp16 Kahan Z-sliding kernel with Z-chunking ----
 *
 * Kahan compensation requires maintaining both __half u and float c arrays.
 * The Z sliding buffer holds reconstructed values (half + compensation).
 * Shared memory holds the XY tile for both u (half) and c (float).
 */

__global__ void heat3d_fp16_kahan_sliding_kernel(const __half* __restrict__ u,
                                                  __half* __restrict__ u_next,
                                                  float* __restrict__ c,
                                                  float* __restrict__ c_next,
                                                  int N, float r,
                                                  int tile_x, int tile_y,
                                                  int z_chunk) {
    const int R = d_reach_smem3d16;
    const int DIAM = 2 * R + 1;
    const int sw = tile_x + 2 * R;
    const int sh = tile_y + 2 * R;
    const int n_elem = sw * sh;

    /* Shared memory layout: __half[n_elem] (aligned) + float[n_elem] */
    extern __shared__ char smem_raw[];
    __half* s_u = (__half*)smem_raw;
    size_t half_bytes = (size_t)n_elem * sizeof(__half);
    size_t aligned    = (half_bytes + 3) & ~(size_t)3;
    float* s_c = (float*)(smem_raw + aligned);

    const int gx = blockIdx.x * tile_x + threadIdx.x;
    const int gy = blockIdx.y * tile_y + threadIdx.y;

    const int bx = blockIdx.x * tile_x - R;
    const int by = blockIdx.y * tile_y - R;

    /* Z-range for this block's chunk */
    const int gz_first = R + blockIdx.z * z_chunk;
    const int gz_last  = min(N - R - 1, gz_first + z_chunk - 1);
    if (gz_first > gz_last) return;

    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_sz = blockDim.x * blockDim.y;

    const bool active = (gx >= R && gx < N - R && gy >= R && gy < N - R);

    /* ---- Phase 1: Prefill Z sliding buffer with reconstructed values ---- */
    float z_buf[2 * MAX_REACH + 1];

    {
        int cx = (gx < N) ? gx : N - 1;
        int cy = (gy < N) ? gy : N - 1;
        for (int m = 0; m < DIAM; m++) {
            int gz = gz_first - R + m;
            gz = max(0, min(N - 1, gz));
            size_t gidx = (size_t)gz * N * N + (size_t)cy * N + cx;
            z_buf[m] = __half2float(u[gidx]) + c[gidx];
        }
    }

    /* ---- Phase 2: Slide through this block's Z chunk ---- */
    for (int gz = gz_first; gz <= gz_last; gz++) {
        /* Load XY tile (u and c) at this Z into shared memory */
        for (int idx = tid; idx < n_elem; idx += block_sz) {
            int sy = idx / sw;
            int sx = idx % sw;
            int ci = max(0, min(N-1, bx + sx));
            int cj = max(0, min(N-1, by + sy));
            size_t gidx = (size_t)gz * N * N + (size_t)cj * N + ci;
            s_u[idx] = u[gidx];
            s_c[idx] = c[gidx];
        }
        __syncthreads();

        if (active) {
            int si = threadIdx.x + R;
            int sj = threadIdx.y + R;

            float center = z_buf[R];
            float lap = 3.0f * d_coeffs_smem3d16[0] * center;

            for (int m = 1; m <= R; m++) {
                int s_xm = sj * sw + (si - m);
                int s_xp = sj * sw + (si + m);
                int s_ym = (sj - m) * sw + si;
                int s_yp = (sj + m) * sw + si;
                float xm = __half2float(s_u[s_xm]) + s_c[s_xm];
                float xp = __half2float(s_u[s_xp]) + s_c[s_xp];
                float ym = __half2float(s_u[s_ym]) + s_c[s_ym];
                float yp = __half2float(s_u[s_yp]) + s_c[s_yp];

                float zn = z_buf[R - m] + z_buf[R + m];

                lap += d_coeffs_smem3d16[m] * (xm + xp + ym + yp + zn);
            }

            float exact_result = center + r * lap;
            __half stored = __float2half(exact_result);
            size_t gidx = (size_t)gz * N * N + (size_t)gy * N + gx;
            u_next[gidx] = stored;
            volatile float stored_back = __half2float(stored);
            c_next[gidx] = exact_result - stored_back;
        }

        __syncthreads();

        /* Slide the Z buffer */
        for (int m = 0; m < DIAM - 1; m++) {
            z_buf[m] = z_buf[m + 1];
        }
        {
            int next_z = max(0, min(N - 1, gz + R + 1));
            int cx = (gx < N) ? gx : N - 1;
            int cy = (gy < N) ? gy : N - 1;
            size_t gidx = (size_t)next_z * N * N + (size_t)cy * N + cx;
            z_buf[DIAM - 1] = __half2float(u[gidx]) + c[gidx];
        }
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

    printf("  [smem sliding] auto-detecting optimal tile...\n");
    TileConfig25D16 tile = query_tile_sliding16(R, N, sizeof(__half));

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
        heat3d_fp16_naive_sliding_kernel<<<grid3, block, tile.smem_bytes>>>(
            d_u, d_u_next, N, r, tile.tile_x, tile.tile_y, tile.z_chunk);
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

    printf("  [smem sliding] auto-detecting optimal Kahan tile...\n");
    /* Kahan needs __half + float per element = 6 bytes */
    size_t smem_per_elem = sizeof(__half) + sizeof(float);
    TileConfig25D16 tile = query_tile_sliding16(R, N, smem_per_elem);

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
    int z_chunks = (interior_z + tile.z_chunk - 1) / tile.z_chunk;

    dim3 block(tile.tile_x, tile.tile_y);
    dim3 grid3((N + tile.tile_x - 1) / tile.tile_x,
               (N + tile.tile_y - 1) / tile.tile_y,
               z_chunks);

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
        heat3d_fp16_kahan_sliding_kernel<<<grid3, block, smem_actual>>>(
            d_u, d_u_next, d_c, d_c_next, N, r,
            tile.tile_x, tile.tile_y, tile.z_chunk);
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

// =============================================================================
// tensor_core_stencil.cu — 2D heat stencil via WMMA Tensor Cores
//
// Reformulates the cross-shaped FTCS stencil as a batched dot-product computed
// through matrix multiply-accumulate (D = A × B + C) using nvcuda::wmma.
//
// Each warp processes a 16×16 tile of output grid points by iterating over the
// 16 rows of the tile.  For each row of 16 output points:
//   - Matrix A (16 × K_PAD, row-major) holds the stencil coefficients
//     (identical rows).
//   - Matrix B (K_PAD × 16, row-major) holds the im2col-unrolled neighbor
//     values with each column corresponding to one output point.
//   - D = A × B is 16×16; the diagonal D[p][p] contains the weighted Laplacian
//     for output point p.
//
// Input/output grids are stored in __half; the MMA accumulates in float.
// =============================================================================

#include "stencil.h"
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ---- error-checking macro (matches project convention) ----------------------
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---- WMMA tile dimensions ---------------------------------------------------
static const int WMMA_M = 16;
static const int WMMA_N = 16;
static const int WMMA_K = 16;

// WMMA requires 128-bit (16 B) alignment for half, 256-bit (32 B) for float.
// We use 32 B universally.
static const size_t SMEM_ALIGN = 32;

__device__ __host__ inline size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

// ---- stencil coefficients in constant memory --------------------------------
__constant__ float d_coeffs_tc[MAX_REACH + 1];
__constant__ int   d_reach_tc;

// ---- Neumann BC kernel (duplicated from heat2d_cuda_fp16.cu to avoid linker
//      issues with __global__ functions across translation units) -------------
__global__ void apply_neumann_bc_fp16_tc(__half* u, int N, int R) {
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

// ---- helper: float-to-half conversion on CPU --------------------------------
static std::vector<__half> float_to_half_tc(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); i++)
        h[i] = __float2half(f[i]);
    return h;
}

// =============================================================================
// Kernel: heat2d_fp16_tensor_core_kernel
//
// Grid launch: blockDim = (32, 1)   — one warp per block
//              gridDim  = (N/16, N/16)
//
// For stencil reach R:
//   S     = 4*R + 1   (number of stencil terms in the cross)
//   K_PAD = ceil(S/16)*16
//
// The kernel iterates over the 16 rows (j_local) of its 16×16 output tile.
// For each row it populates B with the unrolled neighbors of 16 points, runs
// one (or more) WMMA MMA calls, and extracts the diagonal of D.
// =============================================================================
__global__ void heat2d_fp16_tensor_core_kernel(
        const __half* __restrict__ u,
        __half*       __restrict__ u_next,
        int N, float r)
{
    const int R = d_reach_tc;
    const int S = 4 * R + 1;                        // stencil terms
    const int K_PAD = ((S + WMMA_K - 1) / WMMA_K) * WMMA_K;

    // tile origin in global grid (top-left of 16×16 output tile)
    const int tile_col = blockIdx.x * WMMA_M;       // i-direction
    const int tile_row = blockIdx.y * WMMA_N;       // j-direction

    // skip tiles whose halos fall outside the interior [R, N-R)
    if (tile_col < R || tile_row < R ||
        tile_col + WMMA_M > N - R || tile_row + WMMA_N > N - R)
        return;

    const int lane = threadIdx.x;                   // 0..31

    // ---- Shared memory layout (aligned for WMMA) ----------------------------
    //   halo:  (16+2R) × (16+2R)  __half        (no alignment needed)
    //   A:     16 × K_PAD         __half         (32-B aligned)
    //   B:     K_PAD × 16         __half         (32-B aligned)
    //   D:     16 × 16            float          (32-B aligned)
    extern __shared__ char smem_raw[];

    const int halo_dim  = WMMA_M + 2 * R;
    const int halo_size = halo_dim * halo_dim;

    size_t off = 0;
    __half* s_halo = reinterpret_cast<__half*>(smem_raw + off);
    off  = align_up(off + halo_size * sizeof(__half),   SMEM_ALIGN);
    __half* s_A    = reinterpret_cast<__half*>(smem_raw + off);
    off  = align_up(off + WMMA_M * K_PAD * sizeof(__half), SMEM_ALIGN);
    __half* s_B    = reinterpret_cast<__half*>(smem_raw + off);
    off  = align_up(off + K_PAD * WMMA_M * sizeof(__half), SMEM_ALIGN);
    float*  s_D    = reinterpret_cast<float*>(smem_raw + off);

    // ---- 1) Cooperative halo load from global memory ------------------------
    const int g_halo_row0 = tile_row - R;
    const int g_halo_col0 = tile_col - R;

    for (int idx = lane; idx < halo_size; idx += 32) {
        int lj = idx / halo_dim;
        int li = idx % halo_dim;
        int gj = g_halo_row0 + lj;
        int gi = g_halo_col0 + li;
        // clamp (BCs should already be set, but guard for safety)
        gj = max(0, min(gj, N - 1));
        gi = max(0, min(gi, N - 1));
        s_halo[lj * halo_dim + li] = u[gj * N + gi];
    }
    __syncwarp();

    // ---- 2) Build coefficient matrix A (row-major, 16 × K_PAD) --------------
    // Every row is identical:
    //   k=0        : 2*c[0]   (the center weight, factor-of-2 from lap formula)
    //   k=1..4R    : for m = (k-1)/4 + 1, weight = c[m]
    //                direction order per ring: left, right, up, down
    //   k=S..K_PAD : 0  (zero padding)
    for (int k = lane; k < K_PAD; k += 32) {
        __half val;
        if (k == 0) {
            val = __float2half(2.0f * d_coeffs_tc[0]);
        } else if (k < S) {
            int m = (k - 1) / 4 + 1;
            val = __float2half(d_coeffs_tc[m]);
        } else {
            val = __float2half(0.0f);
        }
        // replicate into all 16 rows
        for (int row = 0; row < WMMA_M; row++) {
            s_A[row * K_PAD + k] = val;
        }
    }
    __syncwarp();

    // ---- 3) Iterate over the 16 rows of the output tile ---------------------
    for (int j_local = 0; j_local < WMMA_N; j_local++) {
        const int hj = j_local + R;   // halo-space row

        // ---- 3a) Fill B (K_PAD × 16, row-major) ----------------------------
        // B[k][col] = s_B[k * 16 + col]
        // col ∈ [0,15] = i_local within the tile
        for (int idx = lane; idx < K_PAD * WMMA_M; idx += 32) {
            int k   = idx / WMMA_M;
            int col = idx % WMMA_M;
            int hi  = col + R;         // halo-space column

            __half val;
            if (k == 0) {
                val = s_halo[hj * halo_dim + hi];
            } else if (k < S) {
                int m   = (k - 1) / 4 + 1;
                int dir = (k - 1) % 4;
                switch (dir) {
                    case 0: val = s_halo[hj       * halo_dim + (hi - m)]; break; // left
                    case 1: val = s_halo[hj       * halo_dim + (hi + m)]; break; // right
                    case 2: val = s_halo[(hj - m) * halo_dim +  hi     ]; break; // up
                    case 3: val = s_halo[(hj + m) * halo_dim +  hi     ]; break; // down
                    default: val = __float2half(0.0f);
                }
            } else {
                val = __float2half(0.0f);
            }
            s_B[k * WMMA_M + col] = val;
        }
        __syncwarp();

        // ---- 3b) WMMA MMA over K-dimension chunks --------------------------
        // D (16×16) = A (16×K_PAD) × B (K_PAD×16),  accumulated in float.
        // We interpret:
        //   frag_A = row_major, loaded from s_A with ldm = K_PAD
        //   frag_B = row_major, loaded from s_B with ldm = WMMA_M (=16)
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_D;
        wmma::fill_fragment(frag_D, 0.0f);

        for (int kk = 0; kk < K_PAD; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_A;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_B;

            // A slice: rows [0..15], cols [kk .. kk+15]
            wmma::load_matrix_sync(frag_A, s_A + kk, K_PAD);

            // B slice: rows [kk .. kk+15], cols [0..15]
            // s_B layout: s_B[row * 16 + col],  ldm = 16
            wmma::load_matrix_sync(frag_B, s_B + kk * WMMA_M, WMMA_M);

            wmma::mma_sync(frag_D, frag_A, frag_B, frag_D);
        }

        // ---- 3c) Store D and extract diagonal -------------------------------
        wmma::store_matrix_sync(s_D, frag_D, WMMA_M, wmma::mem_row_major);
        __syncwarp();

        // D[p][p] is the Laplacian for output point (j_local, p)
        // Only 16 values needed; threads 0..15 each handle one point.
        if (lane < WMMA_M) {
            float lap = s_D[lane * WMMA_M + lane];  // D[lane][lane]
            int gi    = tile_col + lane;
            int gj    = tile_row + j_local;
            int g_idx = gj * N + gi;
            float center = __half2float(u[g_idx]);
            u_next[g_idx] = __float2half(center + r * lap);
        }
        __syncwarp();
    }
}

// =============================================================================
// Host launcher: run_cuda_fp16_tensor_core
// Follows the exact pattern of run_cuda_fp16_kahan from heat2d_cuda_fp16.cu
// =============================================================================
StencilResult run_cuda_fp16_tensor_core(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t n_elems = (size_t)N * N;
    size_t half_bytes = n_elems * sizeof(__half);

    assert(N % WMMA_M == 0 && "Grid size N must be a multiple of 16 for WMMA");

    // upload FD coefficients to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_tc, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_tc, &R, sizeof(int)));

    // initialise host grid (identical to other 2D variants)
    std::vector<float> h_f(n_elems, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++)
        for (int i = src_start; i < src_start + src_size; i++)
            h_f[j * N + i] = cfg.temp_source;

    auto h_data = float_to_half_tc(h_f);

    // device allocations
    __half *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));

    // launch configuration: one warp (32 threads) per block, one tile per block
    int tiles_x = N / WMMA_M;
    int tiles_y = N / WMMA_N;
    dim3 grid_dim(tiles_x, tiles_y);
    dim3 block_dim(32, 1);

    // compute dynamic shared memory requirement (with alignment padding)
    int S = 4 * R + 1;
    int K_PAD = ((S + WMMA_K - 1) / WMMA_K) * WMMA_K;
    int halo_dim = WMMA_M + 2 * R;
    size_t smem_bytes = 0;
    smem_bytes  = align_up(smem_bytes + halo_dim * halo_dim * sizeof(__half), SMEM_ALIGN);
    smem_bytes  = align_up(smem_bytes + WMMA_M * K_PAD      * sizeof(__half), SMEM_ALIGN);
    smem_bytes  = align_up(smem_bytes + K_PAD  * WMMA_M     * sizeof(__half), SMEM_ALIGN);
    smem_bytes += WMMA_M * WMMA_N * sizeof(float);

    // check shared memory limit
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (smem_bytes > prop.sharedMemPerBlock) {
        fprintf(stderr, "Tensor Core kernel needs %zu B shared memory, device max = %zu B\n",
                smem_bytes, prop.sharedMemPerBlock);
        StencilResult res;
        res.variant_name = "cuda_fp16_tensor_core";
        res.grid_size = N;
        res.dim = cfg.dim;
        res.stencil_reach = R;
        res.timesteps = cfg.timesteps;
        res.elapsed_ms = -1.0;
        res.effective_bw_gbs = 0.0;
        res.megapoints_per_sec = 0.0;
        res.memory_bytes = 0;
        CUDA_CHECK(cudaFree(d_u));
        CUDA_CHECK(cudaFree(d_u_next));
        return res;
    }

    // boundary-condition launch config
    int bc_threads = 256;
    int bc_blocks = (N + bc_threads - 1) / bc_threads;

    // timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat2d_fp16_tensor_core_kernel<<<grid_dim, block_dim, smem_bytes>>>(
            d_u, d_u_next, N, r);
        apply_neumann_bc_fp16_tc<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        __half* tmp = d_u; d_u = d_u_next; d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // copy results back
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(n_elems);
    for (size_t i = 0; i < n_elems; i++)
        result_f[i] = __half2float(h_data[i]);

    // effective bandwidth: each grid point read + written per timestep
    double bytes_per_step = 2.0 * (double)N * N * sizeof(__half);
    double total_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;
    double mpts = ((double)N * N * cfg.timesteps) / (elapsed_ms / 1000.0) / 1e6;

    StencilResult res;
    res.variant_name = "cuda_fp16_tensor_core";
    res.grid_size = N;
    res.dim = cfg.dim;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * half_bytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

// 2D wave stencil via WMMA Tensor Cores.
//
// The cross-shaped FTCS laplacian is reformulated as a batched dot-product
// computed through matrix multiply-accumulate (D = A × B + C) using
// nvcuda::wmma. Each warp processes a 16×16 output tile by iterating over the
// 16 rows of the tile.
//   - A (16 × K_PAD, row-major) holds replicated stencil coefficients.
//   - B (K_PAD × 16, row-major) holds the im2col-unrolled neighbours,
//     one column per output point.
//   - D = A × B is 16×16; D[p][p] is the laplacian at output point p.
//
// Wave update (no Kahan):  u_next = 2*u - u_prev + s*lap(u).
// First step (t=0):        u_1   =   u_0       + 0.5*s*lap(u_0).

#include "stencil.h"
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static const int WMMA_M = 16;
static const int WMMA_N = 16;
static const int WMMA_K = 16;
static const size_t SMEM_ALIGN = 32;

__device__ __host__ inline size_t align_up_tc(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

__constant__ float d_coeffs_tc[MAX_REACH + 1];
__constant__ int   d_reach_tc;

__global__ void apply_dirichlet_fp16_tc(__half* u, int N, int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        __half z = __float2half(0.0f);
        for (int b = 0; b < R; b++) {
            u[b * N + idx] = z;
            u[(N - 1 - b) * N + idx] = z;
            u[idx * N + b] = z;
            u[idx * N + (N - 1 - b)] = z;
        }
    }
}

static std::vector<__half> float_to_half_tc(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); i++) h[i] = __float2half(f[i]);
    return h;
}

static void init_wave_field_2d_tc(std::vector<float>& u, int N, float amp, float sigma) {
    const float s2 = sigma * sigma;
    const float src[2][2] = {{-0.35f, -0.35f}, {0.35f, 0.35f}};
    for (int j = 0; j < N; j++) {
        float y = -1.0f + 2.0f * j / (N - 1.0f);
        for (int i = 0; i < N; i++) {
            float x = -1.0f + 2.0f * i / (N - 1.0f);
            float v = 0.0f;
            for (int k = 0; k < 2; k++) {
                float dx = x - src[k][0];
                float dy = y - src[k][1];
                v += amp * expf(-(dx * dx + dy * dy) / s2);
            }
            u[j * N + i] = v;
        }
    }
}


// ==========================================================================
// MMA core: load halo, build A & B, run wmma::mma_sync, return D[p][p] (lap).
// Only the time-integration step differs between first-step and main update.
// ==========================================================================
__global__ void wave2d_fp16_tensor_core_first_step_kernel(
        const __half* __restrict__ u0,
        __half*       __restrict__ u1,
        int N, float s)
{
    const int R = d_reach_tc;
    const int S = 4 * R + 1;
    const int K_PAD = ((S + WMMA_K - 1) / WMMA_K) * WMMA_K;

    const int tile_col = blockIdx.x * WMMA_M;
    const int tile_row = blockIdx.y * WMMA_N;
    const int lane = threadIdx.x;

    // Boundary tiles (those whose halo overlaps the BC zone) can't use WMMA
    // safely because we'd read past the array edge. Fall back to a scalar
    // inner-loop for the in-range interior points within the tile.
    if (tile_col < R || tile_row < R ||
        tile_col + WMMA_M > N - R || tile_row + WMMA_N > N - R) {
        for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
            int li = idx % WMMA_M;
            int lj = idx / WMMA_M;
            int gi = tile_col + li;
            int gj = tile_row + lj;
            if (gi >= R && gi < N - R && gj >= R && gj < N - R) {
                float center = __half2float(u0[gj * N + gi]);
                float lap = 2.0f * d_coeffs_tc[0] * center;
                for (int m = 1; m <= R; m++) {
                    lap += d_coeffs_tc[m] * (
                        __half2float(u0[gj * N + (gi - m)]) + __half2float(u0[gj * N + (gi + m)])
                      + __half2float(u0[(gj - m) * N + gi]) + __half2float(u0[(gj + m) * N + gi]));
                }
                u1[gj * N + gi] = __float2half(center + 0.5f * s * lap);
            }
        }
        return;
    }

    extern __shared__ char smem_raw[];
    const int halo_dim  = WMMA_M + 2 * R;
    const int halo_size = halo_dim * halo_dim;

    size_t off = 0;
    __half* s_halo = reinterpret_cast<__half*>(smem_raw + off);
    off  = align_up_tc(off + halo_size * sizeof(__half),     SMEM_ALIGN);
    __half* s_A    = reinterpret_cast<__half*>(smem_raw + off);
    off  = align_up_tc(off + WMMA_M * K_PAD * sizeof(__half), SMEM_ALIGN);
    __half* s_B    = reinterpret_cast<__half*>(smem_raw + off);
    off  = align_up_tc(off + K_PAD * WMMA_M * sizeof(__half), SMEM_ALIGN);
    float*  s_D    = reinterpret_cast<float*>(smem_raw + off);

    const int g_halo_row0 = tile_row - R;
    const int g_halo_col0 = tile_col - R;

    for (int idx = lane; idx < halo_size; idx += 32) {
        int lj = idx / halo_dim;
        int li = idx % halo_dim;
        int gj = max(0, min(g_halo_row0 + lj, N - 1));
        int gi = max(0, min(g_halo_col0 + li, N - 1));
        s_halo[lj * halo_dim + li] = u0[gj * N + gi];
    }
    __syncwarp();

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
        for (int row = 0; row < WMMA_M; row++)
            s_A[row * K_PAD + k] = val;
    }
    __syncwarp();

    for (int j_local = 0; j_local < WMMA_N; j_local++) {
        const int hj = j_local + R;

        for (int idx = lane; idx < K_PAD * WMMA_M; idx += 32) {
            int k   = idx / WMMA_M;
            int col = idx % WMMA_M;
            int hi  = col + R;

            __half val;
            if (k == 0) {
                val = s_halo[hj * halo_dim + hi];
            } else if (k < S) {
                int m   = (k - 1) / 4 + 1;
                int dir = (k - 1) % 4;
                switch (dir) {
                    case 0: val = s_halo[hj       * halo_dim + (hi - m)]; break;
                    case 1: val = s_halo[hj       * halo_dim + (hi + m)]; break;
                    case 2: val = s_halo[(hj - m) * halo_dim +  hi     ]; break;
                    case 3: val = s_halo[(hj + m) * halo_dim +  hi     ]; break;
                    default: val = __float2half(0.0f);
                }
            } else {
                val = __float2half(0.0f);
            }
            s_B[k * WMMA_M + col] = val;
        }
        __syncwarp();

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_D;
        wmma::fill_fragment(frag_D, 0.0f);
        for (int kk = 0; kk < K_PAD; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_A;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_B;
            wmma::load_matrix_sync(frag_A, s_A + kk, K_PAD);
            wmma::load_matrix_sync(frag_B, s_B + kk * WMMA_M, WMMA_M);
            wmma::mma_sync(frag_D, frag_A, frag_B, frag_D);
        }
        wmma::store_matrix_sync(s_D, frag_D, WMMA_M, wmma::mem_row_major);
        __syncwarp();

        if (lane < WMMA_M) {
            float lap = s_D[lane * WMMA_M + lane];
            int gi    = tile_col + lane;
            int gj    = tile_row + j_local;
            int g_idx = gj * N + gi;
            float center = __half2float(u0[g_idx]);
            u1[g_idx] = __float2half(center + 0.5f * s * lap);
        }
        __syncwarp();
    }
}

__global__ void wave2d_fp16_tensor_core_kernel(
        const __half* __restrict__ u_prev,
        const __half* __restrict__ u,
        __half*       __restrict__ u_next,
        int N, float s)
{
    const int R = d_reach_tc;
    const int S = 4 * R + 1;
    const int K_PAD = ((S + WMMA_K - 1) / WMMA_K) * WMMA_K;

    const int tile_col = blockIdx.x * WMMA_M;
    const int tile_row = blockIdx.y * WMMA_N;
    const int lane = threadIdx.x;

    // Scalar fallback for boundary tiles (see comment in first-step kernel).
    if (tile_col < R || tile_row < R ||
        tile_col + WMMA_M > N - R || tile_row + WMMA_N > N - R) {
        for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
            int li = idx % WMMA_M;
            int lj = idx / WMMA_M;
            int gi = tile_col + li;
            int gj = tile_row + lj;
            if (gi >= R && gi < N - R && gj >= R && gj < N - R) {
                float center = __half2float(u[gj * N + gi]);
                float lap = 2.0f * d_coeffs_tc[0] * center;
                for (int m = 1; m <= R; m++) {
                    lap += d_coeffs_tc[m] * (
                        __half2float(u[gj * N + (gi - m)]) + __half2float(u[gj * N + (gi + m)])
                      + __half2float(u[(gj - m) * N + gi]) + __half2float(u[(gj + m) * N + gi]));
                }
                float prev = __half2float(u_prev[gj * N + gi]);
                u_next[gj * N + gi] = __float2half(2.0f * center - prev + s * lap);
            }
        }
        return;
    }

    extern __shared__ char smem_raw[];
    const int halo_dim  = WMMA_M + 2 * R;
    const int halo_size = halo_dim * halo_dim;

    size_t off = 0;
    __half* s_halo = reinterpret_cast<__half*>(smem_raw + off);
    off  = align_up_tc(off + halo_size * sizeof(__half),     SMEM_ALIGN);
    __half* s_A    = reinterpret_cast<__half*>(smem_raw + off);
    off  = align_up_tc(off + WMMA_M * K_PAD * sizeof(__half), SMEM_ALIGN);
    __half* s_B    = reinterpret_cast<__half*>(smem_raw + off);
    off  = align_up_tc(off + K_PAD * WMMA_M * sizeof(__half), SMEM_ALIGN);
    float*  s_D    = reinterpret_cast<float*>(smem_raw + off);

    const int g_halo_row0 = tile_row - R;
    const int g_halo_col0 = tile_col - R;

    for (int idx = lane; idx < halo_size; idx += 32) {
        int lj = idx / halo_dim;
        int li = idx % halo_dim;
        int gj = max(0, min(g_halo_row0 + lj, N - 1));
        int gi = max(0, min(g_halo_col0 + li, N - 1));
        s_halo[lj * halo_dim + li] = u[gj * N + gi];
    }
    __syncwarp();

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
        for (int row = 0; row < WMMA_M; row++)
            s_A[row * K_PAD + k] = val;
    }
    __syncwarp();

    for (int j_local = 0; j_local < WMMA_N; j_local++) {
        const int hj = j_local + R;

        for (int idx = lane; idx < K_PAD * WMMA_M; idx += 32) {
            int k   = idx / WMMA_M;
            int col = idx % WMMA_M;
            int hi  = col + R;

            __half val;
            if (k == 0) {
                val = s_halo[hj * halo_dim + hi];
            } else if (k < S) {
                int m   = (k - 1) / 4 + 1;
                int dir = (k - 1) % 4;
                switch (dir) {
                    case 0: val = s_halo[hj       * halo_dim + (hi - m)]; break;
                    case 1: val = s_halo[hj       * halo_dim + (hi + m)]; break;
                    case 2: val = s_halo[(hj - m) * halo_dim +  hi     ]; break;
                    case 3: val = s_halo[(hj + m) * halo_dim +  hi     ]; break;
                    default: val = __float2half(0.0f);
                }
            } else {
                val = __float2half(0.0f);
            }
            s_B[k * WMMA_M + col] = val;
        }
        __syncwarp();

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_D;
        wmma::fill_fragment(frag_D, 0.0f);
        for (int kk = 0; kk < K_PAD; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_A;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_B;
            wmma::load_matrix_sync(frag_A, s_A + kk, K_PAD);
            wmma::load_matrix_sync(frag_B, s_B + kk * WMMA_M, WMMA_M);
            wmma::mma_sync(frag_D, frag_A, frag_B, frag_D);
        }
        wmma::store_matrix_sync(s_D, frag_D, WMMA_M, wmma::mem_row_major);
        __syncwarp();

        if (lane < WMMA_M) {
            float lap = s_D[lane * WMMA_M + lane];
            int gi    = tile_col + lane;
            int gj    = tile_row + j_local;
            int g_idx = gj * N + gi;
            float center = __half2float(u[g_idx]);
            float prev   = __half2float(u_prev[g_idx]);
            u_next[g_idx] = __float2half(2.0f * center - prev + s * lap);
        }
        __syncwarp();
    }
}


StencilResult run_cuda_fp16_tensor_core(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float s_factor = (cfg.c * cfg.dt / cfg.dx) * (cfg.c * cfg.dt / cfg.dx);
    size_t n_elems = (size_t)N * N;
    size_t hbytes = n_elems * sizeof(__half);

    assert(N % WMMA_M == 0 && "Grid size N must be a multiple of 16 for WMMA");

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_tc, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_tc, &R, sizeof(int)));

    std::vector<float> h_u0(n_elems, cfg.disp_initial);
    init_wave_field_2d_tc(h_u0, N, cfg.source_amplitude, cfg.source_sigma);
    auto h_half = float_to_half_tc(h_u0);

    __half *d_u_prev, *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u_prev, hbytes));
    CUDA_CHECK(cudaMalloc(&d_u, hbytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, hbytes));
    CUDA_CHECK(cudaMemcpy(d_u_prev, h_half.data(), hbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u, 0, hbytes));
    CUDA_CHECK(cudaMemset(d_u_next, 0, hbytes));

    int tiles_x = N / WMMA_M;
    int tiles_y = N / WMMA_N;
    dim3 grid_dim(tiles_x, tiles_y);
    dim3 block_dim(32, 1);

    int S = 4 * R + 1;
    int K_PAD = ((S + WMMA_K - 1) / WMMA_K) * WMMA_K;
    int halo_dim = WMMA_M + 2 * R;
    size_t smem_bytes = 0;
    smem_bytes  = align_up_tc(smem_bytes + halo_dim * halo_dim * sizeof(__half), SMEM_ALIGN);
    smem_bytes  = align_up_tc(smem_bytes + WMMA_M * K_PAD      * sizeof(__half), SMEM_ALIGN);
    smem_bytes  = align_up_tc(smem_bytes + K_PAD  * WMMA_M     * sizeof(__half), SMEM_ALIGN);
    smem_bytes += WMMA_M * WMMA_N * sizeof(float);

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
        res.memory_bytes = 0;
        CUDA_CHECK(cudaFree(d_u_prev));
        CUDA_CHECK(cudaFree(d_u));
        CUDA_CHECK(cudaFree(d_u_next));
        return res;
    }

    int bc_threads = 256;
    int bc_blocks = (N + bc_threads - 1) / bc_threads;

    wave2d_fp16_tensor_core_first_step_kernel<<<grid_dim, block_dim, smem_bytes>>>(
        d_u_prev, d_u, N, s_factor);
    apply_dirichlet_fp16_tc<<<bc_blocks, bc_threads>>>(d_u_prev, N, R);
    apply_dirichlet_fp16_tc<<<bc_blocks, bc_threads>>>(d_u, N, R);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 1; t < cfg.timesteps; t++) {
        wave2d_fp16_tensor_core_kernel<<<grid_dim, block_dim, smem_bytes>>>(
            d_u_prev, d_u, d_u_next, N, s_factor);
        apply_dirichlet_fp16_tc<<<bc_blocks, bc_threads>>>(d_u_next, N, R);
        __half* tmp = d_u_prev;
        d_u_prev = d_u;
        d_u = d_u_next;
        d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_half.data(), d_u, hbytes, cudaMemcpyDeviceToHost));
    std::vector<float> result_f(n_elems);
    for (size_t i = 0; i < n_elems; i++) result_f[i] = __half2float(h_half[i]);

    int interior = N - 2 * R;
    double reads = (2.0 * 2.0 * R + 2.0);
    double writes = 1.0;
    double bytes_per_step = (double)interior * interior * (reads + writes) * sizeof(__half);
    double bw = bytes_per_step * (cfg.timesteps - 1) / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp16_tensor_core";
    res.grid_size = N;
    res.dim = cfg.dim;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes = 3 * hbytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

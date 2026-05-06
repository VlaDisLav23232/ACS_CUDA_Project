// async pipeline variant for the 3D wave equation (fp16 + Kahan).
//
// 2-stage block-scope cuda::memcpy_async pipeline streaming z-slabs of the
// current field (u, c) into double-buffered shared memory. While the block
// computes slab z, slab z+1 is prefetched via cuda::memcpy_async. Follows the
// multi-stage pattern from NVIDIA's CUDA C++ Programming Guide
// (Pipelines / Async Copies). u_prev and c_prev are read directly from global
// at the centre cell only — they don't participate in the laplacian.

#include "stencil.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static const int TILE_X = 16;
static const int TILE_Y = 16;
static const int NUM_STAGES = 2;

__constant__ float d_coeffs_async[MAX_REACH + 1];
__constant__ int   d_reach_async;


// Sync first-step kernel (only runs once, no benefit to making it async).
__global__ void wave3d_fp16_kahan_async_first_step_kernel(const __half* __restrict__ u0,
                                                          __half* __restrict__ u1,
                                                          float* __restrict__ c1,
                                                          int N, float s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_reach_async;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = (size_t)z * N * N + y * N + x;
        float center = __half2float(u0[idx]);
        float lap = 3.0f * d_coeffs_async[0] * center;
        for (int m = 1; m <= R; m++) {
            size_t ox = (size_t)m;
            size_t oy = (size_t)m * N;
            size_t oz = (size_t)m * N * N;
            lap += d_coeffs_async[m] * (
                __half2float(u0[idx - ox]) + __half2float(u0[idx + ox])
              + __half2float(u0[idx - oy]) + __half2float(u0[idx + oy])
              + __half2float(u0[idx - oz]) + __half2float(u0[idx + oz]));
        }
        float exact = center + 0.5f * s * lap;
        __half stored = __float2half(exact);
        u1[idx] = stored;
        c1[idx] = exact - __half2float(stored);
    }
}


__global__ void wave3d_fp16_kahan_25d_async_kernel(const __half* __restrict__ u_prev,
                                                   const __half* __restrict__ u,
                                                   __half* __restrict__ u_next,
                                                   const float* __restrict__ c_prev,
                                                   const float* __restrict__ c,
                                                   float* __restrict__ c_next,
                                                   int N, float s) {
    auto block = cg::this_thread_block();
    int R = d_reach_async;

    int smem_w = TILE_X + 2 * R;
    int smem_h = TILE_Y + 2 * R;
    int slab_cells = smem_w * smem_h;

    extern __shared__ char smem_raw[];
    __half* smem_u_buf[NUM_STAGES];
    float*  smem_c_buf[NUM_STAGES];
    {
        char* p = smem_raw;
        for (int sg = 0; sg < NUM_STAGES; sg++) {
            smem_u_buf[sg] = reinterpret_cast<__half*>(p);
            p += slab_cells * sizeof(__half);
            smem_c_buf[sg] = reinterpret_cast<float*>(p);
            p += slab_cells * sizeof(float);
        }
    }

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, NUM_STAGES> shared_state;
    auto pipe = cuda::make_pipeline(block, &shared_state);

    int tile_x = blockIdx.x * TILE_X + threadIdx.x;
    int tile_y = blockIdx.y * TILE_Y + threadIdx.y;
    int tid = threadIdx.y * TILE_X + threadIdx.x;
    int n_threads = TILE_X * TILE_Y;

    auto stage_slab = [&](int z_in, int slot) {
        for (int idx = tid; idx < slab_cells; idx += n_threads) {
            int sy = idx / smem_w;
            int sx = idx - sy * smem_w;
            int gx = blockIdx.x * TILE_X + sx - R;
            int gy = blockIdx.y * TILE_Y + sy - R;
            gx = max(0, min(gx, N - 1));
            gy = max(0, min(gy, N - 1));
            size_t gidx = (size_t)z_in * N * N + gy * N + gx;
            cuda::memcpy_async(&smem_u_buf[slot][idx], &u[gidx],
                               cuda::aligned_size_t<2>(sizeof(__half)), pipe);
            cuda::memcpy_async(&smem_c_buf[slot][idx], &c[gidx],
                               cuda::aligned_size_t<4>(sizeof(float)), pipe);
        }
    };

    int prefetch_z = 0;
    #pragma unroll
    for (int sg = 0; sg < NUM_STAGES; sg++) {
        if (prefetch_z < N) {
            pipe.producer_acquire();
            stage_slab(prefetch_z, sg);
            pipe.producer_commit();
            prefetch_z++;
        }
    }

    int compute_slot = 0;
    for (int z = 0; z < N; z++) {
        pipe.consumer_wait();
        block.sync();

        if (tile_x >= R && tile_x < N - R &&
            tile_y >= R && tile_y < N - R &&
            z >= R && z < N - R) {

            __half* su = smem_u_buf[compute_slot];
            float*  sc = smem_c_buf[compute_slot];
            int lx = threadIdx.x + R;
            int ly = threadIdx.y + R;

            float center = __half2float(su[ly * smem_w + lx]) + sc[ly * smem_w + lx];
            float lap = 3.0f * d_coeffs_async[0] * center;

            for (int m = 1; m <= R; m++) {
                float xm = __half2float(su[ly * smem_w + (lx - m)]) + sc[ly * smem_w + (lx - m)];
                float xp = __half2float(su[ly * smem_w + (lx + m)]) + sc[ly * smem_w + (lx + m)];
                float ym = __half2float(su[(ly - m) * smem_w + lx]) + sc[(ly - m) * smem_w + lx];
                float yp = __half2float(su[(ly + m) * smem_w + lx]) + sc[(ly + m) * smem_w + lx];

                size_t idx_center = (size_t)z * N * N + tile_y * N + tile_x;
                float zm = __half2float(u[idx_center - (size_t)m * N * N])
                         + c[idx_center - (size_t)m * N * N];
                float zp = __half2float(u[idx_center + (size_t)m * N * N])
                         + c[idx_center + (size_t)m * N * N];

                lap += d_coeffs_async[m] * (xm + xp + ym + yp + zm + zp);
            }

            size_t out_idx = (size_t)z * N * N + tile_y * N + tile_x;
            float prev = __half2float(u_prev[out_idx]) + c_prev[out_idx];
            float exact = 2.0f * center - prev + s * lap;
            __half stored = __float2half(exact);
            u_next[out_idx] = stored;
            c_next[out_idx] = exact - __half2float(stored);
        }

        block.sync();
        pipe.consumer_release();
        block.sync();

        if (prefetch_z < N) {
            pipe.producer_acquire();
            stage_slab(prefetch_z, compute_slot);
            pipe.producer_commit();
            prefetch_z++;
        }

        compute_slot = (compute_slot + 1) % NUM_STAGES;
    }
}


// Dirichlet-zero BC kernels (mirror the sync wave3d kernels but with unique names).
__global__ void apply_dirichlet_3d_async_fp16(__half* u, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b >= N) return;
    __half z = __float2half(0.0f);
    for (int r = 0; r < R; r++) {
        u[(size_t)a * N * N + b * N + r] = z;
        u[(size_t)a * N * N + b * N + (N - 1 - r)] = z;
        u[(size_t)a * N * N + r * N + b] = z;
        u[(size_t)a * N * N + (N - 1 - r) * N + b] = z;
        u[(size_t)r * N * N + a * N + b] = z;
        u[(size_t)(N - 1 - r) * N * N + a * N + b] = z;
    }
}

__global__ void apply_dirichlet_3d_async_float(float* c, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b >= N) return;
    for (int r = 0; r < R; r++) {
        c[(size_t)a * N * N + b * N + r] = 0.0f;
        c[(size_t)a * N * N + b * N + (N - 1 - r)] = 0.0f;
        c[(size_t)a * N * N + r * N + b] = 0.0f;
        c[(size_t)a * N * N + (N - 1 - r) * N + b] = 0.0f;
        c[(size_t)r * N * N + a * N + b] = 0.0f;
        c[(size_t)(N - 1 - r) * N * N + a * N + b] = 0.0f;
    }
}


static void init_wave_field_3d_async(std::vector<float>& u, int N, float amp, float sigma) {
    const float s2 = sigma * sigma;
    const float src[2][3] = {{-0.30f, -0.30f, -0.30f}, {0.30f, 0.30f, 0.30f}};
    for (int z = 0; z < N; z++) {
        float zz = -1.0f + 2.0f * z / (N - 1.0f);
        for (int y = 0; y < N; y++) {
            float yy = -1.0f + 2.0f * y / (N - 1.0f);
            for (int x = 0; x < N; x++) {
                float xx = -1.0f + 2.0f * x / (N - 1.0f);
                float v = 0.0f;
                for (int k = 0; k < 2; k++) {
                    float dx = xx - src[k][0];
                    float dy = yy - src[k][1];
                    float dz = zz - src[k][2];
                    v += amp * expf(-(dx * dx + dy * dy + dz * dz) / s2);
                }
                u[(size_t)z * N * N + y * N + x] = v;
            }
        }
    }
}

static std::vector<__half> float_to_half_3d_async(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); i++) h[i] = __float2half(f[i]);
    return h;
}


StencilResult run_cuda_fp16_kahan_3d_25d_async(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float s = (cfg.c * cfg.dt / cfg.dx) * (cfg.c * cfg.dt / cfg.dx);
    size_t total = (size_t)N * N * N;
    size_t hbytes = total * sizeof(__half);
    size_t fbytes = total * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_async, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_async, &R, sizeof(int)));

    std::vector<float> h_u0(total, cfg.disp_initial);
    init_wave_field_3d_async(h_u0, N, cfg.source_amplitude, cfg.source_sigma);
    auto h_half = float_to_half_3d_async(h_u0);

    __half *d_u_prev, *d_u, *d_u_next;
    float  *d_c_prev, *d_c, *d_c_next;
    CUDA_CHECK(cudaMalloc(&d_u_prev, hbytes));
    CUDA_CHECK(cudaMalloc(&d_u, hbytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, hbytes));
    CUDA_CHECK(cudaMalloc(&d_c_prev, fbytes));
    CUDA_CHECK(cudaMalloc(&d_c, fbytes));
    CUDA_CHECK(cudaMalloc(&d_c_next, fbytes));

    CUDA_CHECK(cudaMemcpy(d_u_prev, h_half.data(), hbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u, 0, hbytes));
    CUDA_CHECK(cudaMemset(d_u_next, 0, hbytes));
    CUDA_CHECK(cudaMemset(d_c_prev, 0, fbytes));
    CUDA_CHECK(cudaMemset(d_c, 0, fbytes));
    CUDA_CHECK(cudaMemset(d_c_next, 0, fbytes));

    // First step: simple sync 3D launch (one-shot, no benefit to async).
    dim3 first_block(8, 8, 4);
    dim3 first_grid((N + 7) / 8, (N + 7) / 8, (N + 3) / 4);

    // Main async loop: 2D grid (one block per XY tile, walks all z-slabs).
    dim3 block(TILE_X, TILE_Y);
    dim3 grid25((N + TILE_X - 1) / TILE_X, (N + TILE_Y - 1) / TILE_Y);
    int smem_tile = (TILE_X + 2 * R) * (TILE_Y + 2 * R);
    size_t smem_bytes = (size_t)NUM_STAGES * smem_tile * (sizeof(__half) + sizeof(float));

    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    wave3d_fp16_kahan_async_first_step_kernel<<<first_grid, first_block>>>(d_u_prev, d_u, d_c, N, s);
    apply_dirichlet_3d_async_fp16<<<bc_grid, bc_block>>>(d_u_prev, N, R);
    apply_dirichlet_3d_async_fp16<<<bc_grid, bc_block>>>(d_u, N, R);
    apply_dirichlet_3d_async_float<<<bc_grid, bc_block>>>(d_c_prev, N, R);
    apply_dirichlet_3d_async_float<<<bc_grid, bc_block>>>(d_c, N, R);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 1; t < cfg.timesteps; t++) {
        wave3d_fp16_kahan_25d_async_kernel<<<grid25, block, smem_bytes>>>(
            d_u_prev, d_u, d_u_next, d_c_prev, d_c, d_c_next, N, s);
        apply_dirichlet_3d_async_fp16<<<bc_grid, bc_block>>>(d_u_next, N, R);
        apply_dirichlet_3d_async_float<<<bc_grid, bc_block>>>(d_c_next, N, R);

        __half* tmp_h = d_u_prev; d_u_prev = d_u; d_u = d_u_next; d_u_next = tmp_h;
        float*  tmp_c = d_c_prev; d_c_prev = d_c; d_c = d_c_next; d_c_next = tmp_c;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::vector<float> h_comp(total);
    CUDA_CHECK(cudaMemcpy(h_half.data(), d_u, hbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_c, fbytes, cudaMemcpyDeviceToHost));

    std::vector<float> result(total);
    for (size_t i = 0; i < total; i++) result[i] = __half2float(h_half[i]) + h_comp[i];

    int interior = N - 2 * R;
    double reads = (2.0 * 3.0 * R + 2.0);
    double writes = 1.0;
    double half_rw = (reads + writes) * sizeof(__half);
    double comp_rw = (reads + writes) * sizeof(float);
    double bytes_per_step = (double)interior * interior * interior * (half_rw + comp_rw);
    double bw = bytes_per_step * (cfg.timesteps - 1) / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp16_kahan_3d_25d_async";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes = 3 * hbytes + 3 * fbytes;
    res.final_grid = result;

    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_c_prev));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_c_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

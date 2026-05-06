// async pipeline variant for fp16 + Kahan 3D heat stencil.
//
// heat3d_fp16_kahan_25d_async_kernel — 2-stage, block-scope pipeline.
// Each block iterates over all z layers for its (bx, by) tile and uses
// double-buffered shared memory: while computing slab z, the next slab z+1 is
// prefetched via cuda::memcpy_async. Follows the multi-stage pattern from
// NVIDIA's CUDA C++ Programming Guide (Pipelines / Async Copies sections).

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

__global__ void heat3d_fp16_kahan_25d_async_kernel(const __half* __restrict__ u,
                                                   __half* __restrict__ u_next,
                                                   const float* __restrict__ c,
                                                   float* __restrict__ c_next,
                                                   int N, float r) {
    auto block = cg::this_thread_block();
    int R = d_reach_async;

    int smem_w = TILE_X + 2 * R;
    int smem_h = TILE_Y + 2 * R;
    int slab_cells = smem_w * smem_h;

    extern __shared__ char smem_raw_25d[];
    __half* smem_u_buf[NUM_STAGES];
    float*  smem_c_buf[NUM_STAGES];
    {
        char* p = smem_raw_25d;
        for (int s = 0; s < NUM_STAGES; s++) {
            smem_u_buf[s] = reinterpret_cast<__half*>(p);
            p += slab_cells * sizeof(__half);
            smem_c_buf[s] = reinterpret_cast<float*>(p);
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
    for (int s = 0; s < NUM_STAGES; s++) {
        if (prefetch_z < N) {
            pipe.producer_acquire();
            stage_slab(prefetch_z, s);
            pipe.producer_commit();
            prefetch_z++;
        }
    }

    int compute_slot = 0;
    for (int z = 0; z < N; z++) {
        pipe.consumer_wait();
        block.sync();   // make staged smem visible to all threads in the block

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

                // z-neighbors come from global; classic 2.5D trade-off.
                size_t idx_center = (size_t)z * N * N + tile_y * N + tile_x;
                float zm = __half2float(u[idx_center - (size_t)m * N * N])
                         + c[idx_center - (size_t)m * N * N];
                float zp = __half2float(u[idx_center + (size_t)m * N * N])
                         + c[idx_center + (size_t)m * N * N];

                lap += d_coeffs_async[m] * (xm + xp + ym + yp + zm + zp);
            }

            float exact_result = center + r * lap;
            __half stored = __float2half(exact_result);
            size_t out_idx = (size_t)z * N * N + tile_y * N + tile_x;
            u_next[out_idx] = stored;
            volatile float stored_back = __half2float(stored);
            c_next[out_idx] = exact_result - stored_back;
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


__global__ void apply_neumann_bc_3d_async_fp16(__half* u, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;

    for (int b = R - 1; b >= 0; b--) {
        u[(size_t)a * N * N + b_idx * N + b]           = u[(size_t)a * N * N + b_idx * N + (b + 1)];
        u[(size_t)a * N * N + b_idx * N + (N - 1 - b)] = u[(size_t)a * N * N + b_idx * N + (N - 2 - b)];
        u[(size_t)a * N * N + b * N + b_idx]           = u[(size_t)a * N * N + (b + 1) * N + b_idx];
        u[(size_t)a * N * N + (N - 1 - b) * N + b_idx] = u[(size_t)a * N * N + (N - 2 - b) * N + b_idx];
        u[(size_t)b * N * N + a * N + b_idx]           = u[(size_t)(b + 1) * N * N + a * N + b_idx];
        u[(size_t)(N - 1 - b) * N * N + a * N + b_idx] = u[(size_t)(N - 2 - b) * N * N + a * N + b_idx];
    }
}

__global__ void apply_neumann_bc_3d_async_comp(float* c, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;

    for (int b = R - 1; b >= 0; b--) {
        c[(size_t)a * N * N + b_idx * N + b]           = c[(size_t)a * N * N + b_idx * N + (b + 1)];
        c[(size_t)a * N * N + b_idx * N + (N - 1 - b)] = c[(size_t)a * N * N + b_idx * N + (N - 2 - b)];
        c[(size_t)a * N * N + b * N + b_idx]           = c[(size_t)a * N * N + (b + 1) * N + b_idx];
        c[(size_t)a * N * N + (N - 1 - b) * N + b_idx] = c[(size_t)a * N * N + (N - 2 - b) * N + b_idx];
        c[(size_t)b * N * N + a * N + b_idx]           = c[(size_t)(b + 1) * N * N + a * N + b_idx];
        c[(size_t)(N - 1 - b) * N * N + a * N + b_idx] = c[(size_t)(N - 2 - b) * N * N + a * N + b_idx];
    }
}

static std::vector<__half> float_to_half_async(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); i++) {
        h[i] = __float2half(f[i]);
    }
    return h;
}


StencilResult run_cuda_fp16_kahan_3d_25d_async(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total = (size_t)N * N * N;
    size_t half_bytes = total * sizeof(__half);
    size_t float_bytes = total * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_async, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_async, &R, sizeof(int)));

    std::vector<float> h_f(total, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int z = src_start; z < src_start + src_size; z++)
        for (int y = src_start; y < src_start + src_size; y++)
            for (int x = src_start; x < src_start + src_size; x++)
                h_f[(size_t)z * N * N + y * N + x] = cfg.temp_source;

    auto h_data = float_to_half_async(h_f);
    std::vector<float> h_comp(total, 0.0f);

    __half* d_u = nullptr;
    __half* d_u_next = nullptr;
    float* d_c = nullptr;
    float* d_c_next = nullptr;

    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_next, float_bytes));

    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_next, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_X, TILE_Y);
    dim3 grid25((N + TILE_X - 1) / TILE_X, (N + TILE_Y - 1) / TILE_Y);

    int smem_tile = (TILE_X + 2 * R) * (TILE_Y + 2 * R);
    size_t smem_bytes = (size_t)NUM_STAGES * smem_tile * (sizeof(__half) + sizeof(float));

    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat3d_fp16_kahan_25d_async_kernel<<<grid25, block, smem_bytes>>>(d_u, d_u_next, d_c, d_c_next, N, r);
        apply_neumann_bc_3d_async_fp16<<<bc_grid, bc_block>>>(d_u_next, N, R);
        apply_neumann_bc_3d_async_comp<<<bc_grid, bc_block>>>(d_c_next, N, R);

        __half* tmp_h = d_u;
        d_u = d_u_next;
        d_u_next = tmp_h;
        float* tmp_c = d_c;
        d_c = d_c_next;
        d_c_next = tmp_c;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(total);
    for (size_t i = 0; i < total; i++)
        result_f[i] = __half2float(h_data[i]) + h_comp[i];

    double bytes_per_step = 2.0 * (double)N * N * N * sizeof(__half)
                          + 2.0 * (double)N * N * N * sizeof(float);
    double total_bw_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bw_bytes / (elapsed_ms / 1000.0) / 1e9;
    double mpts = ((double)N * N * N * cfg.timesteps) / (elapsed_ms / 1000.0) / 1e6;

    StencilResult res;
    res.variant_name = "cuda_fp16_kahan_3d_25d_async";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * half_bytes + 2 * float_bytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_c_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}


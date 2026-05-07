/*
 * 3D heat stencil: fp16 naive/Kahan 2.5D + Z register window + async XY staging.
 *
 * The Z neighbors stay in a per-thread register window. The async pipeline only
 * double-buffers the XY shared-memory tile for the next Z plane inside each
 * chunk, so this is a direct test of whether async still helps after Z traffic
 * was reduced by the register window.
 */

#include "stencil.h"
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK_ZREG_ASYNC(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static const int ZREG_ASYNC_STAGES = 2;

__constant__ float d_coeffs_zreg_async[MAX_REACH + 1];
__constant__ int d_reach_zreg_async;

struct AsyncZRegTileConfig3D16 {
    int tile_x;
    int tile_y;
    int z_chunk;
    size_t smem_bytes;
};

static size_t align4_zreg_async(size_t value) {
    return (value + 3) & ~static_cast<size_t>(3);
}

static size_t async_zreg_stage_bytes(int tile_x, int tile_y, int R, bool with_compensation) {
    size_t tile_elems = static_cast<size_t>(tile_x + 2 * R) * (tile_y + 2 * R);
    if (!with_compensation) return tile_elems * sizeof(__half);
    return align4_zreg_async(tile_elems * sizeof(__half)) + tile_elems * sizeof(float);
}

static AsyncZRegTileConfig3D16 query_fp16_zreg_async_tile(int R, int N, bool with_compensation) {
    cudaDeviceProp prop;
    CUDA_CHECK_ZREG_ASYNC(cudaGetDeviceProperties(&prop, 0));

    size_t smem_avail = prop.sharedMemPerBlock;
    int max_threads = prop.maxThreadsPerBlock;
    int max_regs = prop.regsPerBlock;
    int sm_count = prop.multiProcessorCount;
    int regs_per_thread = (with_compensation ? 2 : 1) * (2 * R + 1) + 40;

    static const int candidates[][2] = {
        {32, 32}, {32, 16}, {16, 32}, {32, 8}, {16, 16}, {16, 8}, {8, 16}, {8, 8}, {4, 4}
    };

    for (const auto& candidate : candidates) {
        int tx = candidate[0];
        int ty = candidate[1];
        int threads = tx * ty;
        if (threads > max_threads) continue;

        size_t smem = ZREG_ASYNC_STAGES * async_zreg_stage_bytes(tx, ty, R, with_compensation);
        if (smem > smem_avail) continue;
        if (threads * regs_per_thread > max_regs) continue;

        int xy_blocks = ((N + tx - 1) / tx) * ((N + ty - 1) / ty);
        int interior_z = N - 2 * R;
        if (interior_z <= 0) continue;

        int target_blocks = sm_count * 4;
        int z_chunks = max(1, (target_blocks + xy_blocks - 1) / xy_blocks);
        z_chunks = min(z_chunks, interior_z);
        int z_chunk = (interior_z + z_chunks - 1) / z_chunks;

        printf("  3D fp16 2.5D+zreg+async tile: %dx%d XY, z_chunk=%d, smem=%zu bytes\n", tx, ty, z_chunk, smem);
        return {tx, ty, z_chunk, smem};
    }

    size_t fallback_smem = ZREG_ASYNC_STAGES * async_zreg_stage_bytes(4, 4, R, with_compensation);
    printf("  warning: fallback 4x4 fp16 2.5D+zreg+async tile\n");
    return {4, 4, 1, fallback_smem};
}

static std::vector<__half> zreg_async_float_to_half_vec(const std::vector<float>& values) {
    std::vector<__half> half_values(values.size());
    for (size_t i = 0; i < values.size(); ++i) half_values[i] = __float2half(values[i]);
    return half_values;
}

__global__ void heat3d_fp16_naive_25d_zreg_async_kernel(
    const __half* __restrict__ u,
    __half* __restrict__ u_next,
    int N,
    float r,
    int tile_x,
    int tile_y,
    int z_chunk
) {
    auto block = cg::this_thread_block();
    const int R = d_reach_zreg_async;
    const int diam = 2 * R + 1;
    const int sw = tile_x + 2 * R;
    const int sh = tile_y + 2 * R;
    const int tile_elems = sw * sh;

    extern __shared__ char smem_raw[];
    __half* s_u[ZREG_ASYNC_STAGES];
    char* p = smem_raw;
    for (int stage = 0; stage < ZREG_ASYNC_STAGES; ++stage) {
        s_u[stage] = reinterpret_cast<__half*>(p);
        p += static_cast<size_t>(tile_elems) * sizeof(__half);
    }

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, ZREG_ASYNC_STAGES> shared_state;
    auto pipe = cuda::make_pipeline(block, &shared_state);

    const int gx = blockIdx.x * tile_x + threadIdx.x;
    const int gy = blockIdx.y * tile_y + threadIdx.y;
    const int bx = blockIdx.x * tile_x - R;
    const int by = blockIdx.y * tile_y - R;
    const int gz_first = R + blockIdx.z * z_chunk;
    const int gz_last = min(N - R - 1, gz_first + z_chunk - 1);
    if (gz_first > gz_last) return;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;
    const bool active = gx >= R && gx < N - R && gy >= R && gy < N - R;

    auto stage_slab = [&](int gz, int slot) {
        for (int idx = tid; idx < tile_elems; idx += block_size) {
            int sy = idx / sw;
            int sx = idx % sw;
            int ci = max(0, min(N - 1, bx + sx));
            int cj = max(0, min(N - 1, by + sy));
            size_t gidx = static_cast<size_t>(gz) * N * N + static_cast<size_t>(cj) * N + ci;
            cuda::memcpy_async(&s_u[slot][idx], &u[gidx], cuda::aligned_size_t<2>(sizeof(__half)), pipe);
        }
    };

    float z_buf[2 * MAX_REACH + 1];
    int cx = gx < N ? gx : N - 1;
    int cy = gy < N ? gy : N - 1;
    for (int m = 0; m < diam; ++m) {
        int gz = max(0, min(N - 1, gz_first - R + m));
        z_buf[m] = __half2float(u[static_cast<size_t>(gz) * N * N + static_cast<size_t>(cy) * N + cx]);
    }

    int prefetch_z = gz_first;
    for (int stage = 0; stage < ZREG_ASYNC_STAGES && prefetch_z <= gz_last; ++stage) {
        pipe.producer_acquire();
        stage_slab(prefetch_z, stage);
        pipe.producer_commit();
        ++prefetch_z;
    }

    int compute_slot = 0;
    for (int gz = gz_first; gz <= gz_last; ++gz) {
        pipe.consumer_wait();
        block.sync();

        if (active) {
            int si = threadIdx.x + R;
            int sj = threadIdx.y + R;
            float center = z_buf[R];
            float lap = 3.0f * d_coeffs_zreg_async[0] * center;

            for (int m = 1; m <= R; ++m) {
                float xy = __half2float(s_u[compute_slot][sj * sw + si - m])
                         + __half2float(s_u[compute_slot][sj * sw + si + m])
                         + __half2float(s_u[compute_slot][(sj - m) * sw + si])
                         + __half2float(s_u[compute_slot][(sj + m) * sw + si]);
                float zn = z_buf[R - m] + z_buf[R + m];
                lap += d_coeffs_zreg_async[m] * (xy + zn);
            }

            size_t out_idx = static_cast<size_t>(gz) * N * N + static_cast<size_t>(gy) * N + gx;
            u_next[out_idx] = __float2half(center + r * lap);
        }

        block.sync();
        pipe.consumer_release();
        block.sync();

        for (int m = 0; m < diam - 1; ++m) z_buf[m] = z_buf[m + 1];
        int next_z = max(0, min(N - 1, gz + R + 1));
        z_buf[diam - 1] = __half2float(u[static_cast<size_t>(next_z) * N * N + static_cast<size_t>(cy) * N + cx]);

        if (prefetch_z <= gz_last) {
            pipe.producer_acquire();
            stage_slab(prefetch_z, compute_slot);
            pipe.producer_commit();
            ++prefetch_z;
        }
        compute_slot = (compute_slot + 1) % ZREG_ASYNC_STAGES;
    }
}

__global__ void heat3d_fp16_kahan_25d_zreg_async_kernel(
    const __half* __restrict__ u,
    __half* __restrict__ u_next,
    const float* __restrict__ c,
    float* __restrict__ c_next,
    int N,
    float r,
    int tile_x,
    int tile_y,
    int z_chunk
) {
    auto block = cg::this_thread_block();
    const int R = d_reach_zreg_async;
    const int diam = 2 * R + 1;
    const int sw = tile_x + 2 * R;
    const int sh = tile_y + 2 * R;
    const int tile_elems = sw * sh;

    extern __shared__ char smem_raw[];
    __half* s_u[ZREG_ASYNC_STAGES];
    float* s_c[ZREG_ASYNC_STAGES];
    char* p = smem_raw;
    for (int stage = 0; stage < ZREG_ASYNC_STAGES; ++stage) {
        s_u[stage] = reinterpret_cast<__half*>(p);
        p += static_cast<size_t>(tile_elems) * sizeof(__half);
        uintptr_t aligned = (reinterpret_cast<uintptr_t>(p) + 3) & ~static_cast<uintptr_t>(3);
        p = reinterpret_cast<char*>(aligned);
        s_c[stage] = reinterpret_cast<float*>(p);
        p += static_cast<size_t>(tile_elems) * sizeof(float);
    }

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, ZREG_ASYNC_STAGES> shared_state;
    auto pipe = cuda::make_pipeline(block, &shared_state);

    const int gx = blockIdx.x * tile_x + threadIdx.x;
    const int gy = blockIdx.y * tile_y + threadIdx.y;
    const int bx = blockIdx.x * tile_x - R;
    const int by = blockIdx.y * tile_y - R;
    const int gz_first = R + blockIdx.z * z_chunk;
    const int gz_last = min(N - R - 1, gz_first + z_chunk - 1);
    if (gz_first > gz_last) return;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;
    const bool active = gx >= R && gx < N - R && gy >= R && gy < N - R;

    auto stage_slab = [&](int gz, int slot) {
        for (int idx = tid; idx < tile_elems; idx += block_size) {
            int sy = idx / sw;
            int sx = idx % sw;
            int ci = max(0, min(N - 1, bx + sx));
            int cj = max(0, min(N - 1, by + sy));
            size_t gidx = static_cast<size_t>(gz) * N * N + static_cast<size_t>(cj) * N + ci;
            cuda::memcpy_async(&s_u[slot][idx], &u[gidx], cuda::aligned_size_t<2>(sizeof(__half)), pipe);
            cuda::memcpy_async(&s_c[slot][idx], &c[gidx], cuda::aligned_size_t<4>(sizeof(float)), pipe);
        }
    };

    float z_buf[2 * MAX_REACH + 1];
    int cx = gx < N ? gx : N - 1;
    int cy = gy < N ? gy : N - 1;
    for (int m = 0; m < diam; ++m) {
        int gz = max(0, min(N - 1, gz_first - R + m));
        size_t idx = static_cast<size_t>(gz) * N * N + static_cast<size_t>(cy) * N + cx;
        z_buf[m] = __half2float(u[idx]) + c[idx];
    }

    int prefetch_z = gz_first;
    for (int stage = 0; stage < ZREG_ASYNC_STAGES && prefetch_z <= gz_last; ++stage) {
        pipe.producer_acquire();
        stage_slab(prefetch_z, stage);
        pipe.producer_commit();
        ++prefetch_z;
    }

    int compute_slot = 0;
    for (int gz = gz_first; gz <= gz_last; ++gz) {
        pipe.consumer_wait();
        block.sync();

        if (active) {
            int si = threadIdx.x + R;
            int sj = threadIdx.y + R;
            float center = z_buf[R];
            float lap = 3.0f * d_coeffs_zreg_async[0] * center;

            for (int m = 1; m <= R; ++m) {
                int s_xm = sj * sw + si - m;
                int s_xp = sj * sw + si + m;
                int s_ym = (sj - m) * sw + si;
                int s_yp = (sj + m) * sw + si;
                float xy = (__half2float(s_u[compute_slot][s_xm]) + s_c[compute_slot][s_xm])
                         + (__half2float(s_u[compute_slot][s_xp]) + s_c[compute_slot][s_xp])
                         + (__half2float(s_u[compute_slot][s_ym]) + s_c[compute_slot][s_ym])
                         + (__half2float(s_u[compute_slot][s_yp]) + s_c[compute_slot][s_yp]);
                float zn = z_buf[R - m] + z_buf[R + m];
                lap += d_coeffs_zreg_async[m] * (xy + zn);
            }

            float exact = center + r * lap;
            __half stored = __float2half(exact);
            size_t out_idx = static_cast<size_t>(gz) * N * N + static_cast<size_t>(gy) * N + gx;
            u_next[out_idx] = stored;
            volatile float stored_back = __half2float(stored);
            c_next[out_idx] = exact - stored_back;
        }

        block.sync();
        pipe.consumer_release();
        block.sync();

        for (int m = 0; m < diam - 1; ++m) z_buf[m] = z_buf[m + 1];
        int next_z = max(0, min(N - 1, gz + R + 1));
        size_t next_idx = static_cast<size_t>(next_z) * N * N + static_cast<size_t>(cy) * N + cx;
        z_buf[diam - 1] = __half2float(u[next_idx]) + c[next_idx];

        if (prefetch_z <= gz_last) {
            pipe.producer_acquire();
            stage_slab(prefetch_z, compute_slot);
            pipe.producer_commit();
            ++prefetch_z;
        }
        compute_slot = (compute_slot + 1) % ZREG_ASYNC_STAGES;
    }
}

__global__ void apply_neumann_bc_25d_zreg_async_half(__half* u, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;
    for (int b = R - 1; b >= 0; --b) {
        u[static_cast<size_t>(a) * N * N + static_cast<size_t>(b_idx) * N + b] = u[static_cast<size_t>(a) * N * N + static_cast<size_t>(b_idx) * N + b + 1];
        u[static_cast<size_t>(a) * N * N + static_cast<size_t>(b_idx) * N + (N - 1 - b)] = u[static_cast<size_t>(a) * N * N + static_cast<size_t>(b_idx) * N + (N - 2 - b)];
        u[static_cast<size_t>(a) * N * N + static_cast<size_t>(b) * N + b_idx] = u[static_cast<size_t>(a) * N * N + static_cast<size_t>(b + 1) * N + b_idx];
        u[static_cast<size_t>(a) * N * N + static_cast<size_t>(N - 1 - b) * N + b_idx] = u[static_cast<size_t>(a) * N * N + static_cast<size_t>(N - 2 - b) * N + b_idx];
        u[static_cast<size_t>(b) * N * N + static_cast<size_t>(a) * N + b_idx] = u[static_cast<size_t>(b + 1) * N * N + static_cast<size_t>(a) * N + b_idx];
        u[static_cast<size_t>(N - 1 - b) * N * N + static_cast<size_t>(a) * N + b_idx] = u[static_cast<size_t>(N - 2 - b) * N * N + static_cast<size_t>(a) * N + b_idx];
    }
}

__global__ void apply_neumann_bc_25d_zreg_async_comp(float* c, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;
    for (int b = R - 1; b >= 0; --b) {
        c[static_cast<size_t>(a) * N * N + static_cast<size_t>(b_idx) * N + b] = c[static_cast<size_t>(a) * N * N + static_cast<size_t>(b_idx) * N + b + 1];
        c[static_cast<size_t>(a) * N * N + static_cast<size_t>(b_idx) * N + (N - 1 - b)] = c[static_cast<size_t>(a) * N * N + static_cast<size_t>(b_idx) * N + (N - 2 - b)];
        c[static_cast<size_t>(a) * N * N + static_cast<size_t>(b) * N + b_idx] = c[static_cast<size_t>(a) * N * N + static_cast<size_t>(b + 1) * N + b_idx];
        c[static_cast<size_t>(a) * N * N + static_cast<size_t>(N - 1 - b) * N + b_idx] = c[static_cast<size_t>(a) * N * N + static_cast<size_t>(N - 2 - b) * N + b_idx];
        c[static_cast<size_t>(b) * N * N + static_cast<size_t>(a) * N + b_idx] = c[static_cast<size_t>(b + 1) * N * N + static_cast<size_t>(a) * N + b_idx];
        c[static_cast<size_t>(N - 1 - b) * N * N + static_cast<size_t>(a) * N + b_idx] = c[static_cast<size_t>(N - 2 - b) * N * N + static_cast<size_t>(a) * N + b_idx];
    }
}

StencilResult run_cuda_fp16_naive_3d_25d_zreg_async(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total_points = static_cast<size_t>(N) * N * N;
    size_t half_bytes = total_points * sizeof(__half);

    AsyncZRegTileConfig3D16 tile = query_fp16_zreg_async_tile(R, N, false);
    CUDA_CHECK_ZREG_ASYNC(cudaMemcpyToSymbol(d_coeffs_zreg_async, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK_ZREG_ASYNC(cudaMemcpyToSymbol(d_reach_zreg_async, &R, sizeof(int)));

    std::vector<float> h_f(total_points, cfg.temp_initial);
    int source_size = N / 8;
    int source_start = N / 2 - source_size / 2;
    for (int z = source_start; z < source_start + source_size; ++z)
        for (int y = source_start; y < source_start + source_size; ++y)
            for (int x = source_start; x < source_start + source_size; ++x)
                h_f[static_cast<size_t>(z) * N * N + static_cast<size_t>(y) * N + x] = cfg.temp_source;

    auto h_data = zreg_async_float_to_half_vec(h_f);
    __half* d_u = nullptr;
    __half* d_u_next = nullptr;
    CUDA_CHECK_ZREG_ASYNC(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK_ZREG_ASYNC(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK_ZREG_ASYNC(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_ZREG_ASYNC(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));

    int interior_z = N - 2 * R;
    if (interior_z <= 0) {
        fprintf(stderr, "error: grid too small for reach %d in 3D\n", R);
        exit(1);
    }
    int z_chunks = (interior_z + tile.z_chunk - 1) / tile.z_chunk;

    dim3 block(tile.tile_x, tile.tile_y);
    dim3 grid((N + tile.tile_x - 1) / tile.tile_x, (N + tile.tile_y - 1) / tile.tile_y, z_chunks);
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK_ZREG_ASYNC(cudaEventCreate(&start));
    CUDA_CHECK_ZREG_ASYNC(cudaEventCreate(&stop));
    CUDA_CHECK_ZREG_ASYNC(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; ++t) {
        heat3d_fp16_naive_25d_zreg_async_kernel<<<grid, block, tile.smem_bytes>>>(d_u, d_u_next, N, r, tile.tile_x, tile.tile_y, tile.z_chunk);
        CUDA_CHECK_ZREG_ASYNC(cudaGetLastError());
        apply_neumann_bc_25d_zreg_async_half<<<bc_grid, bc_block>>>(d_u_next, N, R);
        __half* tmp = d_u;
        d_u = d_u_next;
        d_u_next = tmp;
    }

    CUDA_CHECK_ZREG_ASYNC(cudaEventRecord(stop));
    CUDA_CHECK_ZREG_ASYNC(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK_ZREG_ASYNC(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK_ZREG_ASYNC(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> result_f(total_points);
    for (size_t i = 0; i < total_points; ++i) result_f[i] = __half2float(h_data[i]);

    double reads_per_point = 2.0 * 3.0 * R + 1.0;
    double interior = static_cast<double>(N - 2 * R);
    double bytes_per_step = interior * interior * interior * (reads_per_point + 1.0) * sizeof(__half);
    double bw = bytes_per_step * cfg.timesteps / (elapsed_ms / 1000.0) / 1e9;
    double mpts = (static_cast<double>(total_points) * cfg.timesteps) / (elapsed_ms / 1000.0) / 1e6;

    StencilResult res;
    res.variant_name = "cuda_fp16_naive_3d_25d_zreg_async";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * half_bytes;
    res.final_grid = result_f;

    CUDA_CHECK_ZREG_ASYNC(cudaFree(d_u));
    CUDA_CHECK_ZREG_ASYNC(cudaFree(d_u_next));
    CUDA_CHECK_ZREG_ASYNC(cudaEventDestroy(start));
    CUDA_CHECK_ZREG_ASYNC(cudaEventDestroy(stop));
    return res;
}

StencilResult run_cuda_fp16_kahan_3d_25d_zreg_async(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total_points = static_cast<size_t>(N) * N * N;
    size_t half_bytes = total_points * sizeof(__half);
    size_t float_bytes = total_points * sizeof(float);

    AsyncZRegTileConfig3D16 tile = query_fp16_zreg_async_tile(R, N, true);
    CUDA_CHECK_ZREG_ASYNC(cudaMemcpyToSymbol(d_coeffs_zreg_async, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK_ZREG_ASYNC(cudaMemcpyToSymbol(d_reach_zreg_async, &R, sizeof(int)));

    std::vector<float> h_f(total_points, cfg.temp_initial);
    int source_size = N / 8;
    int source_start = N / 2 - source_size / 2;
    for (int z = source_start; z < source_start + source_size; ++z)
        for (int y = source_start; y < source_start + source_size; ++y)
            for (int x = source_start; x < source_start + source_size; ++x)
                h_f[static_cast<size_t>(z) * N * N + static_cast<size_t>(y) * N + x] = cfg.temp_source;

    auto h_data = zreg_async_float_to_half_vec(h_f);
    std::vector<float> h_comp(total_points, 0.0f);

    __half* d_u = nullptr;
    __half* d_u_next = nullptr;
    float* d_c = nullptr;
    float* d_c_next = nullptr;
    CUDA_CHECK_ZREG_ASYNC(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK_ZREG_ASYNC(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK_ZREG_ASYNC(cudaMalloc(&d_c, float_bytes));
    CUDA_CHECK_ZREG_ASYNC(cudaMalloc(&d_c_next, float_bytes));
    CUDA_CHECK_ZREG_ASYNC(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_ZREG_ASYNC(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_ZREG_ASYNC(cudaMemcpy(d_c, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_ZREG_ASYNC(cudaMemcpy(d_c_next, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));

    int interior_z = N - 2 * R;
    if (interior_z <= 0) {
        fprintf(stderr, "error: grid too small for reach %d in 3D\n", R);
        exit(1);
    }
    int z_chunks = (interior_z + tile.z_chunk - 1) / tile.z_chunk;

    dim3 block(tile.tile_x, tile.tile_y);
    dim3 grid((N + tile.tile_x - 1) / tile.tile_x, (N + tile.tile_y - 1) / tile.tile_y, z_chunks);
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK_ZREG_ASYNC(cudaEventCreate(&start));
    CUDA_CHECK_ZREG_ASYNC(cudaEventCreate(&stop));
    CUDA_CHECK_ZREG_ASYNC(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; ++t) {
        heat3d_fp16_kahan_25d_zreg_async_kernel<<<grid, block, tile.smem_bytes>>>(d_u, d_u_next, d_c, d_c_next, N, r, tile.tile_x, tile.tile_y, tile.z_chunk);
        CUDA_CHECK_ZREG_ASYNC(cudaGetLastError());
        apply_neumann_bc_25d_zreg_async_half<<<bc_grid, bc_block>>>(d_u_next, N, R);
        apply_neumann_bc_25d_zreg_async_comp<<<bc_grid, bc_block>>>(d_c_next, N, R);
        __half* tmp_h = d_u;
        d_u = d_u_next;
        d_u_next = tmp_h;
        float* tmp_c = d_c;
        d_c = d_c_next;
        d_c_next = tmp_c;
    }

    CUDA_CHECK_ZREG_ASYNC(cudaEventRecord(stop));
    CUDA_CHECK_ZREG_ASYNC(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK_ZREG_ASYNC(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK_ZREG_ASYNC(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ZREG_ASYNC(cudaMemcpy(h_comp.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> result_f(total_points);
    for (size_t i = 0; i < total_points; ++i) result_f[i] = __half2float(h_data[i]) + h_comp[i];

    double reads_per_point = 2.0 * 3.0 * R + 1.0;
    double interior = static_cast<double>(N - 2 * R);
    double half_rw = (reads_per_point + 1.0) * sizeof(__half);
    double comp_rw = (reads_per_point + 1.0) * sizeof(float);
    double bytes_per_step = interior * interior * interior * (half_rw + comp_rw);
    double bw = bytes_per_step * cfg.timesteps / (elapsed_ms / 1000.0) / 1e9;
    double mpts = (static_cast<double>(total_points) * cfg.timesteps) / (elapsed_ms / 1000.0) / 1e6;

    StencilResult res;
    res.variant_name = "cuda_fp16_kahan_3d_25d_zreg_async";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * half_bytes + 2 * float_bytes;
    res.final_grid = result_f;

    CUDA_CHECK_ZREG_ASYNC(cudaFree(d_u));
    CUDA_CHECK_ZREG_ASYNC(cudaFree(d_u_next));
    CUDA_CHECK_ZREG_ASYNC(cudaFree(d_c));
    CUDA_CHECK_ZREG_ASYNC(cudaFree(d_c_next));
    CUDA_CHECK_ZREG_ASYNC(cudaEventDestroy(start));
    CUDA_CHECK_ZREG_ASYNC(cudaEventDestroy(stop));
    return res;
}
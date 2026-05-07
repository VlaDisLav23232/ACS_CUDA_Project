/*
 * 3D heat stencil: fp32 2.5D shared-memory tiling + Z register window.
 *
 * XY neighbors are served from shared memory. Z neighbors are kept in a
 * per-thread sliding register window, so each next Z layer needs only one new
 * global load for the thread's column.
 */

#include "stencil.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK_ZREG(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__constant__ float d_coeffs_zreg3d[MAX_REACH + 1];
__constant__ int d_reach_zreg3d;

struct ZRegTileConfig3D {
    int tile_x;
    int tile_y;
    int z_chunk;
    size_t smem_bytes;
};

static ZRegTileConfig3D query_fp32_zreg_tile(int R, int N, size_t elem_bytes) {
    cudaDeviceProp prop;
    CUDA_CHECK_ZREG(cudaGetDeviceProperties(&prop, 0));

    size_t smem_avail = prop.sharedMemPerBlock;
    int max_threads = prop.maxThreadsPerBlock;
    int max_regs = prop.regsPerBlock;
    int sm_count = prop.multiProcessorCount;
    int regs_per_thread = (2 * R + 1) + 32;

    static const int candidates[][2] = {
        {32, 32}, {32, 16}, {16, 32}, {32, 8}, {16, 16}, {16, 8}, {8, 16}, {8, 8}, {4, 4}
    };

    for (const auto& candidate : candidates) {
        int tx = candidate[0];
        int ty = candidate[1];
        int threads = tx * ty;
        if (threads > max_threads) continue;

        size_t smem = static_cast<size_t>(tx + 2 * R) * (ty + 2 * R) * elem_bytes;
        if (smem > smem_avail) continue;
        if (threads * regs_per_thread > max_regs) continue;

        int xy_blocks = ((N + tx - 1) / tx) * ((N + ty - 1) / ty);
        int interior_z = N - 2 * R;
        if (interior_z <= 0) continue;

        int target_blocks = sm_count * 4;
        int z_chunks = max(1, (target_blocks + xy_blocks - 1) / xy_blocks);
        z_chunks = min(z_chunks, interior_z);
        int z_chunk = (interior_z + z_chunks - 1) / z_chunks;

        printf("  3D fp32 2.5D+zreg tile: %dx%d XY, z_chunk=%d, smem=%zu bytes\n", tx, ty, z_chunk, smem);
        return {tx, ty, z_chunk, smem};
    }

    size_t fallback_smem = static_cast<size_t>(4 + 2 * R) * (4 + 2 * R) * elem_bytes;
    printf("  warning: fallback 4x4 2.5D+zreg tile\n");
    return {4, 4, 1, fallback_smem};
}

__global__ void heat3d_fp32_25d_zreg_kernel(
    const float* __restrict__ u,
    float* __restrict__ u_next,
    int N,
    float r,
    int tile_x,
    int tile_y,
    int z_chunk
) {
    const int R = d_reach_zreg3d;
    const int diam = 2 * R + 1;
    const int sw = tile_x + 2 * R;
    const int sh = tile_y + 2 * R;

    extern __shared__ float smem[];

    const int gx = blockIdx.x * tile_x + threadIdx.x;
    const int gy = blockIdx.y * tile_y + threadIdx.y;
    const int bx = blockIdx.x * tile_x - R;
    const int by = blockIdx.y * tile_y - R;

    const int gz_first = R + blockIdx.z * z_chunk;
    const int gz_last = min(N - R - 1, gz_first + z_chunk - 1);
    if (gz_first > gz_last) return;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;
    const int total_xy = sw * sh;
    const bool active = gx >= R && gx < N - R && gy >= R && gy < N - R;

    float z_buf[2 * MAX_REACH + 1];
    int cx = gx < N ? gx : N - 1;
    int cy = gy < N ? gy : N - 1;
    for (int m = 0; m < diam; ++m) {
        int gz = max(0, min(N - 1, gz_first - R + m));
        z_buf[m] = u[static_cast<size_t>(gz) * N * N + static_cast<size_t>(cy) * N + cx];
    }

    for (int gz = gz_first; gz <= gz_last; ++gz) {
        for (int idx = tid; idx < total_xy; idx += block_size) {
            int sy = idx / sw;
            int sx = idx % sw;
            int ci = max(0, min(N - 1, bx + sx));
            int cj = max(0, min(N - 1, by + sy));
            smem[idx] = u[static_cast<size_t>(gz) * N * N + static_cast<size_t>(cj) * N + ci];
        }
        __syncthreads();

        if (active) {
            int si = threadIdx.x + R;
            int sj = threadIdx.y + R;
            float center = z_buf[R];
            float lap = 3.0f * d_coeffs_zreg3d[0] * center;

            for (int m = 1; m <= R; ++m) {
                float xy = smem[sj * sw + si - m]
                         + smem[sj * sw + si + m]
                         + smem[(sj - m) * sw + si]
                         + smem[(sj + m) * sw + si];
                float zn = z_buf[R - m] + z_buf[R + m];
                lap += d_coeffs_zreg3d[m] * (xy + zn);
            }

            u_next[static_cast<size_t>(gz) * N * N + static_cast<size_t>(gy) * N + gx] = center + r * lap;
        }

        __syncthreads();
        for (int m = 0; m < diam - 1; ++m) z_buf[m] = z_buf[m + 1];
        int next_z = max(0, min(N - 1, gz + R + 1));
        z_buf[diam - 1] = u[static_cast<size_t>(next_z) * N * N + static_cast<size_t>(cy) * N + cx];
    }
}

__global__ void apply_neumann_bc_25d_zreg_fp32(float* u, int N, int R) {
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

StencilResult run_cuda_fp32_3d_25d_zreg(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total_points = static_cast<size_t>(N) * N * N;
    size_t grid_bytes = total_points * sizeof(float);

    ZRegTileConfig3D tile = query_fp32_zreg_tile(R, N, sizeof(float));

    CUDA_CHECK_ZREG(cudaMemcpyToSymbol(d_coeffs_zreg3d, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK_ZREG(cudaMemcpyToSymbol(d_reach_zreg3d, &R, sizeof(int)));

    std::vector<float> h_u(total_points, cfg.temp_initial);
    int source_size = N / 8;
    int source_start = N / 2 - source_size / 2;
    for (int z = source_start; z < source_start + source_size; ++z)
        for (int y = source_start; y < source_start + source_size; ++y)
            for (int x = source_start; x < source_start + source_size; ++x)
                h_u[static_cast<size_t>(z) * N * N + static_cast<size_t>(y) * N + x] = cfg.temp_source;

    float* d_u = nullptr;
    float* d_u_next = nullptr;
    CUDA_CHECK_ZREG(cudaMalloc(&d_u, grid_bytes));
    CUDA_CHECK_ZREG(cudaMalloc(&d_u_next, grid_bytes));
    CUDA_CHECK_ZREG(cudaMemcpy(d_u, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_ZREG(cudaMemcpy(d_u_next, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));

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
    CUDA_CHECK_ZREG(cudaEventCreate(&start));
    CUDA_CHECK_ZREG(cudaEventCreate(&stop));
    CUDA_CHECK_ZREG(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; ++t) {
        heat3d_fp32_25d_zreg_kernel<<<grid, block, tile.smem_bytes>>>(d_u, d_u_next, N, r, tile.tile_x, tile.tile_y, tile.z_chunk);
        CUDA_CHECK_ZREG(cudaGetLastError());
        apply_neumann_bc_25d_zreg_fp32<<<bc_grid, bc_block>>>(d_u_next, N, R);
        float* tmp = d_u;
        d_u = d_u_next;
        d_u_next = tmp;
    }

    CUDA_CHECK_ZREG(cudaEventRecord(stop));
    CUDA_CHECK_ZREG(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK_ZREG(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK_ZREG(cudaMemcpy(h_u.data(), d_u, grid_bytes, cudaMemcpyDeviceToHost));

    double reads_per_point = 2.0 * 3.0 * R + 1.0;
    double interior = static_cast<double>(N - 2 * R);
    double bytes_per_step = interior * interior * interior * (reads_per_point + 1.0) * sizeof(float);
    double bw = bytes_per_step * cfg.timesteps / (elapsed_ms / 1000.0) / 1e9;
    double mpts = (static_cast<double>(total_points) * cfg.timesteps) / (elapsed_ms / 1000.0) / 1e6;

    StencilResult res;
    res.variant_name = "cuda_fp32_3d_25d_zreg";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * grid_bytes;
    res.final_grid = h_u;

    CUDA_CHECK_ZREG(cudaFree(d_u));
    CUDA_CHECK_ZREG(cudaFree(d_u_next));
    CUDA_CHECK_ZREG(cudaEventDestroy(start));
    CUDA_CHECK_ZREG(cudaEventDestroy(stop));
    return res;
}

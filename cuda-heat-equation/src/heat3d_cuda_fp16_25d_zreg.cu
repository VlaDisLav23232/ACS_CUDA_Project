/*
 * 3D heat stencil: fp16 naive/Kahan 2.5D shared-memory tiling + Z register window.
 */

#include "stencil.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK_ZREG16(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__constant__ float d_coeffs_zreg3d16[MAX_REACH + 1];
__constant__ int d_reach_zreg3d16;

struct ZRegTileConfig3D16 {
    int tile_x;
    int tile_y;
    int z_chunk;
    size_t smem_bytes;
};

static ZRegTileConfig3D16 query_fp16_zreg_tile(int R, int N, size_t elem_bytes) {
    cudaDeviceProp prop;
    CUDA_CHECK_ZREG16(cudaGetDeviceProperties(&prop, 0));

    size_t smem_avail = prop.sharedMemPerBlock;
    int max_threads = prop.maxThreadsPerBlock;
    int max_regs = prop.regsPerBlock;
    int sm_count = prop.multiProcessorCount;
    int regs_per_thread = 2 * (2 * R + 1) + 32;

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

        printf("  3D fp16 2.5D+zreg tile: %dx%d XY, z_chunk=%d, smem=%zu bytes\n", tx, ty, z_chunk, smem);
        return {tx, ty, z_chunk, smem};
    }

    size_t fallback_smem = static_cast<size_t>(4 + 2 * R) * (4 + 2 * R) * elem_bytes;
    printf("  warning: fallback 4x4 fp16 2.5D+zreg tile\n");
    return {4, 4, 1, fallback_smem};
}

static std::vector<__half> zreg_float_to_half_vec(const std::vector<float>& values) {
    std::vector<__half> half_values(values.size());
    for (size_t i = 0; i < values.size(); ++i) half_values[i] = __float2half(values[i]);
    return half_values;
}

__global__ void heat3d_fp16_naive_25d_zreg_kernel(
    const __half* __restrict__ u,
    __half* __restrict__ u_next,
    int N,
    float r,
    int tile_x,
    int tile_y,
    int z_chunk
) {
    const int R = d_reach_zreg3d16;
    const int diam = 2 * R + 1;
    const int sw = tile_x + 2 * R;
    const int sh = tile_y + 2 * R;

    extern __shared__ __half smem_h[];

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
        z_buf[m] = __half2float(u[static_cast<size_t>(gz) * N * N + static_cast<size_t>(cy) * N + cx]);
    }

    for (int gz = gz_first; gz <= gz_last; ++gz) {
        for (int idx = tid; idx < total_xy; idx += block_size) {
            int sy = idx / sw;
            int sx = idx % sw;
            int ci = max(0, min(N - 1, bx + sx));
            int cj = max(0, min(N - 1, by + sy));
            smem_h[idx] = u[static_cast<size_t>(gz) * N * N + static_cast<size_t>(cj) * N + ci];
        }
        __syncthreads();

        if (active) {
            int si = threadIdx.x + R;
            int sj = threadIdx.y + R;
            float center = z_buf[R];
            float lap = 3.0f * d_coeffs_zreg3d16[0] * center;

            for (int m = 1; m <= R; ++m) {
                float xy = __half2float(smem_h[sj * sw + si - m])
                         + __half2float(smem_h[sj * sw + si + m])
                         + __half2float(smem_h[(sj - m) * sw + si])
                         + __half2float(smem_h[(sj + m) * sw + si]);
                float zn = z_buf[R - m] + z_buf[R + m];
                lap += d_coeffs_zreg3d16[m] * (xy + zn);
            }

            u_next[static_cast<size_t>(gz) * N * N + static_cast<size_t>(gy) * N + gx] = __float2half(center + r * lap);
        }

        __syncthreads();
        for (int m = 0; m < diam - 1; ++m) z_buf[m] = z_buf[m + 1];
        int next_z = max(0, min(N - 1, gz + R + 1));
        z_buf[diam - 1] = __half2float(u[static_cast<size_t>(next_z) * N * N + static_cast<size_t>(cy) * N + cx]);
    }
}

__global__ void heat3d_fp16_kahan_25d_zreg_kernel(
    const __half* __restrict__ u,
    __half* __restrict__ u_next,
    float* __restrict__ c,
    float* __restrict__ c_next,
    int N,
    float r,
    int tile_x,
    int tile_y,
    int z_chunk
) {
    const int R = d_reach_zreg3d16;
    const int diam = 2 * R + 1;
    const int sw = tile_x + 2 * R;
    const int sh = tile_y + 2 * R;
    const int tile_elems = sw * sh;

    extern __shared__ char smem_raw[];
    __half* s_u = reinterpret_cast<__half*>(smem_raw);
    size_t half_bytes = static_cast<size_t>(tile_elems) * sizeof(__half);
    size_t aligned = (half_bytes + 3) & ~static_cast<size_t>(3);
    float* s_c = reinterpret_cast<float*>(smem_raw + aligned);

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

    float z_buf[2 * MAX_REACH + 1];
    int cx = gx < N ? gx : N - 1;
    int cy = gy < N ? gy : N - 1;
    for (int m = 0; m < diam; ++m) {
        int gz = max(0, min(N - 1, gz_first - R + m));
        size_t idx = static_cast<size_t>(gz) * N * N + static_cast<size_t>(cy) * N + cx;
        z_buf[m] = __half2float(u[idx]) + c[idx];
    }

    for (int gz = gz_first; gz <= gz_last; ++gz) {
        for (int idx = tid; idx < tile_elems; idx += block_size) {
            int sy = idx / sw;
            int sx = idx % sw;
            int ci = max(0, min(N - 1, bx + sx));
            int cj = max(0, min(N - 1, by + sy));
            size_t gidx = static_cast<size_t>(gz) * N * N + static_cast<size_t>(cj) * N + ci;
            s_u[idx] = u[gidx];
            s_c[idx] = c[gidx];
        }
        __syncthreads();

        if (active) {
            int si = threadIdx.x + R;
            int sj = threadIdx.y + R;
            float center = z_buf[R];
            float lap = 3.0f * d_coeffs_zreg3d16[0] * center;

            for (int m = 1; m <= R; ++m) {
                int s_xm = sj * sw + si - m;
                int s_xp = sj * sw + si + m;
                int s_ym = (sj - m) * sw + si;
                int s_yp = (sj + m) * sw + si;
                float xy = (__half2float(s_u[s_xm]) + s_c[s_xm])
                         + (__half2float(s_u[s_xp]) + s_c[s_xp])
                         + (__half2float(s_u[s_ym]) + s_c[s_ym])
                         + (__half2float(s_u[s_yp]) + s_c[s_yp]);
                float zn = z_buf[R - m] + z_buf[R + m];
                lap += d_coeffs_zreg3d16[m] * (xy + zn);
            }

            float exact = center + r * lap;
            __half stored = __float2half(exact);
            size_t gidx = static_cast<size_t>(gz) * N * N + static_cast<size_t>(gy) * N + gx;
            u_next[gidx] = stored;
            volatile float stored_back = __half2float(stored);
            c_next[gidx] = exact - stored_back;
        }

        __syncthreads();
        for (int m = 0; m < diam - 1; ++m) z_buf[m] = z_buf[m + 1];
        int next_z = max(0, min(N - 1, gz + R + 1));
        size_t next_idx = static_cast<size_t>(next_z) * N * N + static_cast<size_t>(cy) * N + cx;
        z_buf[diam - 1] = __half2float(u[next_idx]) + c[next_idx];
    }
}

__global__ void apply_neumann_bc_25d_zreg_half(__half* u, int N, int R) {
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

__global__ void apply_neumann_bc_25d_zreg_comp(float* c, int N, int R) {
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

StencilResult run_cuda_fp16_naive_3d_25d_zreg(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total_points = static_cast<size_t>(N) * N * N;
    size_t half_bytes = total_points * sizeof(__half);

    ZRegTileConfig3D16 tile = query_fp16_zreg_tile(R, N, sizeof(__half));
    CUDA_CHECK_ZREG16(cudaMemcpyToSymbol(d_coeffs_zreg3d16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK_ZREG16(cudaMemcpyToSymbol(d_reach_zreg3d16, &R, sizeof(int)));

    std::vector<float> h_f(total_points, cfg.temp_initial);
    int source_size = N / 8;
    int source_start = N / 2 - source_size / 2;
    for (int z = source_start; z < source_start + source_size; ++z)
        for (int y = source_start; y < source_start + source_size; ++y)
            for (int x = source_start; x < source_start + source_size; ++x)
                h_f[static_cast<size_t>(z) * N * N + static_cast<size_t>(y) * N + x] = cfg.temp_source;

    auto h_data = zreg_float_to_half_vec(h_f);
    __half* d_u = nullptr;
    __half* d_u_next = nullptr;
    CUDA_CHECK_ZREG16(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK_ZREG16(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK_ZREG16(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_ZREG16(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));

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
    CUDA_CHECK_ZREG16(cudaEventCreate(&start));
    CUDA_CHECK_ZREG16(cudaEventCreate(&stop));
    CUDA_CHECK_ZREG16(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; ++t) {
        heat3d_fp16_naive_25d_zreg_kernel<<<grid, block, tile.smem_bytes>>>(d_u, d_u_next, N, r, tile.tile_x, tile.tile_y, tile.z_chunk);
        CUDA_CHECK_ZREG16(cudaGetLastError());
        apply_neumann_bc_25d_zreg_half<<<bc_grid, bc_block>>>(d_u_next, N, R);
        __half* tmp = d_u;
        d_u = d_u_next;
        d_u_next = tmp;
    }

    CUDA_CHECK_ZREG16(cudaEventRecord(stop));
    CUDA_CHECK_ZREG16(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK_ZREG16(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK_ZREG16(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    std::vector<float> result_f(total_points);
    for (size_t i = 0; i < total_points; ++i) result_f[i] = __half2float(h_data[i]);

    double reads_per_point = 2.0 * 3.0 * R + 1.0;
    double interior = static_cast<double>(N - 2 * R);
    double bytes_per_step = interior * interior * interior * (reads_per_point + 1.0) * sizeof(__half);
    double bw = bytes_per_step * cfg.timesteps / (elapsed_ms / 1000.0) / 1e9;
    double mpts = (static_cast<double>(total_points) * cfg.timesteps) / (elapsed_ms / 1000.0) / 1e6;

    StencilResult res;
    res.variant_name = "cuda_fp16_naive_3d_25d_zreg";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * half_bytes;
    res.final_grid = result_f;

    CUDA_CHECK_ZREG16(cudaFree(d_u));
    CUDA_CHECK_ZREG16(cudaFree(d_u_next));
    CUDA_CHECK_ZREG16(cudaEventDestroy(start));
    CUDA_CHECK_ZREG16(cudaEventDestroy(stop));
    return res;
}

StencilResult run_cuda_fp16_kahan_3d_25d_zreg(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total_points = static_cast<size_t>(N) * N * N;
    size_t half_bytes = total_points * sizeof(__half);
    size_t float_bytes = total_points * sizeof(float);

    size_t smem_per_elem = sizeof(__half) + sizeof(float);
    ZRegTileConfig3D16 tile = query_fp16_zreg_tile(R, N, smem_per_elem);
    CUDA_CHECK_ZREG16(cudaMemcpyToSymbol(d_coeffs_zreg3d16, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK_ZREG16(cudaMemcpyToSymbol(d_reach_zreg3d16, &R, sizeof(int)));

    std::vector<float> h_f(total_points, cfg.temp_initial);
    int source_size = N / 8;
    int source_start = N / 2 - source_size / 2;
    for (int z = source_start; z < source_start + source_size; ++z)
        for (int y = source_start; y < source_start + source_size; ++y)
            for (int x = source_start; x < source_start + source_size; ++x)
                h_f[static_cast<size_t>(z) * N * N + static_cast<size_t>(y) * N + x] = cfg.temp_source;

    auto h_data = zreg_float_to_half_vec(h_f);
    std::vector<float> h_comp(total_points, 0.0f);

    __half* d_u = nullptr;
    __half* d_u_next = nullptr;
    float* d_c = nullptr;
    float* d_c_next = nullptr;
    CUDA_CHECK_ZREG16(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK_ZREG16(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK_ZREG16(cudaMalloc(&d_c, float_bytes));
    CUDA_CHECK_ZREG16(cudaMalloc(&d_c_next, float_bytes));
    CUDA_CHECK_ZREG16(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_ZREG16(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_ZREG16(cudaMemcpy(d_c, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_ZREG16(cudaMemcpy(d_c_next, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));

    int interior_z = N - 2 * R;
    if (interior_z <= 0) {
        fprintf(stderr, "error: grid too small for reach %d in 3D\n", R);
        exit(1);
    }
    int z_chunks = (interior_z + tile.z_chunk - 1) / tile.z_chunk;

    dim3 block(tile.tile_x, tile.tile_y);
    dim3 grid((N + tile.tile_x - 1) / tile.tile_x, (N + tile.tile_y - 1) / tile.tile_y, z_chunks);
    int tile_elems = (tile.tile_x + 2 * R) * (tile.tile_y + 2 * R);
    size_t h_tile = static_cast<size_t>(tile_elems) * sizeof(__half);
    size_t h_align = (h_tile + 3) & ~static_cast<size_t>(3);
    size_t smem_actual = h_align + static_cast<size_t>(tile_elems) * sizeof(float);
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK_ZREG16(cudaEventCreate(&start));
    CUDA_CHECK_ZREG16(cudaEventCreate(&stop));
    CUDA_CHECK_ZREG16(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; ++t) {
        heat3d_fp16_kahan_25d_zreg_kernel<<<grid, block, smem_actual>>>(d_u, d_u_next, d_c, d_c_next, N, r, tile.tile_x, tile.tile_y, tile.z_chunk);
        CUDA_CHECK_ZREG16(cudaGetLastError());
        apply_neumann_bc_25d_zreg_half<<<bc_grid, bc_block>>>(d_u_next, N, R);
        apply_neumann_bc_25d_zreg_comp<<<bc_grid, bc_block>>>(d_c_next, N, R);
        __half* tmp_h = d_u;
        d_u = d_u_next;
        d_u_next = tmp_h;
        float* tmp_c = d_c;
        d_c = d_c_next;
        d_c_next = tmp_c;
    }

    CUDA_CHECK_ZREG16(cudaEventRecord(stop));
    CUDA_CHECK_ZREG16(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK_ZREG16(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK_ZREG16(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ZREG16(cudaMemcpy(h_comp.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));
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
    res.variant_name = "cuda_fp16_kahan_3d_25d_zreg";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * half_bytes + 2 * float_bytes;
    res.final_grid = result_f;

    CUDA_CHECK_ZREG16(cudaFree(d_u));
    CUDA_CHECK_ZREG16(cudaFree(d_u_next));
    CUDA_CHECK_ZREG16(cudaFree(d_c));
    CUDA_CHECK_ZREG16(cudaFree(d_c_next));
    CUDA_CHECK_ZREG16(cudaEventDestroy(start));
    CUDA_CHECK_ZREG16(cudaEventDestroy(stop));
    return res;
}

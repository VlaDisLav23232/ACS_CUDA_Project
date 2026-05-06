//  WHAT IS 2.5D BLOCKING?
//  The naive 3D kernel reads all 6 neighbors from slow global memory.
//  2.5D blocking loads a 2D XY tile (+border) into FAST shared memory.
//  X and Y neighbors -> shared memory (fast).
//  Z neighbors       -> global memory / L2 cache (not tiled).
//
//  Each thread block handles a TILE_X * TILE_Y patch of the XY plane.
//  It loops over ALL Z layers, processing one layer at a time.
//  For each layer: load tile+halo into shared memory, compute, write back.
//
//  WHY NOT TILE ALL 3 DIMENSIONS?
//  A tile of (16+2) x (16+2) x (16+2) floats = 46,656 bytes.
//  Shared memory is only 48KB. We'd barely fit one tile and have
//  zero room for anything else. So we tile XY only

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

// Tile dimensions for the XY plane.
// Each thread block will be TILE_X * TILE_Y threads.
// TILE must be big enough to amortize the halo overhead.
// 16*16 = 256 threads per block
static const int TILE_X = 16;
static const int TILE_Y = 16;

// Stencil coefficients stored in constant memory (broadcast to all threads for free)
__constant__ float d_coeffs_25d[MAX_REACH + 1];
__constant__ int   d_reach_25d;

//  2.5D fp32 kernel
//
//  How it works:
//  - Block (bx, by, bz) handles one XY tile at z-layer bz.
//  - It loads the XY tile + R-wide halo into shared memory.
//  - XY neighbors come from shared memory (fast).
//  - Z neighbors come from global memory (L2 cache helps).
//  - This gives full parallelism: blocks cover the entire 3D grid.
__global__ void heat3d_fp32_25d_kernel(const float* __restrict__ u,
                                        float* __restrict__ u_next,
                                        int N, float r) {
    int R = d_reach_25d;

    // This thread's global (x, y, z) position
    int tile_x = blockIdx.x * TILE_X + threadIdx.x;
    int tile_y = blockIdx.y * TILE_Y + threadIdx.y;
    int z = blockIdx.z;  // one block per Z layer

    // Shared memory for the XY tile + halo
    extern __shared__ float smem[];
    int smem_w = TILE_X + 2 * R;
    int smem_size = smem_w * (TILE_Y + 2 * R);
    int tid = threadIdx.y * TILE_X + threadIdx.x;
    int n_threads = TILE_X * TILE_Y;

    // Load tile + halo into shared memory
    for (int idx = tid; idx < smem_size; idx += n_threads) {
        int sy = idx / smem_w;
        int sx = idx % smem_w;
        int gx = blockIdx.x * TILE_X + sx - R;
        int gy = blockIdx.y * TILE_Y + sy - R;
        gx = max(0, min(gx, N - 1));
        gy = max(0, min(gy, N - 1));
        smem[idx] = u[(size_t)z * N * N + gy * N + gx];
    }
    __syncthreads();

    // Compute stencil
    if (tile_x >= R && tile_x < N - R &&
        tile_y >= R && tile_y < N - R &&
        z >= R && z < N - R) {

        int lx = threadIdx.x + R;
        int ly = threadIdx.y + R;

        float center = smem[ly * smem_w + lx];
        float lap = 3.0f * d_coeffs_25d[0] * center;

        for (int m = 1; m <= R; m++) {
            // XY from shared memory (fast)
            float xm = smem[ly * smem_w + (lx - m)];
            float xp = smem[ly * smem_w + (lx + m)];
            float ym = smem[(ly - m) * smem_w + lx];
            float yp = smem[(ly + m) * smem_w + lx];

            // Z from global memory (L2 cache helps)
            size_t idx_center = (size_t)z * N * N + tile_y * N + tile_x;
            float zm = u[idx_center - (size_t)m * N * N];
            float zp = u[idx_center + (size_t)m * N * N];

            lap += d_coeffs_25d[m] * (xm + xp + ym + yp + zm + zp);
        }

        u_next[(size_t)z * N * N + tile_y * N + tile_x] = center + r * lap;
    }
}

//  Boundary condition kernels - same as naive versions
__global__ void apply_neumann_bc_3d_25d(float* u, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;

    for (int b = R - 1; b >= 0; b--) {
        // x faces
        u[(size_t)a * N * N + b_idx * N + b]       = u[(size_t)a * N * N + b_idx * N + (b + 1)];
        u[(size_t)a * N * N + b_idx * N + (N-1-b)] = u[(size_t)a * N * N + b_idx * N + (N-2-b)];
        // y faces
        u[(size_t)a * N * N + b * N + b_idx]       = u[(size_t)a * N * N + (b + 1) * N + b_idx];
        u[(size_t)a * N * N + (N-1-b) * N + b_idx] = u[(size_t)a * N * N + (N-2-b) * N + b_idx];
        // z faces
        u[(size_t)b * N * N + a * N + b_idx]       = u[(size_t)(b + 1) * N * N + a * N + b_idx];
        u[(size_t)(N-1-b) * N * N + a * N + b_idx] = u[(size_t)(N-2-b) * N * N + a * N + b_idx];
    }
}

// Same heat source kernel
__global__ void apply_heat_source_3d_25d(float* u, int N, int src_start, int src_size, float temp_source) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= src_start && x < src_start + src_size &&
        y >= src_start && y < src_start + src_size &&
        z >= src_start && z < src_start + src_size) {
        size_t idx = (size_t)z * N * N + y * N + x;
        u[idx] = temp_source;
    }
}

//  Runner function for fp32 2.5D
StencilResult run_cuda_fp32_3d_25d(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total = (size_t)N * N * N;
    size_t grid_bytes = total * sizeof(float);

    // Copy stencil coefficients to GPU constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_25d, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_25d, &R, sizeof(int)));

    // Initialize grid: everything cold, hot cube in the center
    std::vector<float> h_u(total, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int z = src_start; z < src_start + src_size; z++)
        for (int y = src_start; y < src_start + src_size; y++)
            for (int x = src_start; x < src_start + src_size; x++)
                h_u[(size_t)z * N * N + y * N + x] = cfg.temp_source;

    // Allocate GPU memory: two grids (current and next)
    float *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, grid_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));

    // Block = TILE_X * TILE_Y threads (one thread per XY point in the tile).
    // Grid = enough blocks to cover the XY plane * one block per Z layer.
    dim3 block(TILE_X, TILE_Y);
    dim3 grid25((N + TILE_X - 1) / TILE_X, (N + TILE_Y - 1) / TILE_Y, N);

    // Shared memory size: (TILE + 2*R)^2 floats for the XY halo tile
    size_t smem_bytes = (TILE_X + 2 * R) * (TILE_Y + 2 * R) * sizeof(float);

    // BC grid (for boundary condition kernel)
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    // Heat source grid
    dim3 src_block(8, 8, 8);
    dim3 src_grid((src_size + 7) / 8, (src_size + 7) / 8, (src_size + 7) / 8);

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // The main time loop: one kernel launch per timestep
    for (int t = 0; t < cfg.timesteps; t++) {
        // Each block handles one XY tile at one Z layer.
        // Full 3D grid launch - maximum parallelism.
        heat3d_fp32_25d_kernel<<<grid25, block, smem_bytes>>>(d_u, d_u_next, N, r);

        // Apply boundary conditions (separate simple kernel)
        apply_neumann_bc_3d_25d<<<bc_grid, bc_block>>>(d_u_next, N, R);

        // Re-apply constant heat source
        apply_heat_source_3d_25d<<<src_grid, src_block>>>(d_u_next, N, src_start, src_size, cfg.temp_source);

        // Swap pointers: now d_u points to the freshly computed grid
        float* tmp = d_u; d_u = d_u_next; d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Copy result back to CPU
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, grid_bytes, cudaMemcpyDeviceToHost));

    double bytes_per_step = 2.0 * (double)N * N * N * sizeof(float);
    double total_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;
    double mpts = ((double)N * N * N * cfg.timesteps) / (elapsed_ms / 1000.0) / 1e6;

    StencilResult res;
    res.variant_name = "cuda_fp32_3d_25d";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.megapoints_per_sec = mpts;
    res.memory_bytes = 2 * grid_bytes;
    res.final_grid = h_u;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

//  2.5D fp16 + Kahan kernel
//
//  Same structure as fp32 version, but:
//  - Data stored in fp16 (__half), all math done in fp32 (float)
//  - Compensation array c[] tracks rounding errors
//  - Every value is RECONSTRUCTED: half2float(u) + c
__global__ void heat3d_fp16_kahan_25d_kernel(const __half* __restrict__ u,
                                              __half* __restrict__ u_next,
                                              const float* __restrict__ c,
                                              float* __restrict__ c_next,
                                              int N, float r) {
    int R = d_reach_25d;
    int tile_x = blockIdx.x * TILE_X + threadIdx.x;
    int tile_y = blockIdx.y * TILE_Y + threadIdx.y;
    int z = blockIdx.z;

    int smem_w = TILE_X + 2 * R;
    int smem_size = smem_w * (TILE_Y + 2 * R);

    extern __shared__ char smem_raw[];
    __half* smem_u = (__half*)smem_raw;
    float* smem_c = (float*)(smem_raw + smem_size * sizeof(__half));

    int tid = threadIdx.y * TILE_X + threadIdx.x;
    int n_threads = TILE_X * TILE_Y;

    // Load tile + halo for both u[] and c[]
    for (int idx = tid; idx < smem_size; idx += n_threads) {
        int sy = idx / smem_w;
        int sx = idx % smem_w;
        int gx = blockIdx.x * TILE_X + sx - R;
        int gy = blockIdx.y * TILE_Y + sy - R;
        gx = max(0, min(gx, N - 1));
        gy = max(0, min(gy, N - 1));
        size_t gidx = (size_t)z * N * N + gy * N + gx;
        smem_u[idx] = u[gidx];
        smem_c[idx] = c[gidx];
    }
    __syncthreads();

    if (tile_x >= R && tile_x < N - R &&
        tile_y >= R && tile_y < N - R &&
        z >= R && z < N - R) {

        int lx = threadIdx.x + R;
        int ly = threadIdx.y + R;

        float center = __half2float(smem_u[ly * smem_w + lx]) + smem_c[ly * smem_w + lx];
        float lap = 3.0f * d_coeffs_25d[0] * center;

        for (int m = 1; m <= R; m++) {
            float xm = __half2float(smem_u[ly * smem_w + (lx - m)]) + smem_c[ly * smem_w + (lx - m)];
            float xp = __half2float(smem_u[ly * smem_w + (lx + m)]) + smem_c[ly * smem_w + (lx + m)];
            float ym = __half2float(smem_u[(ly - m) * smem_w + lx]) + smem_c[(ly - m) * smem_w + lx];
            float yp = __half2float(smem_u[(ly + m) * smem_w + lx]) + smem_c[(ly + m) * smem_w + lx];

            size_t idx_center = (size_t)z * N * N + tile_y * N + tile_x;
            float zm_val = __half2float(u[idx_center - (size_t)m * N * N])
                         + c[idx_center - (size_t)m * N * N];
            float zp_val = __half2float(u[idx_center + (size_t)m * N * N])
                         + c[idx_center + (size_t)m * N * N];

            lap += d_coeffs_25d[m] * (xm + xp + ym + yp + zm_val + zp_val);
        }

        float exact_result = center + r * lap;
        __half stored = __float2half(exact_result);
        size_t out_idx = (size_t)z * N * N + tile_y * N + tile_x;
        u_next[out_idx] = stored;
        volatile float stored_back = __half2float(stored);
        c_next[out_idx] = exact_result - stored_back;
    }
}

// Boundary condition kernels for fp16 (same as naive versions)
__global__ void apply_neumann_bc_3d_25d_fp16(__half* u, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;

    for (int b = R - 1; b >= 0; b--) {
        u[(size_t)a * N * N + b_idx * N + b]       = u[(size_t)a * N * N + b_idx * N + (b + 1)];
        u[(size_t)a * N * N + b_idx * N + (N-1-b)] = u[(size_t)a * N * N + b_idx * N + (N-2-b)];
        u[(size_t)a * N * N + b * N + b_idx]       = u[(size_t)a * N * N + (b + 1) * N + b_idx];
        u[(size_t)a * N * N + (N-1-b) * N + b_idx] = u[(size_t)a * N * N + (N-2-b) * N + b_idx];
        u[(size_t)b * N * N + a * N + b_idx]       = u[(size_t)(b + 1) * N * N + a * N + b_idx];
        u[(size_t)(N-1-b) * N * N + a * N + b_idx] = u[(size_t)(N-2-b) * N * N + a * N + b_idx];
    }
}

__global__ void apply_neumann_bc_3d_25d_comp(float* c, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b_idx >= N) return;

    for (int b = R - 1; b >= 0; b--) {
        c[(size_t)a * N * N + b_idx * N + b]       = c[(size_t)a * N * N + b_idx * N + (b + 1)];
        c[(size_t)a * N * N + b_idx * N + (N-1-b)] = c[(size_t)a * N * N + b_idx * N + (N-2-b)];
        c[(size_t)a * N * N + b * N + b_idx]       = c[(size_t)a * N * N + (b + 1) * N + b_idx];
        c[(size_t)a * N * N + (N-1-b) * N + b_idx] = c[(size_t)a * N * N + (N-2-b) * N + b_idx];
        c[(size_t)b * N * N + a * N + b_idx]       = c[(size_t)(b + 1) * N * N + a * N + b_idx];
        c[(size_t)(N-1-b) * N * N + a * N + b_idx] = c[(size_t)(N-2-b) * N * N + a * N + b_idx];
    }
}

//  Runner function for fp16 + Kahan 2.5D
static std::vector<__half> float_to_half_25d(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); i++)
        h[i] = __float2half(f[i]);
    return h;
}

StencilResult run_cuda_fp16_kahan_3d_25d(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t total = (size_t)N * N * N;
    size_t half_bytes = total * sizeof(__half);
    size_t float_bytes = total * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_25d, cfg.fd_coeffs, (R + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_reach_25d, &R, sizeof(int)));

    std::vector<float> h_f(total, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int z = src_start; z < src_start + src_size; z++)
        for (int y = src_start; y < src_start + src_size; y++)
            for (int x = src_start; x < src_start + src_size; x++)
                h_f[(size_t)z * N * N + y * N + x] = cfg.temp_source;

    auto h_data = float_to_half_25d(h_f);
    std::vector<float> h_comp(total, 0.0f);

    __half *d_u, *d_u_next;
    float *d_c, *d_c_next;
    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_c_next, float_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_next, h_comp.data(), float_bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_X, TILE_Y);
    dim3 grid25((N + TILE_X - 1) / TILE_X, (N + TILE_Y - 1) / TILE_Y, N);

    // Shared memory: fp16 tile + fp32 compensation tile
    int smem_tile = (TILE_X + 2 * R) * (TILE_Y + 2 * R);
    size_t smem_bytes = smem_tile * sizeof(__half) + smem_tile * sizeof(float);

    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat3d_fp16_kahan_25d_kernel<<<grid25, block, smem_bytes>>>(
            d_u, d_u_next, d_c, d_c_next, N, r);
        apply_neumann_bc_3d_25d_fp16<<<bc_grid, bc_block>>>(d_u_next, N, R);
        apply_neumann_bc_3d_25d_comp<<<bc_grid, bc_block>>>(d_c_next, N, R);

        __half* tmp_h = d_u; d_u = d_u_next; d_u_next = tmp_h;
        float* tmp_c = d_c; d_c = d_c_next; d_c_next = tmp_c;
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
    res.variant_name = "cuda_fp16_kahan_3d_25d";
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

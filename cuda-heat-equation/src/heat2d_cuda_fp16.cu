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

static const int BLOCK_X = 16;
static const int BLOCK_Y = 16;

// fp16 storage, fp32 compute, NO Kahan
// the idea: load __half from memory (saves bandwidth), convert to float for math,
// then convert back to __half for storage.
// problem: every timestep we lose bits when converting float result back to half
__global__ void heat2d_fp16_naive_kernel(const __half* __restrict__ u,
                                          __half* __restrict__ u_next,
                                          int N, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
        float center = __half2float(u[j * N + i]);
        float left   = __half2float(u[j * N + (i - 1)]);
        float right  = __half2float(u[j * N + (i + 1)]);
        float up     = __half2float(u[(j - 1) * N + i]);
        float down   = __half2float(u[(j + 1) * N + i]);

        float result = center + r * (left + right + up + down - 4.0f * center);
        u_next[j * N + i] = __float2half(result);
    }
}

// fp16 storage, fp32 compute, WITH Kahan compensated summation
// key insight: we keep a separate fp32 compensation array c[] that tracks
// the error accumulated from half->float->half roundtrips.
// each step: we subtract the previous compensation before computing,
// then figure out how much got lost and store that for next time
__global__ void heat2d_fp16_kahan_kernel(const __half* __restrict__ u,
                                          __half* __restrict__ u_next,
                                          float* __restrict__ c,
                                          float* __restrict__ c_next,
                                          int N, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
        int idx = j * N + i;

        // load from fp16 storage and add back the compensation
        float center = __half2float(u[idx])       + c[idx];
        float left   = __half2float(u[j*N+(i-1)]) + c[j*N+(i-1)];
        float right  = __half2float(u[j*N+(i+1)]) + c[j*N+(i+1)];
        float up     = __half2float(u[(j-1)*N+i]) + c[(j-1)*N+i];
        float down   = __half2float(u[(j+1)*N+i]) + c[(j+1)*N+i];

        // FTCS stencil update in full fp32
        float exact_result = center + r * (left + right + up + down - 4.0f * center);

        // store the result as fp16 (lossy!)
        __half stored = __float2half(exact_result);
        u_next[idx] = stored;

        // the compensation is: what we wanted to store minus what actually got stored
        // this is the Kahan trick - we track the rounding error
        volatile float stored_back = __half2float(stored);
        c_next[idx] = exact_result - stored_back;
    }
}

__global__ void apply_neumann_bc_fp16(__half* u, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        u[0 * N + idx] = u[1 * N + idx];
        u[(N-1) * N + idx] = u[(N-2) * N + idx];
        u[idx * N + 0] = u[idx * N + 1];
        u[idx * N + (N-1)] = u[idx * N + (N-2)];
    }
}

__global__ void apply_neumann_bc_comp(float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[0 * N + idx] = c[1 * N + idx];
        c[(N-1) * N + idx] = c[(N-2) * N + idx];
        c[idx * N + 0] = c[idx * N + 1];
        c[idx * N + (N-1)] = c[idx * N + (N-2)];
    }
}

// helper to convert float array to half on host
static std::vector<__half> float_to_half(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); i++)
        h[i] = __float2half(f[i]);
    return h;
}

StencilResult run_cuda_fp16_naive(const StencilConfig& cfg) {
    int N = cfg.nx;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t n_elems = N * N;
    size_t half_bytes = n_elems * sizeof(__half);

    std::vector<float> h_f(n_elems, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++)
        for (int i = src_start; i < src_start + src_size; i++)
            h_f[j * N + i] = cfg.temp_source;

    auto h_data = float_to_half(h_f);

    __half *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, half_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_data.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_data.data(), half_bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);
    int bc_threads = 256;
    int bc_blocks = (N + bc_threads - 1) / bc_threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat2d_fp16_naive_kernel<<<grid, block>>>(d_u, d_u_next, N, r);
        apply_neumann_bc_fp16<<<bc_blocks, bc_threads>>>(d_u_next, N);
        __half* tmp = d_u; d_u = d_u_next; d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> result_f(n_elems);
    for (size_t i = 0; i < n_elems; i++)
        result_f[i] = __half2float(h_data[i]);

    double bytes_per_step = (double)(N - 2) * (N - 2) * 6 * sizeof(__half);
    double total_bytes = bytes_per_step * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp16_naive";
    res.grid_size = N;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    res.memory_bytes = 2 * half_bytes;
    res.final_grid = result_f;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return res;
}

StencilResult run_cuda_fp16_kahan(const StencilConfig& cfg) {
    int N = cfg.nx;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    size_t n_elems = N * N;
    size_t half_bytes = n_elems * sizeof(__half);
    size_t float_bytes = n_elems * sizeof(float);

    std::vector<float> h_f(n_elems, cfg.temp_initial);
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++)
        for (int i = src_start; i < src_start + src_size; i++)
            h_f[j * N + i] = cfg.temp_source;

    auto h_data = float_to_half(h_f);
    std::vector<float> h_comp(n_elems, 0.0f); // compensation starts at zero

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

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);
    int bc_threads = 256;
    int bc_blocks = (N + bc_threads - 1) / bc_threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.timesteps; t++) {
        heat2d_fp16_kahan_kernel<<<grid, block>>>(d_u, d_u_next, d_c, d_c_next, N, r);
        apply_neumann_bc_fp16<<<bc_blocks, bc_threads>>>(d_u_next, N);
        apply_neumann_bc_comp<<<bc_blocks, bc_threads>>>(d_c_next, N);
        // swap both grid and compensation pointers
        __half* tmp_h = d_u; d_u = d_u_next; d_u_next = tmp_h;
        float* tmp_c = d_c; d_c = d_c_next; d_c_next = tmp_c;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_u, half_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));

    // reconstruct full-precision result: stored_half + compensation
    std::vector<float> result_f(n_elems);
    for (size_t i = 0; i < n_elems; i++)
        result_f[i] = __half2float(h_data[i]) + h_comp[i];

    // bandwidth: reads 5 halfs + 5 floats (compensation), writes 1 half + 1 float
    double read_bytes = (double)(N-2)*(N-2) * (5*sizeof(__half) + 5*sizeof(float));
    double write_bytes = (double)(N-2)*(N-2) * (sizeof(__half) + sizeof(float));
    double total_bytes = (read_bytes + write_bytes) * cfg.timesteps;
    double bw = total_bytes / (elapsed_ms / 1000.0) / 1e9;

    StencilResult res;
    res.variant_name = "cuda_fp16_kahan";
    res.grid_size = N;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = elapsed_ms;
    res.effective_bw_gbs = bw;
    // extra memory for compensation arrays
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

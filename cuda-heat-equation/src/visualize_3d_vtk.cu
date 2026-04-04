#include "stencil.h"
#include "vtk_export.h"
#include "fd_coefficients.h"
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__constant__ float d_fd_coeffs[MAX_REACH + 1];
__constant__ int d_fd_reach;
__global__ void heat3d_kernel(const float* __restrict__ u, float* __restrict__ u_next, int N, float r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_fd_reach;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = (size_t)z * N * N + y * N + x;
        float center = u[idx];
        float lap = 3.0f * d_fd_coeffs[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_fd_coeffs[m] * (u[idx - m] + u[idx + m]
                                   + u[idx - (size_t)m * N] + u[idx + (size_t)m * N]
                                   + u[idx - (size_t)m * N * N] + u[idx + (size_t)m * N * N]);
        }
        u_next[idx] = center + r * lap;
    }
}

__global__ void neumann_bc_kernel(float* u, int N, int R) {
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

__global__ void heat_source_kernel(float* u, int N, int src_start, int src_size, float temp_source) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + src_start;
    int y = blockIdx.y * blockDim.y + threadIdx.y + src_start;
    int z = blockIdx.z * blockDim.z + threadIdx.z + src_start;
    
    if (x < src_start + src_size && y < src_start + src_size && z < src_start + src_size) {
        size_t idx = (size_t)z * N * N + y * N + x;
        u[idx] = temp_source;
    }
}

struct HeatSource {
    int x, y, z, size;
    float temp;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <mode> [N] [timesteps] [save_interval] [reach]\n", argv[0]);
        printf("\nModes:\n");
        printf("  center - Single heat source at center\n");
        printf("  random - Multiple small random heat sources\n");
        printf("\nExamples:\n");
        printf("  %s center 64 5000 500 2\n", argv[0]);
        printf("  %s random 64 5000 500 2\n", argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    
    if (mode != "center" && mode != "random") {
        printf("ERROR: Invalid mode '%s'\n", mode.c_str());
        printf("Mode must be 'center' or 'random'\n\n");
        printf("Usage: %s <mode> [N] [timesteps] [save_interval] [reach]\n", argv[0]);
        printf("\nExamples:\n");
        printf("  %s center 64 5000 500 2\n", argv[0]);
        printf("  %s random 64 5000 500 2\n", argv[0]);
        return 1;
    }
    
    int N = 64, timesteps = 100, save_interval = 10, reach = 1;
    
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) timesteps = std::atoi(argv[3]);
    if (argc > 4) save_interval = std::atoi(argv[4]);
    if (argc > 5) reach = std::atoi(argv[5]);
    
    if (N < 8 || N > 512) {
        printf("ERROR: Grid size N=%d out of range [8, 512]\n", N);
        return 1;
    }
    
    printf("3D Heat Equation Simulation\n");
    printf("Mode: %s\n", mode.c_str());
    printf("Grid: %dx%dx%d\n", N, N, N);
    printf("Timesteps: %d, Save every: %d, Reach: %d\n\n", timesteps, save_interval, reach);
    
    float dx = 1.0f / N;
    float k = 0.1f;
    float temp_initial = 20.0f;
    float temp_source = 100.0f;
    
    FDCoefficients fd = compute_fd_coefficients(reach, 3);
    float stability_limit = static_cast<float>(fd.stability_limit);
    float dt = 0.8f * stability_limit * dx * dx / k / 3.0f;
    float r = k * dt / (dx * dx);
    
    size_t total = (size_t)N * N * N;
    size_t grid_bytes = total * sizeof(float);
    std::vector<float> h_u(total, temp_initial);
    
    std::vector<HeatSource> sources;
    
    if (mode == "center") {
        int src_size = N / 8;
        int src_start = N / 2 - src_size / 2;
        sources.push_back({src_start, src_start, src_start, src_size, temp_source});
        
        for (int z = src_start; z < src_start + src_size; z++)
            for (int y = src_start; y < src_start + src_size; y++)
                for (int x = src_start; x < src_start + src_size; x++)
                    h_u[(size_t)z * N * N + y * N + x] = temp_source;
        
        printf("Center heat source: %dx%dx%d at (%d,%d,%d)\n\n", 
               src_size, src_size, src_size, src_start, src_start, src_start);
    } 
    else {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist_pos(N/4, 3*N/4);
        
        int num_sources = 5 + (N / 32);
        int src_size = std::max(2, N / 16);
        
        for (int i = 0; i < num_sources; i++) {
            int sx = dist_pos(rng) - src_size/2;
            int sy = dist_pos(rng) - src_size/2;
            int sz = dist_pos(rng) - src_size/2;
            
            sources.push_back({sx, sy, sz, src_size, temp_source});
            
            for (int z = sz; z < sz + src_size && z < N; z++)
                for (int y = sy; y < sy + src_size && y < N; y++)
                    for (int x = sx; x < sx + src_size && x < N; x++)
                        if (x >= 0 && y >= 0 && z >= 0)
                            h_u[(size_t)z * N * N + y * N + x] = temp_source;
        }
        
        printf("%d random heat sources, size: %dx%dx%d each\n\n", 
               num_sources, src_size, src_size, src_size);
    }
    
    float fd_coeffs[MAX_REACH + 1];
    for (int i = 0; i <= reach; i++) fd_coeffs[i] = static_cast<float>(fd.c[i]);
    CUDA_CHECK(cudaMemcpyToSymbol(d_fd_coeffs, fd_coeffs, (reach + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_fd_reach, &reach, sizeof(int)));
    
    float *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, grid_bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_next, h_u.data(), grid_bytes, cudaMemcpyHostToDevice));
    
    dim3 block(8, 8, 4);
    dim3 grid3((N + 7) / 8, (N + 7) / 8, (N + 3) / 4);
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);
    
    VTK::write_timestep_vti("output/heat3d", 0, h_u, N, N, N);
    printf("Running simulation...\n");
    for (int t = 1; t <= timesteps; t++) {
        heat3d_kernel<<<grid3, block>>>(d_u, d_u_next, N, r);
        neumann_bc_kernel<<<bc_grid, bc_block>>>(d_u_next, N, reach);
        
        for (const auto& src : sources) {
            dim3 src_block(8, 8, 8);
            dim3 src_grid((src.size + 7) / 8, (src.size + 7) / 8, (src.size + 7) / 8);
            heat_source_kernel<<<src_grid, src_block>>>(d_u_next, N, src.x, src.size, src.temp);
        }
        
        float* tmp = d_u; d_u = d_u_next; d_u_next = tmp;
        
        if (t % save_interval == 0 || t == timesteps) {
            CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, grid_bytes, cudaMemcpyDeviceToHost));
            VTK::write_timestep_vti("output/heat3d", t, h_u, N, N, N);
            printf("Saved timestep %d\n", t);
        }
    }

    VTK::write_pvd_collection("output/heat3d.pvd", "heat3d", timesteps, save_interval, dt);
    printf("\nDone! ParaView file: output/heat3d.pvd\n");
    
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    
    return 0;
}

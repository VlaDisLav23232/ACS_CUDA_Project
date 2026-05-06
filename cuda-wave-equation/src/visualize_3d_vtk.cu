#include "fd_coefficients.h"
#include "stencil.h"
#include "vtk_export.h"
#include <cstdio>
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

__global__ void first_step_kernel(const float* __restrict__ u0,
                                  float* __restrict__ u1,
                                  int N, float s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_fd_reach;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = static_cast<size_t>(z) * N * N + y * N + x;
        float center = u0[idx];
        float lap = 3.0f * d_fd_coeffs[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_fd_coeffs[m] * (u0[idx - m] + u0[idx + m]
                                   + u0[idx - static_cast<size_t>(m) * N] + u0[idx + static_cast<size_t>(m) * N]
                                   + u0[idx - static_cast<size_t>(m) * N * N] + u0[idx + static_cast<size_t>(m) * N * N]);
        }
        u1[idx] = center + 0.5f * s * lap;
    }
}

__global__ void wave3d_kernel(const float* __restrict__ u_prev,
                              const float* __restrict__ u,
                              float* __restrict__ u_next,
                              int N, float s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int R = d_fd_reach;

    if (x >= R && x < N - R && y >= R && y < N - R && z >= R && z < N - R) {
        size_t idx = static_cast<size_t>(z) * N * N + y * N + x;
        float center = u[idx];
        float lap = 3.0f * d_fd_coeffs[0] * center;
        for (int m = 1; m <= R; m++) {
            lap += d_fd_coeffs[m] * (u[idx - m] + u[idx + m]
                                   + u[idx - static_cast<size_t>(m) * N] + u[idx + static_cast<size_t>(m) * N]
                                   + u[idx - static_cast<size_t>(m) * N * N] + u[idx + static_cast<size_t>(m) * N * N]);
        }
        u_next[idx] = 2.0f * center - u_prev[idx] + s * lap;
    }
}

__global__ void dirichlet_bc_3d(float* u, int N, int R) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b >= N) return;

    for (int r = 0; r < R; r++) {
        u[static_cast<size_t>(a) * N * N + b * N + r] = 0.0f;
        u[static_cast<size_t>(a) * N * N + b * N + (N - 1 - r)] = 0.0f;
        u[static_cast<size_t>(a) * N * N + r * N + b] = 0.0f;
        u[static_cast<size_t>(a) * N * N + (N - 1 - r) * N + b] = 0.0f;
        u[static_cast<size_t>(r) * N * N + a * N + b] = 0.0f;
        u[static_cast<size_t>(N - 1 - r) * N * N + a * N + b] = 0.0f;
    }
}

static void init_wave_field_3d(std::vector<float>& u, int N, float amp, float sigma) {
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
                u[static_cast<size_t>(z) * N * N + y * N + x] = v;
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 64;
    int timesteps = (argc > 2) ? std::atoi(argv[2]) : 1000;
    int save_interval = (argc > 3) ? std::atoi(argv[3]) : 100;
    int reach = (argc > 4) ? std::atoi(argv[4]) : 1;

    if (reach < 1 || reach > MAX_REACH) {
        std::fprintf(stderr, "reach must be 1..%d\n", MAX_REACH);
        return 1;
    }

    const float c = 1.0f;
    const float dx = 2.0f / (N - 1.0f);
    FDCoefficients fd = compute_fd_coefficients(reach, 3);
    const float s = 0.8f * static_cast<float>(fd.stability_limit);
    const float dt = std::sqrt(s) * dx / c;

    std::vector<float> coeffs(MAX_REACH + 1, 0.0f);
    for (int i = 0; i <= reach; i++) coeffs[i] = static_cast<float>(fd.c[i]);

    CUDA_CHECK(cudaMemcpyToSymbol(d_fd_coeffs, coeffs.data(), (reach + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_fd_reach, &reach, sizeof(int)));

    size_t total = static_cast<size_t>(N) * N * N;
    size_t bytes = total * sizeof(float);

    std::vector<float> h_u(total, 0.0f);
    init_wave_field_3d(h_u, N, 5.0f, 0.12f);

    float *d_u_prev, *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc(&d_u_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, bytes));
    CUDA_CHECK(cudaMemcpy(d_u_prev, h_u.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u, 0, bytes));
    CUDA_CHECK(cudaMemset(d_u_next, 0, bytes));

    dim3 block(8, 8, 4);
    dim3 grid3((N + 7) / 8, (N + 7) / 8, (N + 3) / 4);
    dim3 bc_block(16, 16);
    dim3 bc_grid((N + 15) / 16, (N + 15) / 16);

    first_step_kernel<<<grid3, block>>>(d_u_prev, d_u, N, s);
    dirichlet_bc_3d<<<bc_grid, bc_block>>>(d_u_prev, N, reach);
    dirichlet_bc_3d<<<bc_grid, bc_block>>>(d_u, N, reach);

    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u_prev, bytes, cudaMemcpyDeviceToHost));
    VTK::write_timestep_vti("output/wave3d", 0, h_u, N, N, N, dx, dx, dx);

    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, bytes, cudaMemcpyDeviceToHost));
    VTK::write_timestep_vti("output/wave3d", 1, h_u, N, N, N, dx, dx, dx);

    for (int t = 1; t < timesteps; t++) {
        wave3d_kernel<<<grid3, block>>>(d_u_prev, d_u, d_u_next, N, s);
        dirichlet_bc_3d<<<bc_grid, bc_block>>>(d_u_next, N, reach);

        float* tmp = d_u_prev;
        d_u_prev = d_u;
        d_u = d_u_next;
        d_u_next = tmp;

        if ((t + 1) % save_interval == 0 || t + 1 == timesteps) {
            CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, bytes, cudaMemcpyDeviceToHost));
            VTK::write_timestep_vti("output/wave3d", t + 1, h_u, N, N, N, dx, dx, dx);
            std::printf("saved timestep %d\n", t + 1);
        }
    }

    VTK::write_pvd_collection("output/wave3d.pvd", "wave3d", timesteps, save_interval, dt);
    std::printf("done. open output/wave3d.pvd in ParaView\n");

    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    return 0;
}

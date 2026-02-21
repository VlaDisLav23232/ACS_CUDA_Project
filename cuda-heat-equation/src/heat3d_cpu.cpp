#include "stencil.h"
#include <chrono>
#include <cmath>
#include <cstdio>

StencilResult run_cpu_fp64_3d(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    double dx = cfg.dx;
    double dt = cfg.dt;
    double k = cfg.k;
    double r = k * dt / (dx * dx);
    size_t total = (size_t)N * N * N;

    double coeffs[MAX_REACH + 1];
    for (int i = 0; i <= R; i++) coeffs[i] = cfg.fd_coeffs[i];

    std::vector<double> u(total, cfg.temp_initial);
    std::vector<double> u_next(total, cfg.temp_initial);

    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int z = src_start; z < src_start + src_size; z++)
        for (int y = src_start; y < src_start + src_size; y++)
            for (int x = src_start; x < src_start + src_size; x++)
                u[(size_t)z * N * N + y * N + x] = cfg.temp_source;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < cfg.timesteps; t++) {
        for (int z = R; z < N - R; z++) {
            for (int y = R; y < N - R; y++) {
                for (int x = R; x < N - R; x++) {
                    size_t idx = (size_t)z * N * N + y * N + x;
                    double center = u[idx];
                    double lap = 3.0 * coeffs[0] * center;
                    for (int m = 1; m <= R; m++) {
                        lap += coeffs[m] * (u[idx - m] + u[idx + m]
                                          + u[idx - (size_t)m * N] + u[idx + (size_t)m * N]
                                          + u[idx - (size_t)m * N * N] + u[idx + (size_t)m * N * N]);
                    }
                    u_next[idx] = center + r * lap;
                }
            }
        }

        for (int b = R - 1; b >= 0; b--) {
            for (int z = 0; z < N; z++) {
                for (int y = 0; y < N; y++) {
                    size_t row = (size_t)z * N * N + y * N;
                    u_next[row + b]       = u_next[row + (b + 1)];
                    u_next[row + (N-1-b)] = u_next[row + (N-2-b)];
                }
            }
            for (int z = 0; z < N; z++) {
                for (int x = 0; x < N; x++) {
                    size_t base = (size_t)z * N * N + x;
                    u_next[base + (size_t)b * N]       = u_next[base + (size_t)(b + 1) * N];
                    u_next[base + (size_t)(N-1-b) * N] = u_next[base + (size_t)(N-2-b) * N];
                }
            }
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
                    size_t base = (size_t)y * N + x;
                    u_next[(size_t)b * N * N + base]       = u_next[(size_t)(b + 1) * N * N + base];
                    u_next[(size_t)(N-1-b) * N * N + base] = u_next[(size_t)(N-2-b) * N * N + base];
                }
            }
        }

        std::swap(u, u_next);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    StencilResult res;
    res.variant_name = "cpu_fp64_3d";
    res.grid_size = N;
    res.dim = 3;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = ms;
    res.effective_bw_gbs = 0.0;
    res.memory_bytes = 2 * total * sizeof(double);
    res.final_grid.resize(total);
    for (size_t i = 0; i < total; i++)
        res.final_grid[i] = (float)u[i];
    return res;
}

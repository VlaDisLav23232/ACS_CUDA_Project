#include "stencil.h"
#include <chrono>
#include <cmath>
#include <cstdio>

StencilResult run_cpu_fp64(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    double dx = cfg.dx;
    double dt = cfg.dt;
    double k = cfg.k;
    double r = k * dt / (dx * dx);

    double coeffs[MAX_REACH + 1];
    for (int i = 0; i <= R; i++) coeffs[i] = cfg.fd_coeffs[i];

    std::vector<double> u(N * N, cfg.temp_initial);
    std::vector<double> u_next(N * N, cfg.temp_initial);

    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++)
        for (int i = src_start; i < src_start + src_size; i++)
            u[j * N + i] = cfg.temp_source;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < cfg.timesteps; t++) {
        for (int j = R; j < N - R; j++) {
            for (int i = R; i < N - R; i++) {
                double center = u[j * N + i];
                double lap = 2.0 * coeffs[0] * center;
                for (int m = 1; m <= R; m++) {
                    lap += coeffs[m] * (u[j * N + (i - m)] + u[j * N + (i + m)]
                                      + u[(j - m) * N + i] + u[(j + m) * N + i]);
                }
                u_next[j * N + i] = center + r * lap;
            }
        }
        for (int b = R - 1; b >= 0; b--) {
            for (int i = 0; i < N; i++) {
                u_next[b * N + i]       = u_next[(b + 1) * N + i];
                u_next[(N-1-b) * N + i] = u_next[(N-2-b) * N + i];
            }
            for (int j = 0; j < N; j++) {
                u_next[j * N + b]       = u_next[j * N + (b + 1)];
                u_next[j * N + (N-1-b)] = u_next[j * N + (N-2-b)];
            }
        }
        std::swap(u, u_next);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    StencilResult res;
    res.variant_name = "cpu_fp64";
    res.grid_size = N;
    res.dim = cfg.dim;
    res.stencil_reach = R;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = ms;
    res.effective_bw_gbs = 0.0;
    res.memory_bytes = 2 * N * N * sizeof(double);
    res.final_grid.resize(N * N);
    for (int i = 0; i < N * N; i++)
        res.final_grid[i] = (float)u[i];
    return res;
}

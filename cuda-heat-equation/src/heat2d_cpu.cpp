#include "stencil.h"
#include <chrono>
#include <cmath>
#include <cstdio>

// CPU reference implementation using fp64 for maximum accuracy
// this is our ground truth to compare all GPU variants against
StencilResult run_cpu_fp64(const StencilConfig& cfg) {
    int N = cfg.nx;
    double dx = cfg.dx;
    double dt = cfg.dt;
    double k = cfg.k;
    double r = k * dt / (dx * dx);

    // two buffers for ping-pong
    std::vector<double> u(N * N, cfg.temp_initial);
    std::vector<double> u_next(N * N, cfg.temp_initial);

    // place a heat source block in the center (same idea as heat2d.py)
    int src_size = N / 8;
    int src_start = N / 2 - src_size / 2;
    for (int j = src_start; j < src_start + src_size; j++) {
        for (int i = src_start; i < src_start + src_size; i++) {
            u[j * N + i] = cfg.temp_source;
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < cfg.timesteps; t++) {
        // FTCS update for interior points
        for (int j = 1; j < N - 1; j++) {
            for (int i = 1; i < N - 1; i++) {
                double center = u[j * N + i];
                double left   = u[j * N + (i - 1)];
                double right  = u[j * N + (i + 1)];
                double up     = u[(j - 1) * N + i];
                double down   = u[(j + 1) * N + i];
                u_next[j * N + i] = center + r * (left + right + up + down - 4.0 * center);
            }
        }
        // insulated boundaries (Neumann: derivative = 0)
        for (int i = 0; i < N; i++) {
            u_next[0 * N + i] = u_next[1 * N + i];
            u_next[(N-1) * N + i] = u_next[(N-2) * N + i];
            u_next[i * N + 0] = u_next[i * N + 1];
            u_next[i * N + (N-1)] = u_next[i * N + (N-2)];
        }

        std::swap(u, u_next);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // convert to float for the result
    StencilResult res;
    res.variant_name = "cpu_fp64";
    res.grid_size = N;
    res.timesteps = cfg.timesteps;
    res.elapsed_ms = ms;
    res.effective_bw_gbs = 0.0; // not meaningful for CPU
    res.memory_bytes = 2 * N * N * sizeof(double);
    res.final_grid.resize(N * N);
    for (int i = 0; i < N * N; i++) {
        res.final_grid[i] = (float)u[i];
    }
    return res;
}

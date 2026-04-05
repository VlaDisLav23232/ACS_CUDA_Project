#include "stencil.h"
#include <algorithm>
#include <chrono>
#include <cmath>

static void apply_dirichlet_2d(std::vector<double>& u, int N, int R) {
    for (int b = 0; b < R; b++) {
        for (int i = 0; i < N; i++) {
            u[b * N + i] = 0.0;
            u[(N - 1 - b) * N + i] = 0.0;
            u[i * N + b] = 0.0;
            u[i * N + (N - 1 - b)] = 0.0;
        }
    }
}

static void init_wave_field_2d(std::vector<double>& u, int N, float amp, float sigma) {
    std::fill(u.begin(), u.end(), 0.0);
    const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma);
    const double src[2][2] = {{-0.35, -0.35}, {0.35, 0.35}};

    for (int j = 0; j < N; j++) {
        double y = -1.0 + 2.0 * j / (N - 1.0);
        for (int i = 0; i < N; i++) {
            double x = -1.0 + 2.0 * i / (N - 1.0);
            double v = 0.0;
            for (int k = 0; k < 2; k++) {
                double dx = x - src[k][0];
                double dy = y - src[k][1];
                v += amp * std::exp(-(dx * dx + dy * dy) / s2);
            }
            u[j * N + i] = v;
        }
    }
}

StencilResult run_cpu_fp64(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    double s = (cfg.c * cfg.dt / cfg.dx) * (cfg.c * cfg.dt / cfg.dx);

    double coeffs[MAX_REACH + 1];
    for (int i = 0; i <= R; i++) coeffs[i] = cfg.fd_coeffs[i];

    std::vector<double> u_prev(N * N, cfg.disp_initial);
    std::vector<double> u(N * N, cfg.disp_initial);
    std::vector<double> u_next(N * N, cfg.disp_initial);

    init_wave_field_2d(u_prev, N, cfg.source_amplitude, cfg.source_sigma);
    apply_dirichlet_2d(u_prev, N, R);

    for (int j = R; j < N - R; j++) {
        for (int i = R; i < N - R; i++) {
            int idx = j * N + i;
            double center = u_prev[idx];
            double lap = 2.0 * coeffs[0] * center;
            for (int m = 1; m <= R; m++) {
                lap += coeffs[m] * (u_prev[j * N + (i - m)] + u_prev[j * N + (i + m)]
                                  + u_prev[(j - m) * N + i] + u_prev[(j + m) * N + i]);
            }
            u[idx] = center + 0.5 * s * lap;
        }
    }
    apply_dirichlet_2d(u, N, R);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 1; t < cfg.timesteps; t++) {
        for (int j = R; j < N - R; j++) {
            for (int i = R; i < N - R; i++) {
                int idx = j * N + i;
                double center = u[idx];
                double lap = 2.0 * coeffs[0] * center;
                for (int m = 1; m <= R; m++) {
                    lap += coeffs[m] * (u[j * N + (i - m)] + u[j * N + (i + m)]
                                      + u[(j - m) * N + i] + u[(j + m) * N + i]);
                }
                u_next[idx] = 2.0 * center - u_prev[idx] + s * lap;
            }
        }
        apply_dirichlet_2d(u_next, N, R);
        std::swap(u_prev, u);
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
    res.memory_bytes = 3 * static_cast<size_t>(N) * N * sizeof(double);
    res.final_grid.resize(static_cast<size_t>(N) * N);
    for (size_t i = 0; i < res.final_grid.size(); i++) res.final_grid[i] = static_cast<float>(u[i]);
    return res;
}

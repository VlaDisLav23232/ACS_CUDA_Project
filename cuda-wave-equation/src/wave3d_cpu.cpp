#include "stencil.h"
#include <algorithm>
#include <chrono>
#include <cmath>

static void apply_dirichlet_3d(std::vector<double>& u, int N, int R) {
    for (int b = 0; b < R; b++) {
        for (int z = 0; z < N; z++) {
            for (int y = 0; y < N; y++) {
                u[static_cast<size_t>(z) * N * N + y * N + b] = 0.0;
                u[static_cast<size_t>(z) * N * N + y * N + (N - 1 - b)] = 0.0;
            }
        }
        for (int z = 0; z < N; z++) {
            for (int x = 0; x < N; x++) {
                u[static_cast<size_t>(z) * N * N + b * N + x] = 0.0;
                u[static_cast<size_t>(z) * N * N + (N - 1 - b) * N + x] = 0.0;
            }
        }
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                u[static_cast<size_t>(b) * N * N + y * N + x] = 0.0;
                u[static_cast<size_t>(N - 1 - b) * N * N + y * N + x] = 0.0;
            }
        }
    }
}

static void init_wave_field_3d(std::vector<double>& u, int N, float amp, float sigma) {
    std::fill(u.begin(), u.end(), 0.0);
    const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma);
    const double src[2][3] = {{-0.30, -0.30, -0.30}, {0.30, 0.30, 0.30}};

    for (int z = 0; z < N; z++) {
        double zz = -1.0 + 2.0 * z / (N - 1.0);
        for (int y = 0; y < N; y++) {
            double yy = -1.0 + 2.0 * y / (N - 1.0);
            for (int x = 0; x < N; x++) {
                double xx = -1.0 + 2.0 * x / (N - 1.0);
                double v = 0.0;
                for (int k = 0; k < 2; k++) {
                    double dx = xx - src[k][0];
                    double dy = yy - src[k][1];
                    double dz = zz - src[k][2];
                    v += amp * std::exp(-(dx * dx + dy * dy + dz * dz) / s2);
                }
                u[static_cast<size_t>(z) * N * N + y * N + x] = v;
            }
        }
    }
}

StencilResult run_cpu_fp64_3d(const StencilConfig& cfg) {
    int N = cfg.nx;
    int R = cfg.stencil_reach;
    double s = (cfg.c * cfg.dt / cfg.dx) * (cfg.c * cfg.dt / cfg.dx);
    size_t total = static_cast<size_t>(N) * N * N;

    double coeffs[MAX_REACH + 1];
    for (int i = 0; i <= R; i++) coeffs[i] = cfg.fd_coeffs[i];

    std::vector<double> u_prev(total, cfg.disp_initial);
    std::vector<double> u(total, cfg.disp_initial);
    std::vector<double> u_next(total, cfg.disp_initial);

    init_wave_field_3d(u_prev, N, cfg.source_amplitude, cfg.source_sigma);
    apply_dirichlet_3d(u_prev, N, R);

    for (int z = R; z < N - R; z++) {
        for (int y = R; y < N - R; y++) {
            for (int x = R; x < N - R; x++) {
                size_t idx = static_cast<size_t>(z) * N * N + y * N + x;
                double center = u_prev[idx];
                double lap = 3.0 * coeffs[0] * center;
                for (int m = 1; m <= R; m++) {
                    lap += coeffs[m] * (u_prev[idx - m] + u_prev[idx + m]
                                      + u_prev[idx - static_cast<size_t>(m) * N] + u_prev[idx + static_cast<size_t>(m) * N]
                                      + u_prev[idx - static_cast<size_t>(m) * N * N] + u_prev[idx + static_cast<size_t>(m) * N * N]);
                }
                u[idx] = center + 0.5 * s * lap;
            }
        }
    }
    apply_dirichlet_3d(u, N, R);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 1; t < cfg.timesteps; t++) {
        for (int z = R; z < N - R; z++) {
            for (int y = R; y < N - R; y++) {
                for (int x = R; x < N - R; x++) {
                    size_t idx = static_cast<size_t>(z) * N * N + y * N + x;
                    double center = u[idx];
                    double lap = 3.0 * coeffs[0] * center;
                    for (int m = 1; m <= R; m++) {
                        lap += coeffs[m] * (u[idx - m] + u[idx + m]
                                          + u[idx - static_cast<size_t>(m) * N] + u[idx + static_cast<size_t>(m) * N]
                                          + u[idx - static_cast<size_t>(m) * N * N] + u[idx + static_cast<size_t>(m) * N * N]);
                    }
                    u_next[idx] = 2.0 * center - u_prev[idx] + s * lap;
                }
            }
        }
        apply_dirichlet_3d(u_next, N, R);
        std::swap(u_prev, u);
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
    res.memory_bytes = 3 * total * sizeof(double);
    res.final_grid.resize(total);
    for (size_t i = 0; i < total; i++) res.final_grid[i] = static_cast<float>(u[i]);
    return res;
}

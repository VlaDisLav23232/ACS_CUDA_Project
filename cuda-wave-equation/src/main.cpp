#include "stencil.h"
#include "metrics.h"
#include "fd_coefficients.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>

StencilConfig default_config() {
    StencilConfig cfg;
    cfg.nx = 256;
    cfg.ny = 256;
    cfg.nz = 1;
    cfg.dim = 2;
    cfg.dx = 0.04f;
    cfg.dt = 0.002f;
    cfg.c = 1.0f;
    cfg.timesteps = 3000;
    cfg.disp_initial = 0.0f;
    cfg.source_amplitude = 1.0f;
    cfg.source_sigma = 0.12f;
    cfg.stencil_reach = 1;
    std::memset(cfg.fd_coeffs, 0, sizeof(cfg.fd_coeffs));
    cfg.stability_limit = 1.0f;
    return cfg;
}

static void print_usage() {
    std::printf("usage: wave_stencil [options]\n");
    std::printf("  -n <size>       grid size NxN or NxNxN (default 256)\n");
    std::printf("  -t <steps>      timesteps (default 3000)\n");
    std::printf("  -d <dim>        dimensionality: 2 or 3 (default 2)\n");
    std::printf("  -r <reach>      stencil reach per axis: 1,4,8 (default 1)\n");
    std::printf("  -v <variant>    cpu|fp32|fp16|kahan|all (default all)\n");
    std::printf("  -o <path>       CSV output path (default results/benchmarks.csv)\n");
    std::printf("  -h              show this help\n");
}

int main(int argc, char** argv) {
    StencilConfig cfg = default_config();
    std::string variant = "all";
    std::string csv_path = "results/benchmarks.csv";

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            cfg.nx = cfg.ny = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            cfg.timesteps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            cfg.dim = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            cfg.stencil_reach = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            variant = argv[++i];
        } else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (std::strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
    }

    if (cfg.dim != 2 && cfg.dim != 3) {
        std::fprintf(stderr, "error: dimension must be 2 or 3\n");
        return 1;
    }
    if (cfg.stencil_reach < 1 || cfg.stencil_reach > MAX_REACH) {
        std::fprintf(stderr, "error: reach must be 1..%d\n", MAX_REACH);
        return 1;
    }

    cfg.nz = (cfg.dim == 3) ? cfg.nx : 1;

    FDCoefficients fd = compute_fd_coefficients(cfg.stencil_reach, cfg.dim);
    for (int i = 0; i <= cfg.stencil_reach; i++) cfg.fd_coeffs[i] = static_cast<float>(fd.c[i]);
    cfg.stability_limit = static_cast<float>(fd.stability_limit);

    cfg.dx = 2.0f / (cfg.nx - 1);
    const float safety = 0.80f;
    const float s = safety * cfg.stability_limit;
    cfg.dt = std::sqrt(s) * cfg.dx / cfg.c;

    if (cfg.dim == 2) std::printf("grid: %dx%d, ", cfg.nx, cfg.ny);
    else std::printf("grid: %dx%dx%d, ", cfg.nx, cfg.ny, cfg.nz);
    std::printf("dx=%.6f, dt=%.6f, c=%.3f, reach=%d\n", cfg.dx, cfg.dt, cfg.c, cfg.stencil_reach);
    std::printf("wave CFL^2 s = %.6f (needs < %.6f for %dD reach-%d leapfrog)\n",
                s, cfg.stability_limit, cfg.dim, cfg.stencil_reach);
    print_fd_coefficients(fd);

    ensure_csv_header(csv_path);

    if (cfg.dim == 2) {
        std::printf("\n--- CPU fp64 reference ---\n");
        StencilResult cpu_result = run_cpu_fp64(cfg);
        cpu_result.max_abs_error = 0.0;
        cpu_result.l2_error = 0.0;
        print_summary(cpu_result);
        write_csv_row(csv_path, cpu_result);

        if (variant == "all" || variant == "fp32") {
            std::printf("\n--- CUDA fp32 ---\n");
            StencilResult r32 = run_cuda_fp32(cfg);
            compute_errors(r32, cpu_result.final_grid);
            print_summary(r32);
            write_csv_row(csv_path, r32);
        }
        if (variant == "all" || variant == "fp16") {
            std::printf("\n--- CUDA fp16 (no Kahan) ---\n");
            StencilResult r16 = run_cuda_fp16_naive(cfg);
            compute_errors(r16, cpu_result.final_grid);
            print_summary(r16);
            write_csv_row(csv_path, r16);
        }
        if (variant == "all" || variant == "kahan") {
            std::printf("\n--- CUDA fp16 + Kahan ---\n");
            StencilResult rk = run_cuda_fp16_kahan(cfg);
            compute_errors(rk, cpu_result.final_grid);
            print_summary(rk);
            write_csv_row(csv_path, rk);
        }
    } else {
        std::printf("\n--- CPU fp64 reference (3D) ---\n");
        StencilResult cpu_result = run_cpu_fp64_3d(cfg);
        cpu_result.max_abs_error = 0.0;
        cpu_result.l2_error = 0.0;
        print_summary(cpu_result);
        write_csv_row(csv_path, cpu_result);

        if (variant == "all" || variant == "fp32") {
            std::printf("\n--- CUDA fp32 (3D) ---\n");
            StencilResult r32 = run_cuda_fp32_3d(cfg);
            compute_errors(r32, cpu_result.final_grid);
            print_summary(r32);
            write_csv_row(csv_path, r32);
        }
        if (variant == "all" || variant == "fp16") {
            std::printf("\n--- CUDA fp16 (3D, no Kahan) ---\n");
            StencilResult r16 = run_cuda_fp16_naive_3d(cfg);
            compute_errors(r16, cpu_result.final_grid);
            print_summary(r16);
            write_csv_row(csv_path, r16);
        }
        if (variant == "all" || variant == "kahan") {
            std::printf("\n--- CUDA fp16 + Kahan (3D) ---\n");
            StencilResult rk = run_cuda_fp16_kahan_3d(cfg);
            compute_errors(rk, cpu_result.final_grid);
            print_summary(rk);
            write_csv_row(csv_path, rk);
        }
    }

    std::printf("\nresults written to %s\n", csv_path.c_str());
    return 0;
}

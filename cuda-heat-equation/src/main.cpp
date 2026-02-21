#include "stencil.h"
#include "metrics.h"
#include <cstdio>
#include <cstring>

StencilConfig default_config() {
    StencilConfig cfg;
    cfg.nx = 256;
    cfg.ny = 256;
    cfg.dx = 0.04f;
    cfg.dt = 0.0004f;
    cfg.k = 0.466f;
    cfg.timesteps = 5000;
    cfg.temp_initial = 0.0f;
    cfg.temp_source = 200.0f;
    return cfg;
}

void print_usage() {
    printf("usage: heat_stencil [options]\n");
    printf("  -n <size>       grid size NxN (default 256)\n");
    printf("  -t <steps>      timesteps (default 5000)\n");
    printf("  -v <variant>    cpu|fp32|fp16|kahan|all (default all)\n");
    printf("  -o <path>       CSV output path (default results/benchmarks.csv)\n");
    printf("  -h              show this help\n");
}

int main(int argc, char** argv) {
    StencilConfig cfg = default_config();
    std::string variant = "all";
    std::string csv_path = "results/benchmarks.csv";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            cfg.nx = cfg.ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) {
            cfg.timesteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0 && i+1 < argc) {
            variant = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
    }

    // recalculate dx based on grid size so domain stays 2.0 x 2.0
    cfg.dx = 2.0f / (cfg.nx - 1);
    // auto-compute dt for stability: r = k*dt/dx^2 < 0.25, use r=0.2 for margin
    cfg.dt = 0.2f * cfg.dx * cfg.dx / cfg.k;

    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);
    printf("grid: %dx%d, dx=%.4f, dt=%.6f, k=%.3f\n", cfg.nx, cfg.ny, cfg.dx, cfg.dt, cfg.k);
    printf("stability factor r = %.6f (needs < 0.25 for 2D FTCS)\n", r);

    ensure_csv_header(csv_path);

    // always run CPU first as ground truth
    printf("\n--- CPU fp64 reference ---\n");
    StencilResult cpu_result = run_cpu_fp64(cfg);
    cpu_result.max_abs_error = 0.0;
    cpu_result.l2_error = 0.0;
    print_summary(cpu_result);
    write_csv_row(csv_path, cpu_result);

    if (variant == "all" || variant == "fp32") {
        printf("\n--- CUDA fp32 ---\n");
        StencilResult r32 = run_cuda_fp32(cfg);
        compute_errors(r32, cpu_result.final_grid);
        print_summary(r32);
        write_csv_row(csv_path, r32);
    }

    if (variant == "all" || variant == "fp16") {
        printf("\n--- CUDA fp16 (no Kahan) ---\n");
        StencilResult r16 = run_cuda_fp16_naive(cfg);
        compute_errors(r16, cpu_result.final_grid);
        print_summary(r16);
        write_csv_row(csv_path, r16);
    }

    if (variant == "all" || variant == "kahan") {
        printf("\n--- CUDA fp16 + Kahan ---\n");
        StencilResult rk = run_cuda_fp16_kahan(cfg);
        compute_errors(rk, cpu_result.final_grid);
        print_summary(rk);
        write_csv_row(csv_path, rk);
    }

    printf("\nresults written to %s\n", csv_path.c_str());
    return 0;
}

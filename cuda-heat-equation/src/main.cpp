#include "stencil.h"
#include "metrics.h"
#include "fd_coefficients.h"
#include <cstdio>
#include <cstring>
#include <limits>

StencilConfig default_config() {
    StencilConfig cfg;
    cfg.nx = 256;
    cfg.ny = 256;
    cfg.nz = 1;
    cfg.dim = 2;
    cfg.dx = 0.04f;
    cfg.dt = 0.0004f;
    cfg.k = 0.466f;
    cfg.timesteps = 5000;
    cfg.temp_initial = 0.0f;
    cfg.temp_source = 200.0f;
    cfg.stencil_reach = 1;
    memset(cfg.fd_coeffs, 0, sizeof(cfg.fd_coeffs));
    cfg.stability_limit = 0.25f;
    return cfg;
}

void print_usage() {
    printf("usage: heat_stencil [options]\n");
    printf("  -n <size>       grid size NxN or NxNxN (default 256)\n");
    printf("  -t <steps>      timesteps (default 5000)\n");
    printf("  -d <dim>        dimensionality: 2 or 3 (default 2)\n");
    printf("  -r <reach>      stencil reach per axis: 1,4,8 (default 1)\n");
    printf("  -v <variant>    cpu|fp32|fp16|kahan|kahan_reg|cfp16|cfp16_kahan|cfp16_kahan_tiled|fp32_25d|fp16_25d_zreg|fp16_25d_zreg_async|fp32_25d_zreg|kahan_25d|kahan_25d_zreg|kahan_25d_zreg_async|kahan_25d_async|all (default all)\n");
    printf("  -o <path>       CSV output path (default results/benchmarks.csv)\n");
    printf("  --no-reference  skip CPU reference and write NaN accuracy fields for GPU-only timing\n");
    printf("  -h              show this help\n");
}

void finalize_gpu_result(StencilResult& result, const StencilResult* cpu_result, const std::string& csv_path) {
    if (cpu_result) {
        compute_errors(result, cpu_result->final_grid);
    } else {
        result.max_abs_error = std::numeric_limits<double>::quiet_NaN();
        result.l2_error = std::numeric_limits<double>::quiet_NaN();
    }
    print_summary(result);
    write_csv_row(csv_path, result);
}

int main(int argc, char** argv) {
    StencilConfig cfg = default_config();
    std::string variant = "all";
    std::string csv_path = "../results/benchmarks.csv";
    bool use_reference = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            cfg.nx = cfg.ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) {
            cfg.timesteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i+1 < argc) {
            cfg.dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i+1 < argc) {
            cfg.stencil_reach = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0 && i+1 < argc) {
            variant = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "--no-reference") == 0) {
            use_reference = false;
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
    }

    if (cfg.dim != 2 && cfg.dim != 3) {
        fprintf(stderr, "error: dimension must be 2 or 3\n");
        return 1;
    }
    if (cfg.stencil_reach < 1 || cfg.stencil_reach > MAX_REACH) {
        fprintf(stderr, "error: reach must be 1..%d\n", MAX_REACH);
        return 1;
    }
    cfg.nz = (cfg.dim == 3) ? cfg.nx : 1;

    FDCoefficients fd = compute_fd_coefficients(cfg.stencil_reach, cfg.dim);
    for (int i = 0; i <= cfg.stencil_reach; i++)
        cfg.fd_coeffs[i] = (float)fd.c[i];
    cfg.stability_limit = (float)fd.stability_limit;

    cfg.dx = 2.0f / (cfg.nx - 1);
    float safety = 0.8f;
    cfg.dt = safety * cfg.stability_limit * cfg.dx * cfg.dx / cfg.k;
    float r = cfg.k * cfg.dt / (cfg.dx * cfg.dx);

    if (cfg.dim == 2)
        printf("grid: %dx%d, ", cfg.nx, cfg.ny);
    else
        printf("grid: %dx%dx%d, ", cfg.nx, cfg.ny, cfg.nz);
    printf("dx=%.4f, dt=%.6f, k=%.3f, reach=%d\n", cfg.dx, cfg.dt, cfg.k, cfg.stencil_reach);
    printf("stability factor r = %.6f (needs < %.6f for %dD reach-%d FTCS)\n",
           r, cfg.stability_limit, cfg.dim, cfg.stencil_reach);
    print_fd_coefficients(fd);

    ensure_csv_header(csv_path);

    if (cfg.dim == 2) {
        StencilResult cpu_result;
        StencilResult* cpu_ref = nullptr;
        if (use_reference) {
            printf("\n--- CPU fp64 reference ---\n");
            cpu_result = run_cpu_fp64(cfg);
            cpu_result.max_abs_error = 0.0;
            cpu_result.l2_error = 0.0;
            print_summary(cpu_result);
            write_csv_row(csv_path, cpu_result);
            cpu_ref = &cpu_result;
        }

        if (variant == "all" || variant == "fp32") {
            printf("\n--- CUDA fp32 ---\n");
            StencilResult r32 = run_cuda_fp32(cfg);
            finalize_gpu_result(r32, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "fp16") {
            printf("\n--- CUDA fp16 (no Kahan) ---\n");
            StencilResult r16 = run_cuda_fp16_naive(cfg);
            finalize_gpu_result(r16, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "kahan") {
            printf("\n--- CUDA fp16 + Kahan ---\n");
            StencilResult rk = run_cuda_fp16_kahan(cfg);
            finalize_gpu_result(rk, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "tensor") {
            if (cfg.nx % 16 != 0) {
                fprintf(stderr, "warning: tensor core variant requires N%%16==0, skipping (N=%d)\n", cfg.nx);
            } else {
                printf("\n--- CUDA fp16 Tensor Core (WMMA) ---\n");
                StencilResult rtc = run_cuda_fp16_tensor_core(cfg);
                finalize_gpu_result(rtc, cpu_ref, csv_path);
            }
        }
        if (variant == "all" || variant == "cfp16") {
            printf("\n--- CUDA custom fp16 (1-4-11, no Kahan) ---\n");
            StencilResult rcf16 = run_cuda_cfp16_naive(cfg);
            finalize_gpu_result(rcf16, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "cfp16_kahan") {
            printf("\n--- CUDA custom fp16 (1-4-11) + Kahan ---\n");
            StencilResult rcfk = run_cuda_cfp16_kahan(cfg);
            finalize_gpu_result(rcfk, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "cfp16_kahan_tiled") {
            printf("\n--- CUDA custom fp16 (1-4-11) + Kahan (tiled) ---\n");
            StencilResult rcfkt = run_cuda_cfp16_kahan_tiled(cfg);
            finalize_gpu_result(rcfkt, cpu_ref, csv_path);
        }
    } else {
        StencilResult cpu_result;
        StencilResult* cpu_ref = nullptr;
        if (use_reference) {
            printf("\n--- CPU fp64 reference (3D) ---\n");
            cpu_result = run_cpu_fp64_3d(cfg);
            cpu_result.max_abs_error = 0.0;
            cpu_result.l2_error = 0.0;
            print_summary(cpu_result);
            write_csv_row(csv_path, cpu_result);
            cpu_ref = &cpu_result;
        }

        if (variant == "all" || variant == "fp32") {
            printf("\n--- CUDA fp32 (3D) ---\n");
            StencilResult r32 = run_cuda_fp32_3d(cfg);
            finalize_gpu_result(r32, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "fp16") {
            printf("\n--- CUDA fp16 (3D, no Kahan) ---\n");
            StencilResult r16 = run_cuda_fp16_naive_3d(cfg);
            finalize_gpu_result(r16, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "kahan") {
            printf("\n--- CUDA fp16 + Kahan (3D) ---\n");
            StencilResult rk = run_cuda_fp16_kahan_3d(cfg);
            finalize_gpu_result(rk, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "cfp16_kahan_3d_tiled") {
            printf("\n--- CUDA custom fp16 (1-4-11) + Kahan (tiled, 3D) ---\n");
            StencilResult rcfkt3d = run_cuda_cfp16_kahan_3d_tiled(cfg);
            finalize_gpu_result(rcfkt3d, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "fp32_25d") {
            printf("\n--- CUDA fp32 2.5D blocking (3D) ---\n");
            StencilResult r25 = run_cuda_fp32_3d_25d(cfg);
            finalize_gpu_result(r25, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "fp32_25d_zreg") {
            printf("\n--- CUDA fp32 2.5D + Z-register blocking (3D) ---\n");
            StencilResult rz25 = run_cuda_fp32_3d_25d_zreg(cfg);
            finalize_gpu_result(rz25, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "fp16_25d_zreg") {
            printf("\n--- CUDA fp16 2.5D + Z-register blocking (3D, no Kahan) ---\n");
            StencilResult rhz25 = run_cuda_fp16_naive_3d_25d_zreg(cfg);
            finalize_gpu_result(rhz25, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "fp16_25d_zreg_async") {
            printf("\n--- CUDA fp16 2.5D + Z-register + async XY staging (3D, no Kahan) ---\n");
            StencilResult rhaz25 = run_cuda_fp16_naive_3d_25d_zreg_async(cfg);
            finalize_gpu_result(rhaz25, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "kahan_25d") {
            printf("\n--- CUDA fp16 + Kahan 2.5D blocking (3D) ---\n");
            StencilResult rk25 = run_cuda_fp16_kahan_3d_25d(cfg);
            finalize_gpu_result(rk25, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "kahan_25d_zreg") {
            printf("\n--- CUDA fp16 + Kahan 2.5D + Z-register blocking (3D) ---\n");
            StencilResult rkz25 = run_cuda_fp16_kahan_3d_25d_zreg(cfg);
            finalize_gpu_result(rkz25, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "kahan_25d_zreg_async") {
            printf("\n--- CUDA fp16 + Kahan 2.5D + Z-register + async XY staging (3D) ---\n");
            StencilResult rkaz25 = run_cuda_fp16_kahan_3d_25d_zreg_async(cfg);
            finalize_gpu_result(rkaz25, cpu_ref, csv_path);
        }
        if (variant == "all" || variant == "kahan_25d_async") {
            printf("\n--- CUDA fp16 + Kahan async memory pipeline (3D, 2.5D tiling) ---\n");
            StencilResult rk25_async = run_cuda_fp16_kahan_3d_25d_async(cfg);
            finalize_gpu_result(rk25_async, cpu_ref, csv_path);
        }
    }

    printf("\nresults written to %s\n", csv_path.c_str());
    return 0;
}

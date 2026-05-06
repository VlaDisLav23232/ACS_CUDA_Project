#include "metrics.h"
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sys/stat.h>

void compute_errors(StencilResult& result, const std::vector<float>& reference) {
    if (result.final_grid.size() != reference.size()) {
        std::fprintf(stderr, "grid size mismatch in error computation\n");
        return;
    }

    double max_err = 0.0;
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    int bad_count = 0;

    for (size_t i = 0; i < reference.size(); i++) {
        double v = static_cast<double>(result.final_grid[i]);
        double r = static_cast<double>(reference[i]);
        if (std::isnan(v) || std::isinf(v) || std::isnan(r) || std::isinf(r)) {
            bad_count++;
            continue;
        }
        double d = v - r;
        double ad = std::fabs(d);
        if (ad > max_err) max_err = ad;
        sum_sq += d * d;
        ref_sq += r * r;
    }

    if (bad_count > 0) {
        std::fprintf(stderr, "warning: %d NaN/Inf values skipped\n", bad_count);
    }

    result.max_abs_error = max_err;
    result.l2_error = (ref_sq > 0.0) ? std::sqrt(sum_sq / ref_sq) : std::sqrt(sum_sq);
}

void ensure_csv_header(const std::string& filepath) {
    struct stat st;
    if (stat(filepath.c_str(), &st) != 0) {
        std::ofstream f(filepath);
        f << "timestamp,variant,dim,reach,grid_size,timesteps,elapsed_ms,max_abs_error,l2_error,bandwidth_gbs,memory_bytes\n";
    }
}

void write_csv_row(const std::string& filepath, const StencilResult& result) {
    std::ofstream f(filepath, std::ios::app);
    time_t now = time(nullptr);
    char tbuf[64];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    f << tbuf << ","
      << result.variant_name << ","
      << result.dim << ","
      << result.stencil_reach << ","
      << result.grid_size << ","
      << result.timesteps << ","
      << result.elapsed_ms << ","
      << result.max_abs_error << ","
      << result.l2_error << ","
      << result.effective_bw_gbs << ","
      << result.memory_bytes << "\n";
}

void print_summary(const StencilResult& result) {
    std::printf("  variant:    %s\n", result.variant_name.c_str());
    if (result.dim == 3) {
        std::printf("  grid:       %dx%dx%d\n", result.grid_size, result.grid_size, result.grid_size);
    } else {
        std::printf("  grid:       %dx%d\n", result.grid_size, result.grid_size);
    }
    std::printf("  reach:      %d\n", result.stencil_reach);
    std::printf("  timesteps:  %d\n", result.timesteps);
    std::printf("  time:       %.2f ms\n", result.elapsed_ms);
    std::printf("  max error:  %.6e\n", result.max_abs_error);
    std::printf("  L2 error:   %.6e\n", result.l2_error);
    if (result.effective_bw_gbs > 0.0) std::printf("  bandwidth:  %.1f GB/s\n", result.effective_bw_gbs);
    std::printf("  memory:     %.2f MB\n", result.memory_bytes / (1024.0 * 1024.0));
}

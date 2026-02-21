#include "metrics.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <ctime>
#include <sys/stat.h>

void compute_errors(StencilResult& result, const std::vector<float>& reference) {
    if (result.final_grid.size() != reference.size()) {
        fprintf(stderr, "grid size mismatch in error computation!\n");
        return;
    }

    double max_err = 0.0;
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    int nan_count = 0;

    for (size_t i = 0; i < reference.size(); i++) {
        double val = (double)result.final_grid[i];
        double ref = (double)reference[i];
        if (std::isnan(val) || std::isinf(val) || std::isnan(ref) || std::isinf(ref)) {
            nan_count++;
            continue;
        }
        double diff = val - ref;
        double abs_diff = fabs(diff);
        if (abs_diff > max_err) max_err = abs_diff;
        sum_sq += diff * diff;
        ref_sq += ref * ref;
    }

    if (nan_count > 0)
        fprintf(stderr, "  warning: %d NaN/Inf values found in grid\n", nan_count);

    result.max_abs_error = max_err;
    result.l2_error = (ref_sq > 0.0) ? sqrt(sum_sq / ref_sq) : sqrt(sum_sq);
}

void ensure_csv_header(const std::string& filepath) {
    struct stat st;
    if (stat(filepath.c_str(), &st) != 0) {
        std::ofstream f(filepath);
        f << "timestamp,variant,dim,reach,grid_size,timesteps,elapsed_ms,max_abs_error,l2_error,bandwidth_gbs,memory_bytes\n";
        f.close();
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
    f.close();
}

void print_summary(const StencilResult& result) {
    printf("  variant:    %s\n", result.variant_name.c_str());
    if (result.dim == 3)
        printf("  grid:       %dx%dx%d\n", result.grid_size, result.grid_size, result.grid_size);
    else
        printf("  grid:       %dx%d\n", result.grid_size, result.grid_size);
    printf("  reach:      %d\n", result.stencil_reach);
    printf("  timesteps:  %d\n", result.timesteps);
    printf("  time:       %.2f ms\n", result.elapsed_ms);
    printf("  max error:  %.6e\n", result.max_abs_error);
    printf("  L2 error:   %.6e\n", result.l2_error);
    if (result.effective_bw_gbs > 0)
        printf("  bandwidth:  %.1f GB/s\n", result.effective_bw_gbs);
    printf("  memory:     %.2f MB\n", result.memory_bytes / (1024.0 * 1024.0));
}

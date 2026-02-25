#pragma once
#include <string>
#include <vector>

static const int MAX_REACH = 8;

struct StencilConfig {
    int nx;
    int ny;
    int nz;
    int dim;
    float dx;
    float dt;
    float k;
    int timesteps;
    float temp_initial;
    float temp_source;
    int stencil_reach;
    float fd_coeffs[MAX_REACH + 1];
    float stability_limit;
};

struct StencilResult {
    std::string variant_name;
    int grid_size;
    int dim;
    int stencil_reach;
    int timesteps;
    double elapsed_ms;
    double max_abs_error;
    double l2_error;
    double effective_bw_gbs;
    size_t memory_bytes;
    std::vector<float> final_grid;
};

StencilConfig default_config();

StencilResult run_cpu_fp64(const StencilConfig& cfg);
StencilResult run_cuda_fp32(const StencilConfig& cfg);
StencilResult run_cuda_fp16_naive(const StencilConfig& cfg);
StencilResult run_cuda_fp16_kahan(const StencilConfig& cfg);

StencilResult run_cpu_fp64_3d(const StencilConfig& cfg);
StencilResult run_cuda_fp32_3d(const StencilConfig& cfg);
StencilResult run_cuda_fp16_naive_3d(const StencilConfig& cfg);
StencilResult run_cuda_fp16_kahan_3d(const StencilConfig& cfg);

// shared memory tiling variants
StencilResult run_cuda_fp32_smem(const StencilConfig& cfg);
StencilResult run_cuda_fp16_naive_smem(const StencilConfig& cfg);
StencilResult run_cuda_fp16_kahan_smem(const StencilConfig& cfg);
StencilResult run_cuda_fp32_smem_3d(const StencilConfig& cfg);
StencilResult run_cuda_fp16_naive_smem_3d(const StencilConfig& cfg);
StencilResult run_cuda_fp16_kahan_smem_3d(const StencilConfig& cfg);

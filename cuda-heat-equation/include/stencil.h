#pragma once
#include <string>
#include <vector>

struct StencilConfig {
    int nx;
    int ny;
    float dx;
    float dt;
    float k;
    int timesteps;
    float temp_initial;
    float temp_source;
};

// shared result container - every variant fills this
struct StencilResult {
    std::string variant_name;
    int grid_size;
    int timesteps;
    double elapsed_ms;
    double max_abs_error;   // vs fp64 reference
    double l2_error;
    double effective_bw_gbs;
    size_t memory_bytes;
    std::vector<float> final_grid; // always converted to float for comparison
};

StencilConfig default_config();

// CPU reference (fp64 internally for ground truth)
StencilResult run_cpu_fp64(const StencilConfig& cfg);

// CUDA fp32
StencilResult run_cuda_fp32(const StencilConfig& cfg);

// CUDA fp16 naive (no Kahan)
StencilResult run_cuda_fp16_naive(const StencilConfig& cfg);

// CUDA fp16 with Kahan compensated summation
StencilResult run_cuda_fp16_kahan(const StencilConfig& cfg);

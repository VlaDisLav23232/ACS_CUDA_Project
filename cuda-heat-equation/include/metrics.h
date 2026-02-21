#pragma once
#include "stencil.h"
#include <string>
#include <vector>

// compute error metrics between a result and the ground truth
void compute_errors(StencilResult& result, const std::vector<float>& reference);

// append one result row to CSV file
void write_csv_row(const std::string& filepath, const StencilResult& result);

// write CSV header if file doesnt exist yet
void ensure_csv_header(const std::string& filepath);

// print a quick summary to stdout
void print_summary(const StencilResult& result);

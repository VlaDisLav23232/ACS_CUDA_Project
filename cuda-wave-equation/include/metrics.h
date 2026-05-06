#pragma once
#include "stencil.h"
#include <string>
#include <vector>

void compute_errors(StencilResult& result, const std::vector<float>& reference);
void write_csv_row(const std::string& filepath, const StencilResult& result);
void ensure_csv_header(const std::string& filepath);
void print_summary(const StencilResult& result);

#!/usr/bin/env bash
# Benchmark: compare baseline (global-memory) vs sliding-window shared memory
# for 3D stencil kernels across grid sizes and stencil reaches.
set -euo pipefail

BINARY="./build/heat_stencil"
CSV="results/smem_comparison.csv"

# Build if needed
if [[ ! -x "$BINARY" ]]; then
    echo "Binary not found, building..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)
fi

# Fresh CSV
rm -f "$CSV"

echo "========================================"
echo "  3D Shared Memory Sliding-Window Benchmark"
echo "========================================"
echo ""

# 3D benchmarks: compare fp32 vs fp32_smem, fp16 vs fp16_smem, kahan vs kahan_smem
for R in 1 4 8; do
    for N in 32 64 128; do
        echo "--- 3D  N=$N  R=$R ---"

        # Run each variant separately so they all get compared vs the same CPU reference
        for V in fp32 fp32_smem fp16_smem kahan_smem; do
            echo "  variant: $V"
            "$BINARY" -n "$N" -t 500 -d 3 -r "$R" -v "$V" -o "$CSV" 2>&1 | grep -E "variant:|time:|max error:|bandwidth:|sliding tile:"
        done
        echo ""
    done
done

echo ""
echo "All benchmarks written to $CSV"
echo ""
echo "To generate plots, run:"
echo "  python3 scripts/plot_results.py $CSV results"

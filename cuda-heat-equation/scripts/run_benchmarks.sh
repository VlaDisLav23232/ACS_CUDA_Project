#!/usr/bin/env bash
set -euo pipefail

BINARY="./build/heat_stencil"
CSV="results.csv"

if [[ ! -x "$BINARY" ]]; then
    echo "binary not found, building..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)
fi

GRIDS=(64 128 256 512 1024)
TIMESTEPS=(100 500 1000 5000)

for N in "${GRIDS[@]}"; do
    for T in "${TIMESTEPS[@]}"; do
        echo "=== N=$N  T=$T ==="
        "$BINARY" -n "$N" -t "$T" -v all -o "$CSV"
        echo ""
    done
done

echo "all benchmarks written to $CSV"

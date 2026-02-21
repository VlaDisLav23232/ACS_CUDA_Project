#!/usr/bin/env bash
set -euo pipefail

BINARY="./build/heat_stencil"
CSV="results/benchmarks.csv"

if [[ ! -x "$BINARY" ]]; then
    echo "binary not found, building..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)
fi

rm -f "$CSV"

echo "=== 2D benchmarks ==="
for R in 1 4 8; do
    for N in 64 128 256 512; do
        echo "--- 2D  N=$N  R=$R ---"
        "$BINARY" -n "$N" -t 1000 -d 2 -r "$R" -o "$CSV"
    done
done

echo ""
echo "=== 3D benchmarks ==="
for R in 1 4 8; do
    for N in 32 64 128; do
        echo "--- 3D  N=$N  R=$R ---"
        "$BINARY" -n "$N" -t 500 -d 3 -r "$R" -o "$CSV"
    done
done

echo ""
echo "all benchmarks written to $CSV"

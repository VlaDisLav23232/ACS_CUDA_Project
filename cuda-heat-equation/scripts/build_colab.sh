#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build-colab"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_ARCH="${CUDA_ARCH:-}"

detect_cuda_arch() {
    if [[ -n "${CUDA_ARCH}" ]]; then
        printf '%s\n' "${CUDA_ARCH}"
        return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local cc
        cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .')"
        if [[ "${cc}" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "${cc}"
            return 0
        fi
    fi

    if command -v python3 >/dev/null 2>&1; then
        local cc_py
        cc_py="$(python3 - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"{major}{minor}")
except Exception:
    pass
PY
)"
        if [[ "${cc_py}" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "${cc_py}"
            return 0
        fi
    fi

    printf '%s\n' "75"
}

ARCH="$(detect_cuda_arch)"
echo "Building cuda-heat-equation for CUDA architecture sm_${ARCH}"

cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"

cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo
echo "Build complete:"
echo "  ${BUILD_DIR}/heat_stencil"

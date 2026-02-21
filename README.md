# Compensated summation for stencil computation on CUDA with reduced-precision storage

A semester project for the ACS (Архітектура Комп'ютерних Систем) course. We solve the heat equation on an NVIDIA GPU in 2D and 3D with configurable stencil reach (R=1, 4, 8), storing data in float16 to save memory bandwidth, and using Kahan compensated summation to recover float32-level accuracy despite the reduced-precision storage.

## What this project does

Heat spreads through a material. We simulate this by discretizing the domain into a grid (NxN or NxNxN) and updating each cell using its neighbors (a stencil operation), thousands of times. The core question: **can we store the temperature data in half precision (16-bit float) instead of single precision (32-bit float) without losing accuracy?**

The answer is yes, if we use Kahan compensated summation. Without compensation, half-precision errors compound catastrophically over thousands of timesteps. With compensation, we recover the exact same accuracy as full float32 computation. This holds across all dimensions (2D, 3D) and all stencil reaches (R=1, 4, 8).

## Theory

### The heat equation

The heat equation in $d$ dimensions:

$$\frac{\partial u}{\partial t} = k \nabla^2 u$$

We use the FTCS (Forward Time, Central Space) finite difference scheme. For a stencil with reach $R$, the second derivative along each axis uses $2R+1$ points (R on each side + center). The central FD coefficients $c_0, c_1, \ldots, c_R$ approximate $\partial^2/\partial x^2$ with order $2R$:

$$\frac{\partial^2 u}{\partial x^2} \approx \frac{1}{\Delta x^2} \left[ c_0 u_i + \sum_{m=1}^{R} c_m (u_{i-m} + u_{i+m}) \right]$$

The full update in $d$ dimensions:

$$u^{n+1} = u^n + r \cdot \left[ d \cdot c_0 \cdot u^n + \sum_{\text{axis}} \sum_{m=1}^{R} c_m \left( u^n_{+m} + u^n_{-m} \right) \right]$$

where $r = k \cdot \Delta t / \Delta x^2$. The stencil is always axis-aligned (+ shape in 2D, star shape in 3D, no diagonals).

### Stability (CFL condition)

The stability limit depends on reach and dimensionality. We compute it from the worst-case eigenvalue of the FD operator via von Neumann analysis:

$$r_{max} = \frac{2}{d \cdot |\lambda(\pi)|} \quad \text{where} \quad \lambda(\pi) = c_0 + 2\sum_{k=1}^{R} c_k (-1)^k$$

| Reach | Order | 2D $r_{max}$ | 3D $r_{max}$ | Points (2D) | Points (3D) |
|---|---|---|---|---|---|
| R=1 | 2 | 0.250 | 0.167 | 5 | 7 |
| R=4 | 8 | 0.154 | 0.103 | 17 | 25 |
| R=8 | 16 | 0.135 | 0.090 | 33 | 49 |

Our code auto-computes $\Delta t$ with 80% safety margin: $\Delta t = 0.8 \cdot r_{max} \cdot \Delta x^2 / k$.

### FD coefficients

The coefficients are computed at runtime by solving the $R \times R$ linear system that arises from matching Taylor expansion terms. For example:

- R=1: $c_0 = -2, \; c_1 = 1$ (standard 3-point)
- R=4: $c_0 = -2.847, \; c_1 = 1.600, \; c_2 = -0.200, \; c_3 = 0.025, \; c_4 = -0.002$
- R=8: 9 coefficients, 16th-order accurate

### Float16 and the precision problem

IEEE 754 float16 has 10 bits of mantissa (~3.3 decimal digits) versus float32's 23 bits (~7 digits). When we store a stencil result as float16, we lose roughly 3-4 decimal digits every timestep. Over thousands of timesteps, these tiny rounding errors accumulate and the solution drifts far from the correct answer.

### Kahan compensated summation

Kahan's algorithm (1965) tracks what was lost in each rounding operation and adds it back next time:

1. Load the `__half` value from GPU memory and add the stored compensation to recover the "true" value
2. Do all stencil arithmetic in float32 (including loading and compensating all $2dR$ neighbors)
3. Store the result back as `__half` (this is lossy)
4. Compute what was lost: `compensation = exact_result - half2float(stored_half)`
5. Save the compensation (in float32) for the next timestep

## Results

We ran a comprehensive benchmark: 2D with N=64,128,256,512 and 3D with N=32,64,128, each at R=1,4,8, total of 84 measurements across 4 computation variants.

### 2D results (T=1000)

| N | Reach | Variant | Time (ms) | Max error | Bandwidth |
|---|---|---|---|---|---|
| 512 | R=1 | CPU fp64 | 334.1 | 0 (ref) | n/a |
| 512 | R=1 | CUDA fp32 | 29.3 | 9.16e-05 | 213 GB/s |
| 512 | R=1 | fp16 naive | 21.5 | **3.01** | 145 GB/s |
| 512 | R=1 | fp16+Kahan | 42.0 | **9.16e-05** | 223 GB/s |
| 512 | R=4 | CPU fp64 | 1190.5 | 0 (ref) | n/a |
| 512 | R=4 | CUDA fp32 | 50.2 | 1.83e-04 | 365 GB/s |
| 512 | R=4 | fp16 naive | 44.9 | **8.77** | 204 GB/s |
| 512 | R=4 | fp16+Kahan | 85.5 | **1.83e-04** | 321 GB/s |
| 512 | R=8 | CPU fp64 | 5522.1 | 0 (ref) | n/a |
| 512 | R=8 | CUDA fp32 | 80.5 | 1.07e-04 | 415 GB/s |
| 512 | R=8 | fp16 naive | 75.1 | **5.85** | 223 GB/s |
| 512 | R=8 | fp16+Kahan | 131.6 | **1.07e-04** | 381 GB/s |

### 3D results (T=500)

| N | Reach | Variant | Time (ms) | Speedup vs CPU | Max error | Bandwidth |
|---|---|---|---|---|---|---|
| 128 | R=1 | CPU fp64 | 1,970 | 1.0x | 0 (ref) | n/a |
| 128 | R=1 | CUDA fp32 | 164 | 12.0x | 5.72e-06 | 195 GB/s |
| 128 | R=1 | fp16 naive | 134 | 14.7x | **0.221** | 119 GB/s |
| 128 | R=1 | fp16+Kahan | 241 | 8.2x | **5.72e-06** | 199 GB/s |
| 128 | R=4 | CPU fp64 | 40,406 | 1.0x | 0 (ref) | n/a |
| 128 | R=4 | CUDA fp32 | 425 | 95.1x | 1.14e-05 | 212 GB/s |
| 128 | R=4 | fp16 naive | 374 | 108.1x | **0.375** | 120 GB/s |
| 128 | R=4 | fp16+Kahan | 755 | 53.5x | **1.14e-05** | 179 GB/s |
| 128 | R=8 | CPU fp64 | 369,728 | 1.0x | 0 (ref) | n/a |
| 128 | R=8 | CUDA fp32 | 674 | 548.6x | 4.73e-04 | 209 GB/s |
| 128 | R=8 | fp16 naive | 618 | 598.3x | **0.558** | 114 GB/s |
| 128 | R=8 | fp16+Kahan | 1,263 | 292.8x | **4.73e-04** | 167 GB/s |

### Analysis

**Kahan works perfectly across all configurations.** In every single one of 84 measurements, the fp16+Kahan variant produces identical max error to fp32. This holds for 2D and 3D, for R=1 and R=8, for N=32 and N=512. The compensation mechanism universally rescues half-precision from catastrophic drift.

**Naive fp16 error scales with grid size and reach.** At 2D N=512 R=8, the naive error is 5.85 (on a signal of magnitude ~200). Wider stencils amplify the effect because more rounded values are summed per step. This makes naive fp16 unusable for production simulations.

**GPU speedup is massive in 3D.** The CPU is single-threaded and must iterate over N^3 points. At N=128 R=8, the GPU achieves **549x speedup** for fp32 and **293x** for Kahan. Even with Kahan's overhead, the GPU is nearly 300x faster than the CPU for 3D R=8 problems.

**Higher reach = more arithmetic intensity = better cache utilization.** At 2D N=512, measured bandwidth exceeds the 192 GB/s DRAM peak (up to 415 GB/s for fp32 R=8). This is because L2 cache (1 MB on TU117) serves repeated neighbor reads. Wider stencils have more data reuse, so cache amplification increases with reach.

**3D keeps bandwidth near peak.** 3D N=128 gives the GPU enough work to saturate memory: fp32 reaches 195-212 GB/s, very close to the 192 GB/s theoretical DRAM peak. The Kahan variant achieves 167-199 GB/s despite extra reads/writes.

**Kahan overhead is ~1.7x.** Across all configurations, Kahan is about 1.5-1.9x slower than fp32 due to the compensation array reads/writes and extra boundary kernels. The tradeoff: fp32-level accuracy with half the temperature storage.

### Plots

All plots are generated into [cuda-heat-equation/results/](cuda-heat-equation/results/).

| Plot | Description |
|---|---|
| [summary.png](cuda-heat-equation/results/summary.png) | 4-panel overview: bandwidth by reach (2D/3D), Kahan vs naive accuracy, 3D speedup |
| [2d_accuracy.png](cuda-heat-equation/results/2d_accuracy.png) | Max error vs N for each reach (2D) |
| [2d_bandwidth.png](cuda-heat-equation/results/2d_bandwidth.png) | Effective bandwidth vs N (2D) |
| [2d_speedup.png](cuda-heat-equation/results/2d_speedup.png) | GPU speedup over CPU (2D) |
| [3d_accuracy.png](cuda-heat-equation/results/3d_accuracy.png) | Max error vs N for each reach (3D) |
| [3d_bandwidth.png](cuda-heat-equation/results/3d_bandwidth.png) | Effective bandwidth vs N (3D) |
| [3d_speedup.png](cuda-heat-equation/results/3d_speedup.png) | GPU speedup over CPU (3D) |

## Project structure

```
ACS_CUDA_Project/
  README.md                                    this file
  .gitignore                                   ignores build/, *.o, *.csv, *.png
  heat1d.py                                    Python 1D heat reference
  heat2d.py                                    Python 2D heat reference

  cuda-heat-equation/                          main C++/CUDA project
    CMakeLists.txt                             build system (sm_75, C++17)
    include/
      stencil.h                                StencilConfig + StencilResult structs
      fd_coefficients.h                        FD coefficient solver for arbitrary reach
      metrics.h                                error computation + CSV declarations
    src/
      main.cpp                                 CLI parsing, FD coefficients, 2D/3D dispatch
      heat2d_cpu.cpp                           CPU fp64 reference, variable reach
      heat2d_cuda.cu                           CUDA fp32 kernel, variable reach
      heat2d_cuda_fp16.cu                      fp16 naive + fp16 Kahan, variable reach
      heat3d_cpu.cpp                           3D CPU fp64 reference, variable reach
      heat3d_cuda.cu                           3D CUDA fp32, 8x8x4 thread blocks
      heat3d_cuda_fp16.cu                      3D fp16 naive + fp16 Kahan
      metrics.cpp                              error metrics, CSV writer, NaN handling
    scripts/
      run_benchmarks.sh                        sweep: 2D + 3D, R=1/4/8, multiple N
      plot_results.py                          7-plot visualization suite
    results/                                   benchmark CSV + generated plots
```

## How to build and run

### Prerequisites
- NVIDIA GPU with compute capability 7.5+ (tested on GTX 1650)
- NVIDIA driver 550+ with working `nvidia-smi`
- CUDA Toolkit 12.x (`nvcc`)
- CMake 3.18+
- GCC 13 or 14
- Python 3 with `pandas` and `matplotlib` (for plots)

### Build
```bash
cd cuda-heat-equation
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Run
```bash
# 2D, reach=1, 256x256, 1000 steps (backward compatible with Phase 1)
./build/heat_stencil -n 256 -t 1000

# 2D, wider stencil (8th-order FD)
./build/heat_stencil -n 256 -t 1000 -r 4

# 3D, reach=1, 64x64x64, 500 steps
./build/heat_stencil -n 64 -t 500 -d 3

# 3D, maximum reach, large grid
./build/heat_stencil -n 128 -t 500 -d 3 -r 8

# full benchmark sweep (takes ~7 minutes)
bash scripts/run_benchmarks.sh

# generate plots
python3 scripts/plot_results.py
```

### CLI flags
| Flag | Default | Description |
|---|---|---|
| `-n <size>` | 256 | Grid size (NxN for 2D, NxNxN for 3D) |
| `-t <steps>` | 5000 | Number of timesteps |
| `-d <dim>` | 2 | Dimensionality: 2 or 3 |
| `-r <reach>` | 1 | Stencil reach per axis: 1 to 8 |
| `-v <variant>` | all | `cpu`, `fp32`, `fp16`, `kahan`, or `all` |
| `-o <path>` | results/benchmarks.csv | CSV output path |

## Hardware

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 Mobile / Max-Q (TU117) |
| Compute capability | 7.5 |
| CUDA cores | 1024 (14 SMs x 64 + 128 special func) |
| VRAM | 4 GB GDDR6 |
| Memory bandwidth | 192 GB/s theoretical (12 Gbps x 128-bit bus / 8) |
| CPU | AMD Ryzen 5 5600H (6 cores, 12 threads) |
| RAM | 14 GB |
| OS | Debian 13 (trixie), kernel 6.12.73 |
| CUDA | 12.4.131, driver 550.163.01 |

## What was completed

### Phase 1 (complete)
- 2D heat equation solver with FTCS, 5-point stencil (R=1)
- Four variants: CPU fp64, CUDA fp32, CUDA fp16 naive, CUDA fp16+Kahan
- Benchmark framework with CSV output, error metrics, bandwidth measurement
- Benchmark sweep (80 data points) and 4-panel visualization

### Phase 2 (complete)
- Configurable stencil reach R=1 to 8 (axis-aligned, no diagonals)
- FD coefficient computation via Gaussian elimination at runtime
- Automatic CFL stability limit from von Neumann analysis
- 3D heat equation: CPU fp64 reference, CUDA fp32, fp16 naive, fp16+Kahan
- Full benchmark sweep (84 data points across 2D/3D, R=1/4/8)
- 7-plot visualization suite

## Next steps

1. **Shared memory tiling.** Current kernels read from global memory for every neighbor. Loading a tile into shared memory would reduce redundant DRAM reads, especially for wider stencils where R=8 means 49 loads per thread in 3D.

2. **OpenMP CPU baseline.** The current CPU reference is single-threaded. Adding `#pragma omp parallel for` would give a fairer CPU vs GPU comparison on the 6-core Ryzen 5 5600H (expect ~5x speedup).

3. **Temporal blocking.** Instead of one timestep per kernel launch, compute multiple timesteps inside a single kernel using shared memory. This reduces kernel launch overhead and global memory traffic. Particularly valuable for R=1 where arithmetic intensity is lowest.

4. **Roofline model analysis.** Plot our measured performance on a roofline chart (FLOP/s vs arithmetic intensity) to determine whether we are memory-bound or compute-bound for each (dim, reach) combination. This would guide which optimization to prioritize.

5. **Larger 3D grids.** N=256 in 3D (256^3 = 16.7M points, ~128 MB for fp32) would fit in our 4 GB VRAM and stress the GPU more. N=128 R=8 already shows ~549x speedup over CPU; N=256 would push this further.

6. **Wave equation / advection-diffusion.** Different PDEs have different stencil patterns. The FD coefficient infrastructure supports any central second-derivative stencil, so adding a wave equation ($u_{tt} = c^2 \nabla^2 u$) would demonstrate the generality of the Kahan compensation approach.

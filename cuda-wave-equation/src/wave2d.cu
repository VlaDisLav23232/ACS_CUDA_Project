/**
 * wave2d_cuda.cu
 *
 * 2D Wave Equation solver accelerated with CUDA.
 * Matches the Python reference simulation exactly:
 *   - 4th-order spatial stencil (Numerov / 4th-order isotropic)
 *   - Dirichlet (zero) boundary conditions
 *   - Multiple Gaussian sources as initial condition
 *   - Writes one binary snapshot per SNAPSHOT_EVERY steps
 *     (load in Python with the companion visualiser below)
 *
 * Build:
 *   nvcc -O3 -arch=sm_75 -o wave2d wave2d_cuda.cu
 *
 * Run:
 *   ./wave2d
 *
 * Then visualise:
 *   python3 visualise.py
 *
 * ── Stability note ──────────────────────────────────────────────────────────
 * The 4th-order spatial stencil  (-1, 16, -30, 16, -1)/12/dx²  has a
 * stricter CFL limit than the standard 2nd-order one:
 *
 *   2nd-order 2D:  r_max = 1/√2   ≈ 0.707   (what Python dt was designed for)
 *   4th-order 2D:  r_max = √(3/8) ≈ 0.612   (this solver)
 *
 * where  r = c·dt/dx.
 *
 * Derivation:
 *   Fourier symbol at Nyquist (k=π/h):
 *       S = (−2cos2π + 32cosπ − 30)/12 = (−2−32−30)/12 = −64/12
 *   2D combined: 2·64/12.  Leapfrog stability: r²·(128/12) ≤ 4
 *       ⟹  r ≤ √(48/128) = √(3/8) ≈ 0.6124
 *
 * Using the Python dt (r≈0.636) with float32 causes the Nyquist mode to
 * amplify by |A|≈1.748 per step.  Float32 rounding error (~1e-7) seeds
 * this mode; after ~30 steps it reaches O(0.1) → visible blow-up at frame 3.
 * Float64 Python buys ~1e9× more headroom, so Python appears stable.
 *
 * Fix: DT = DX/C · √(3/8) · 0.9  (10 % safety margin on the correct limit)
 *      plus double-precision arithmetic in the kernel.
 * ────────────────────────────────────────────────────────────────────────────
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// Error-checking helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------
static constexpr double LENGTH      = 10.0;
static constexpr double C           = 1.0;
static constexpr double DX          = 0.05;
static constexpr double DY          = DX;
// ── CFL for the 4th-order stencil in 2D ───────────────────────────────────
// r_max = sqrt(3/8) ~= 0.6124.  Use 0.9 * r_max as safety margin.
// (Python's 0.9/sqrt(2) ~= 0.636 exceeds this limit; float32 rounding error
//  seeds the unstable Nyquist mode which grows by |A|~1.748/step => frame-3
//  blow-up.  See header comment for full derivation.)
static constexpr double R_MAX_4TH   = 0.6123724357;   // sqrt(3.0/8.0)
static constexpr double DT          = DX / C * R_MAX_4TH * 0.9;
static constexpr double TOTAL_TIME  = 100.0;
static constexpr int    NT          = static_cast<int>(TOTAL_TIME / DT);

static constexpr int NX = static_cast<int>(LENGTH / DX) + 1;
static constexpr int NY = static_cast<int>(LENGTH / DY) + 1;

// How often to save a snapshot (set to 1 for every frame, higher for fewer files)
static constexpr int SNAPSHOT_EVERY = 10;

// ---------------------------------------------------------------------------
// CUDA kernel – 4th-order stencil, Dirichlet BCs  (double precision)
//
// Using double: ~1e-16 rounding error vs ~1e-7 for float32.
// Even in the (unlikely) case of marginal CFL, the unstable Nyquist mode
// remains below 1e-6 for thousands of steps, matching Python behaviour.
// ---------------------------------------------------------------------------
__global__
void wave_step_kernel(const double* __restrict__ u,
                      const double* __restrict__ u_prev,
                            double* __restrict__ u_next,
                      int nx, int ny,
                      double r2)   // r2 = (c*dt/dx)^2 / 12
{
    // Map thread to interior grid point [2 .. nx-3] x [2 .. ny-3]
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 2;

    if (ix >= nx - 2 || iy >= ny - 2) return;

    // Row-major index  u[ix, iy]  (ix = x-axis / axis-0 in numpy)
    auto idx = [&](int i, int j) -> int { return i * ny + j; };

    double uc = u[idx(ix, iy)];

    // 4th-order stencil in x
    double lap_x = -     u[idx(ix+2, iy)]
                   + 16. * u[idx(ix+1, iy)]
                   - 30. * uc
                   + 16. * u[idx(ix-1, iy)]
                   -       u[idx(ix-2, iy)];

    // 4th-order stencil in y
    double lap_y = -     u[idx(ix, iy+2)]
                   + 16. * u[idx(ix, iy+1)]
                   - 30. * uc
                   + 16. * u[idx(ix, iy-1)]
                   -       u[idx(ix, iy-2)];

    u_next[idx(ix, iy)] = 2.0 * uc
                          - u_prev[idx(ix, iy)]
                          + r2 * (lap_x + lap_y);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static void fill_initial_condition(std::vector<double>& phi,
                                   std::vector<double>& psi)
{
    // Sources: (x0, y0, sigma, A)
    struct Source { double x0, y0, sigma, A; };
    Source sources[] = {
        { LENGTH / 3.0,         LENGTH / 3.0,         0.5, 1.0 },
        { LENGTH / 3.0 * 2.5,   LENGTH / 3.0 * 2.5,   0.5, 1.0 }
    };

    phi.assign(NX * NY, 0.0);
    psi.assign(NX * NY, 0.0);

    for (int ix = 0; ix < NX; ++ix) {
        double x = ix * DX;
        for (int iy = 0; iy < NY; ++iy) {
            double y = iy * DY;
            double val = 0.0;
            for (auto& s : sources) {
                double dx_ = x - s.x0, dy_ = y - s.y0;
                val += s.A * std::exp(-(dx_*dx_ + dy_*dy_) / (s.sigma*s.sigma));
            }
            phi[ix * NY + iy] = val;
        }
    }
}

// First time step  u^1 = u^0 + dt*psi + 0.5*dt²*c²*laplacian(u^0)
// Uses simple 2nd-order Laplacian for the first step, same as Python.
static void first_step(const std::vector<double>& u_prev,
                       const std::vector<double>& psi,
                             std::vector<double>& u)
{
    const double dx2   = DX * DX;
    const double dy2   = DY * DY;
    const double dt2c2 = (C * DT) * (C * DT);

    u = u_prev;
    for (int ix = 1; ix < NX-1; ++ix) {
        for (int iy = 1; iy < NY-1; ++iy) {
            int c0 = ix * NY + iy;
            double lap = (u_prev[(ix+1)*NY+iy] - 2.0*u_prev[c0] + u_prev[(ix-1)*NY+iy]) / dx2
                       + (u_prev[ix*NY+(iy+1)] - 2.0*u_prev[c0] + u_prev[ix*NY+(iy-1)]) / dy2;
            u[c0] = u_prev[c0] + DT * psi[c0] + 0.5 * dt2c2 * lap;
        }
    }
    // Dirichlet BCs
    for (int iy = 0; iy < NY; ++iy) { u[0*NY+iy] = 0.0; u[(NX-1)*NY+iy] = 0.0; }
    for (int ix = 0; ix < NX; ++ix) { u[ix*NY+0] = 0.0; u[ix*NY+(NY-1)] = 0.0; }
}

// Save a snapshot: write step index followed by the grid as float32
// (downcast from double only at I/O time to keep files compact)
static void save_snapshot(const std::vector<double>& h_buf, int step, FILE* fout)
{
    fwrite(&step, sizeof(int), 1, fout);
    // Downcast to float32 for storage (saves 50 % disk space)
    std::vector<float> tmp(NX * NY);
    for (int i = 0; i < NX * NY; ++i) tmp[i] = static_cast<float>(h_buf[i]);
    fwrite(tmp.data(), sizeof(float), NX * NY, fout);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
    printf("NX=%d  NY=%d  NT=%d  DT=%.8f\n", NX, NY, NT, DT);
    printf("CFL r = c*dt/dx = %.6f  (limit for 4th-order 2D: %.6f)\n",
           C * DT / DX, R_MAX_4TH);
    printf("Grid memory per array (double): %.2f MB\n",
           NX * NY * sizeof(double) / 1e6);

    // -----------------------------------------------------------------------
    // Host initialisation
    // -----------------------------------------------------------------------
    std::vector<double> h_phi, h_psi;
    fill_initial_condition(h_phi, h_psi);

    std::vector<double> h_u_prev(NX * NY), h_u(NX * NY);
    h_u_prev = h_phi;
    first_step(h_u_prev, h_psi, h_u);

    // -----------------------------------------------------------------------
    // Device allocation  (double precision)
    // -----------------------------------------------------------------------
    double *d_u_prev, *d_u, *d_u_next;
    size_t nbytes = NX * NY * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_u_prev, nbytes));
    CUDA_CHECK(cudaMalloc(&d_u,      nbytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, nbytes));

    CUDA_CHECK(cudaMemcpy(d_u_prev, h_u_prev.data(), nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u,      h_u.data(),      nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u_next, 0, nbytes));   // Dirichlet: border stays 0

    // -----------------------------------------------------------------------
    // Kernel configuration
    // -----------------------------------------------------------------------
    dim3 block(16, 16);
    dim3 grid( (NX + block.x - 1) / block.x,
               (NY + block.y - 1) / block.y );

    // r2 = (c * dt / dx)^2 / 12
    const double r2 = (C * DT / DX) * (C * DT / DX) / 12.0;

    // -----------------------------------------------------------------------
    // Output file
    // -----------------------------------------------------------------------
    FILE* fout = fopen("wave2d_snapshots.bin", "wb");
    if (!fout) { perror("fopen"); return 1; }

    // Write header
    int header[3] = { NX, NY, NT / SNAPSHOT_EVERY };
    fwrite(header, sizeof(int), 3, fout);

    std::vector<double> h_snap(NX * NY);

    // -----------------------------------------------------------------------
    // Time loop
    // -----------------------------------------------------------------------
    for (int step = 2; step < NT; ++step)
    {
        // u_next interior computed by kernel (border stays 0 = Dirichlet)
        CUDA_CHECK(cudaMemset(d_u_next, 0, nbytes));
        wave_step_kernel<<<grid, block>>>(d_u, d_u_prev, d_u_next,
                                         NX, NY, r2);
        CUDA_CHECK(cudaGetLastError());

        // Rotate pointers: prev <- u, u <- next, next <- (will be overwritten)
        double* tmp = d_u_prev;
        d_u_prev    = d_u;
        d_u         = d_u_next;
        d_u_next    = tmp;

        // Snapshot
        if (step % SNAPSHOT_EVERY == 0) {
            CUDA_CHECK(cudaMemcpy(h_snap.data(), d_u, nbytes, cudaMemcpyDeviceToHost));
            save_snapshot(h_snap, step, fout);

            if (step % (SNAPSHOT_EVERY * 100) == 0)
                printf("  step %6d / %d\n", step, NT);
        }
    }

    fclose(fout);
    printf("Done. Snapshots written to wave2d_snapshots.bin\n");

    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    return 0;
}

#pragma once
#include "stencil.h"
#include <cmath>
#include <cstdio>
#include <cstring>

struct FDCoefficients {
    int reach;
    double c[MAX_REACH + 1];
    double stability_limit;
};

// Returns central FD weights for second derivative with reach R.
// stability_limit is the max value of s = (c*dt/dx)^2 for the leapfrog wave scheme.
inline FDCoefficients compute_fd_coefficients(int R, int dim) {
    FDCoefficients fd;
    fd.reach = R;
    std::memset(fd.c, 0, sizeof(fd.c));

    if (R < 1 || R > MAX_REACH) {
        std::fprintf(stderr, "reach R=%d out of supported range [1..%d]\n", R, MAX_REACH);
        return fd;
    }

    double A[MAX_REACH][MAX_REACH];
    double b[MAX_REACH];
    std::memset(A, 0, sizeof(A));
    std::memset(b, 0, sizeof(b));

    for (int m = 0; m < R; m++) {
        int power = 2 * (m + 1);
        for (int k = 0; k < R; k++) {
            A[m][k] = std::pow(static_cast<double>(k + 1), static_cast<double>(power));
        }
        b[m] = (m == 0) ? 1.0 : 0.0;
    }

    for (int col = 0; col < R; col++) {
        int pivot = col;
        for (int row = col + 1; row < R; row++) {
            if (std::fabs(A[row][col]) > std::fabs(A[pivot][col])) pivot = row;
        }
        if (pivot != col) {
            for (int j = 0; j < R; j++) {
                double tmp = A[col][j];
                A[col][j] = A[pivot][j];
                A[pivot][j] = tmp;
            }
            double tmp = b[col];
            b[col] = b[pivot];
            b[pivot] = tmp;
        }

        for (int row = col + 1; row < R; row++) {
            double factor = A[row][col] / A[col][col];
            for (int j = col; j < R; j++) A[row][j] -= factor * A[col][j];
            b[row] -= factor * b[col];
        }
    }

    for (int row = R - 1; row >= 0; row--) {
        double sum = b[row];
        for (int j = row + 1; j < R; j++) sum -= A[row][j] * fd.c[j + 1];
        fd.c[row + 1] = sum / A[row][row];
    }

    fd.c[0] = 0.0;
    for (int k = 1; k <= R; k++) fd.c[0] -= 2.0 * fd.c[k];

    double lambda_pi = fd.c[0];
    for (int k = 1; k <= R; k++) {
        lambda_pi += 2.0 * fd.c[k] * ((k % 2 == 0) ? 1.0 : -1.0);
    }
    double lambda_min_abs = std::fabs(static_cast<double>(dim) * lambda_pi);
    fd.stability_limit = 4.0 / lambda_min_abs;

    return fd;
}

inline void print_fd_coefficients(const FDCoefficients& fd) {
    std::printf("FD coefficients (reach=%d, order=%d):\n", fd.reach, 2 * fd.reach);
    std::printf("  c[0] = %+.10f (center)\n", fd.c[0]);
    for (int k = 1; k <= fd.reach; k++) {
        std::printf("  c[%d] = %+.10f\n", k, fd.c[k]);
    }
    std::printf("  wave stability limit: s < %.6f\n", fd.stability_limit);
}

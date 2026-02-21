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

// solve the linear system for central FD coefficients of the second derivative
// with reach R. returns coefficients c[0..R] where c[k] = c[-k] by symmetry.
// the system is:
//   c0 + 2*sum(c_k) = 0
//   sum(c_k * k^2)  = 1
//   sum(c_k * k^(2m)) = 0  for m=2..R
inline FDCoefficients compute_fd_coefficients(int R, int dim) {
    FDCoefficients fd;
    fd.reach = R;
    memset(fd.c, 0, sizeof(fd.c));

    if (R < 1 || R > MAX_REACH) {
        fprintf(stderr, "reach R=%d out of supported range [1..%d]\n", R, MAX_REACH);
        return fd;
    }

    // build and solve Rx R system for c[1]..c[R]
    // equations: sum_{k=1}^{R} c_k * k^{2m} = rhs_m
    //   m=1: rhs = 1
    //   m=2..R: rhs = 0

    double A[MAX_REACH][MAX_REACH];
    double b[MAX_REACH];
    memset(A, 0, sizeof(A));
    memset(b, 0, sizeof(b));

    for (int m = 0; m < R; m++) {
        int power = 2 * (m + 1); // powers 2, 4, 6, ...
        for (int k = 0; k < R; k++) {
            A[m][k] = pow((double)(k + 1), (double)power);
        }
        b[m] = (m == 0) ? 1.0 : 0.0;
    }

    // gaussian elimination with partial pivoting
    for (int col = 0; col < R; col++) {
        int pivot = col;
        for (int row = col + 1; row < R; row++) {
            if (fabs(A[row][col]) > fabs(A[pivot][col])) pivot = row;
        }
        if (pivot != col) {
            for (int j = 0; j < R; j++) {
                double tmp = A[col][j]; A[col][j] = A[pivot][j]; A[pivot][j] = tmp;
            }
            double tmp = b[col]; b[col] = b[pivot]; b[pivot] = tmp;
        }
        for (int row = col + 1; row < R; row++) {
            double factor = A[row][col] / A[col][col];
            for (int j = col; j < R; j++) A[row][j] -= factor * A[col][j];
            b[row] -= factor * b[col];
        }
    }

    // back substitution
    for (int row = R - 1; row >= 0; row--) {
        double sum = b[row];
        for (int j = row + 1; j < R; j++) sum -= A[row][j] * fd.c[j + 1];
        fd.c[row + 1] = sum / A[row][row];
    }

    // center weight from constraint: c0 + 2*sum(c_k) = 0
    fd.c[0] = 0.0;
    for (int k = 1; k <= R; k++) fd.c[0] -= 2.0 * fd.c[k];

    // stability limit for FTCS:
    // worst-case eigenvalue per axis at theta=pi:
    //   lambda = c0 + 2*sum(c_k * cos(k*pi)) = c0 + 2*sum(c_k * (-1)^k)
    // total over d dimensions: d * lambda
    // stability: |1 + (k*dt/dx^2) * d * lambda| <= 1
    // since lambda < 0: r * d * |lambda| <= 2, so r <= 2/(d*|lambda|)
    double lambda_pi = fd.c[0];
    for (int k = 1; k <= R; k++) {
        lambda_pi += 2.0 * fd.c[k] * ((k % 2 == 0) ? 1.0 : -1.0);
    }
    double abs_lambda = fabs(lambda_pi);
    fd.stability_limit = 2.0 / ((double)dim * abs_lambda);

    return fd;
}

inline void print_fd_coefficients(const FDCoefficients& fd) {
    printf("FD coefficients (reach=%d, order=%d):\n", fd.reach, 2 * fd.reach);
    printf("  c[0] = %+.10f (center)\n", fd.c[0]);
    for (int k = 1; k <= fd.reach; k++) {
        printf("  c[%d] = %+.10f\n", k, fd.c[k]);
    }
    printf("  stability limit: r < %.6f\n", fd.stability_limit);
}

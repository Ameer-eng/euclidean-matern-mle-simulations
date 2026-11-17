"""Numba kernels: parity-block assembly, irrep projection, sparse quadratic diagonals."""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def assemble_parity_block(X, k, G, w, tables, out):
    """out[tb, x, y] = w_x w_y sum_{c in {0,1}^d} s_c tables[tb, D_c(x, y)].

    For axis q the squared offset is (x_q - y_q)^2 if c_q = 0 and
    (G-1-x_q-y_q)^2 if c_q = 1; s_c = (-1)^{#{q < k : c_q = 1}} (odd axes first).
    Full rows are computed so the result is exactly symmetric and the work is
    balanced across threads.
    """
    N, d = X.shape
    ntab = tables.shape[0]
    ncomb = 1 << d
    signs = np.empty(ncomb)
    for c in range(ncomb):
        s = 1.0
        for q in range(k):
            if (c >> q) & 1:
                s = -s
        signs[c] = s
    for ix in prange(N):
        sq0 = np.empty(d, np.int64)
        sq1 = np.empty(d, np.int64)
        acc = np.empty(ntab)
        for iy in range(N):
            for q in range(d):
                a = X[ix, q]
                b = X[iy, q]
                e0 = a - b
                e1 = G - 1 - a - b
                sq0[q] = e0 * e0
                sq1[q] = e1 * e1
            for tb in range(ntab):
                acc[tb] = 0.0
            for c in range(ncomb):
                D = 0
                for q in range(d):
                    if (c >> q) & 1:
                        D += sq1[q]
                    else:
                        D += sq0[q]
                s = signs[c]
                for tb in range(ntab):
                    acc[tb] += s * tables[tb, D]
            ww = w[ix] * w[iy]
            for tb in range(ntab):
                out[tb, ix, iy] = acc[tb] * ww


@njit(parallel=True, cache=True)
def project_block(B, indptr, indices, data, T, R):
    """R = U^T B U for symmetric dense B (N x N) and U in CSC (N x M).

    T is an (M, N) workspace: T[a, :] = (B u_a)^T.  R is symmetrised.
    """
    M = indptr.shape[0] - 1
    N = B.shape[0]
    for a in prange(M):
        for j in range(N):
            T[a, j] = 0.0
        for p in range(indptr[a], indptr[a + 1]):
            x = indices[p]
            u = data[p]
            for j in range(N):
                T[a, j] += u * B[x, j]
    for a in prange(M):
        for b in range(M):
            s = 0.0
            for p in range(indptr[b], indptr[b + 1]):
                s += data[p] * T[a, indices[p]]
            R[a, b] = s
    for a in prange(M):
        for b in range(a + 1, M):
            v = 0.5 * (R[a, b] + R[b, a])
            R[a, b] = v
            R[b, a] = v


@njit(parallel=True, cache=True)
def add_quad_diag(indptr, indices, data, M, out):
    """out[x] += u_x^T M u_x for the rows u_x of a CSR matrix U (N x m)."""
    nrow = indptr.shape[0] - 1
    for x in prange(nrow):
        s = 0.0
        for p in range(indptr[x], indptr[x + 1]):
            a = indices[p]
            ua = data[p]
            for q in range(indptr[x], indptr[x + 1]):
                s += ua * M[a, indices[q]] * data[q]
        out[x] += s


@njit(parallel=True, cache=True)
def assemble_rows(X, rows, k, G, w, tables, out):
    """Rows `rows` of the parity block: out[tb, i, y] = B_tb[rows[i], y]
    (same formula as `assemble_parity_block`)."""
    N, d = X.shape
    nr = rows.shape[0]
    ntab = tables.shape[0]
    ncomb = 1 << d
    signs = np.empty(ncomb)
    for c in range(ncomb):
        s = 1.0
        for q in range(k):
            if (c >> q) & 1:
                s = -s
        signs[c] = s
    for ir in prange(nr):
        ix = rows[ir]
        sq0 = np.empty(d, np.int64)
        sq1 = np.empty(d, np.int64)
        acc = np.empty(ntab)
        for iy in range(N):
            for q in range(d):
                a = X[ix, q]
                b = X[iy, q]
                e0 = a - b
                e1 = G - 1 - a - b
                sq0[q] = e0 * e0
                sq1[q] = e1 * e1
            for tb in range(ntab):
                acc[tb] = 0.0
            for c in range(ncomb):
                D = 0
                for q in range(d):
                    if (c >> q) & 1:
                        D += sq1[q]
                    else:
                        D += sq0[q]
                s = signs[c]
                for tb in range(ntab):
                    acc[tb] += s * tables[tb, D]
            ww = w[ix] * w[iy]
            for tb in range(ntab):
                out[tb, ir, iy] = acc[tb] * ww


@njit(parallel=True, cache=True)
def project_block_rep(Brep, x_orb, x_ginv, act, indptr, indices, data, T, R):
    """R = U^T B U using only representative rows of the H-invariant matrix B:
    B[x, y] = Brep[x_orb[x], act[x_ginv[x], y]].  T is an (M, N) workspace."""
    M = indptr.shape[0] - 1
    N = Brep.shape[1]
    for a in prange(M):
        for j in range(N):
            T[a, j] = 0.0
        for p in range(indptr[a], indptr[a + 1]):
            x = indices[p]
            u = data[p]
            r = x_orb[x]
            g = x_ginv[x]
            for j in range(N):
                T[a, j] += u * Brep[r, act[g, j]]
    for a in prange(M):
        for b in range(M):
            s = 0.0
            for p in range(indptr[b], indptr[b + 1]):
                s += data[p] * T[a, indices[p]]
            R[a, b] = s
    for a in prange(M):
        for b in range(a + 1, M):
            v = 0.5 * (R[a, b] + R[b, a])
            R[a, b] = v
            R[b, a] = v


# ---------------------------------------------------------------- dense helpers
@njit(parallel=True, cache=True)
def fill_lower_from_upper(A):
    """Make a C-ordered A symmetric in place from its upper triangle.
    (dpotri's lower triangle, in Fortran order, is the upper triangle of A.T.)"""
    N = A.shape[0]
    for i in prange(N):
        for j in range(i):
            A[i, j] = A[j, i]


@njit(parallel=True, cache=True)
def sum_prod(A, B):
    """sum_ij A_ij B_ij (= tr(A B) for symmetric B), no temporary."""
    N, M = A.shape
    s = 0.0
    for i in prange(N):
        r = 0.0
        for j in range(M):
            r += A[i, j] * B[i, j]
        s += r
    return s


@njit(parallel=True, cache=True)
def trace_AB(A, B):
    """tr(A B) = sum_ij A_ij B_ji for square A, B (tiled for cache reuse)."""
    N = A.shape[0]
    bs = 64
    nb = (N + bs - 1) // bs
    s = 0.0
    for bi in prange(nb):
        i0 = bi * bs
        i1 = min(i0 + bs, N)
        r = 0.0
        for j0 in range(0, N, bs):
            j1 = min(j0 + bs, N)
            for i in range(i0, i1):
                for j in range(j0, j1):
                    r += A[i, j] * B[j, i]
        s += r
    return s

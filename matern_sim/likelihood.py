"""Exact Gaussian (quasi-)likelihood on the grid, computed blockwise.

theta = (eta, nu, log a, log t)  (spectral coordinates of eq. 3.1, log a, log t).

  ell(theta) = -1/2 { n log 2pi + sum_rho mult_rho log det B_rho
                       + sum_rho tr(B_rho^{-1} Z_rho Z_rho^T) },

where Z_rho (N_rho x mult_rho) collects the block coordinates Q^T Y belonging
to the copies of block rho.  Every block is processed independently, so peak
memory is set by the largest parity class.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import LinAlgError
from scipy.linalg.lapack import dpotrf, dpotri

from .assemble import (add_quad_diag, assemble_rows, fill_lower_from_upper,
                       project_block_rep, sum_prod, trace_AB)
from .kernel import matern_tables
from .symmetry import GridSymmetry

LOG2PI = math.log(2.0 * math.pi)


def chol_logdet_inv(B):
    """log det B and B^{-1} (C-ordered, symmetric).  B must be a symmetric,
    C-contiguous scratch matrix: it is overwritten (factorised in place via its
    Fortran-ordered transpose view, so no copy is made)."""
    c, info = dpotrf(B.T, lower=1, clean=0, overwrite_a=1)
    if info != 0:
        raise LinAlgError(f"dpotrf info={info}")
    logdet = 2.0 * np.log(np.diag(c)).sum()
    inv, info = dpotri(c, lower=1, overwrite_c=1)
    if info != 0:
        raise LinAlgError(f"dpotri info={info}")
    Binv = inv.T
    fill_lower_from_upper(Binv)
    return logdet, Binv


class GridModel:
    def __init__(self, d, G, sym=None, verbose=False):
        self.d, self.G = int(d), int(G)
        self.sym = sym if sym is not None else GridSymmetry(d, G, verbose=verbose)
        self.n = self.sym.n
        self.Dmax = self.sym.Dmax

    # -------------------------------------------------------------- blocks
    def class_signal_blocks(self, ci, tables):
        """Projected signal blocks of class ci for each row of `tables`:
        list over irreps of arrays (ntab, N_rho, N_rho)."""
        cls = self.sym.classes[ci]
        ntab = tables.shape[0]
        # only the rows at orbit representatives (B is invariant under H)
        P = np.empty((ntab, len(cls.reps), cls.N))
        assemble_rows(cls.X, cls.reps, cls.k, self.G, cls.w,
                      np.ascontiguousarray(tables), P)
        res = []
        Tmax = max(ir.N for ir in cls.irreps)
        Tbuf = np.empty(Tmax * cls.N)
        for ir in cls.irreps:
            U = ir.U[0]
            T = Tbuf[: ir.N * cls.N].reshape(ir.N, cls.N)
            out = np.empty((ntab, ir.N, ir.N))
            for tb in range(ntab):
                project_block_rep(P[tb], cls.x_orb, cls.x_ginv, cls.act,
                                  U.indptr, U.indices, U.data, T, out[tb])
            res.append(out)
        del P
        return res

    def iter_blocks(self, theta, which=("K", "Knu", "Ka")):
        """Yield (ci, ri, mult, signal matrices) for every distinct block."""
        tables = matern_tables(theta, self.d, self.G, self.Dmax, which)
        for ci, cls in enumerate(self.sym.classes):
            mats = self.class_signal_blocks(ci, tables)
            for ri, ir in enumerate(cls.irreps):
                yield ci, ri, cls.n_patterns * ir.f, mats[ri]
            del mats

    # ---------------------------------------------------- likelihood + score
    def loglik_grad(self, theta, Z, grad=True):
        t = math.exp(theta[3])
        which = ("K", "Knu", "Ka") if grad else ("K",)
        ll = -0.5 * self.n * LOG2PI
        g = np.zeros(4)
        for ci, ri, mult, S in self.iter_blocks(theta, which):
            Zb = Z[(ci, ri)]
            B = S[0].copy()
            B[np.diag_indices_from(B)] += t
            logdet, Binv = chol_logdet_inv(B)
            W = Binv @ Zb
            ll -= 0.5 * (mult * logdet + np.sum(Zb * W))
            if grad:
                for p in range(3):
                    dB = S[p]
                    g[p] -= 0.5 * (mult * sum_prod(Binv, dB) - np.sum(W * (dB @ W)))
                g[3] -= 0.5 * t * (mult * np.trace(Binv) - np.sum(W * W))
        return (ll, g) if grad else ll

    # ------------------------------------------------- Gaussian information
    def fisher(self, theta, kurtosis_matrix=False):
        """Expected Gaussian information I(theta) (4x4, spectral coordinates).

        With kurtosis_matrix=True also returns Dk with
          Dk_pq = sum_i (C^{-1} C_p C^{-1})_ii (C^{-1} C_q C^{-1})_ii,
        so that Cov(score) = I + (kappa4/4) Dk exactly (eq. A.11), where
        kappa4 = E eps^4 - 3 tau^4 is the error fourth cumulant.
        """
        t = math.exp(theta[3])
        I = np.zeros((4, 4))
        red = [np.zeros((4, cls.N)) for cls in self.sym.classes]
        for ci, ri, mult, S in self.iter_blocks(theta):
            B = S[0].copy()
            B[np.diag_indices_from(B)] += t
            _, Binv = chol_logdet_inv(B)
            A = [Binv @ S[p] for p in range(3)] + [t * Binv]
            for p in range(4):
                for q in range(p, 4):
                    v = 0.5 * mult * trace_AB(A[p], A[q])
                    I[p, q] += v
                    if q != p:
                        I[q, p] += v
            if kurtosis_matrix:
                ir = self.sym.classes[ci].irreps[ri]
                for p in range(4):
                    Mp = A[p] @ Binv
                    Mp = 0.5 * (Mp + Mp.T)
                    for Ucsr in ir.U_csr:
                        add_quad_diag(Ucsr.indptr, Ucsr.indices, Ucsr.data, Mp, red[ci][p])
            del A
        if not kurtosis_matrix:
            return I
        diag = np.stack([self.sym.grid_diagonal([red[ci][p] for ci in range(len(red))])
                         for p in range(4)])
        Dk = diag @ diag.T
        return I, Dk

    # -------------------------------------------- signal eigendecomposition
    def signal_eigen(self, theta):
        """{(ci, ri): (V, lam, mult)} for the signal blocks S_rho(theta)."""
        out = {}
        for ci, ri, mult, S in self.iter_blocks(theta, ("K",)):
            lam, V = np.linalg.eigh(S[0])
            out[(ci, ri)] = (V, np.maximum(lam, 0.0), mult)
        return out


def draw_field_blocks(eig, rng):
    """Latent Gaussian field in block coordinates, N(0, S(theta0))."""
    Z = {}
    for key in sorted(eig):
        V, lam, mult = eig[key]
        xi = rng.standard_normal((V.shape[0], mult))
        Z[key] = V @ (np.sqrt(lam)[:, None] * xi)
    return Z


def oracle_nugget_profile(eig, Z):
    """Pieces of the criterion in t with (eta, nu, a) fixed at the truth:
    ell(t) = -1/2 sum_i [ m_i log(lam_i + t) + s2_i / (lam_i + t) ] + const."""
    lam, s2, m = [], [], []
    for key in sorted(eig):
        V, l, mult = eig[key]
        W = V.T @ Z[key]
        lam.append(l)
        s2.append(np.sum(W * W, axis=1))
        m.append(np.full(l.shape, float(mult)))
    return np.concatenate(lam), np.concatenate(s2), np.concatenate(m)


# --------------------------------------------------------------------------
# Dense reference implementation (validation only; small n)
# --------------------------------------------------------------------------
def dense_covariance(d, G, theta):
    from scipy.spatial.distance import pdist, squareform
    g = (np.arange(G) + 0.5) / G
    pts = np.stack(np.meshgrid(*([g] * d), indexing="ij"), -1).reshape(-1, d)
    D2 = np.rint(squareform(pdist(pts, "sqeuclidean")) * G * G).astype(int)
    tab = matern_tables(theta, d, G, d * (G - 1) ** 2, ("K", "Knu", "Ka"))
    C = tab[0][D2] + math.exp(theta[3]) * np.eye(len(pts))
    dC = [tab[0][D2], tab[1][D2], tab[2][D2], math.exp(theta[3]) * np.eye(len(pts))]
    return C, dC


def dense_loglik(d, G, theta, y):
    C, _ = dense_covariance(d, G, theta)
    L = np.linalg.cholesky(C)
    a = np.linalg.solve(L, y)
    return -0.5 * (len(y) * LOG2PI + 2 * np.log(np.diag(L)).sum() + a @ a)

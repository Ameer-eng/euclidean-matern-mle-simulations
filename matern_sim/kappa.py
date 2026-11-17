"""Residual estimator of the error fourth cumulant (Lemma A.12, eq. A.54).

  A_hat = t_hat C(theta_hat)^{-1},   r_hat = A_hat Y,   v_hat_i = t_hat (A_hat)_ii,
  kappa_tilde = sum_i (r_i^4 - 6 v_i r_i^2 + 3 v_i^2) / sum_{i,j} (A_hat)_ij^4.

Everything is in the original observation (grid) basis: entrywise fourth powers are
not invariant under the orthogonal change of basis Q, so the blockwise inverse is
mapped back to the grid before they are taken (Appendix D.3).

  r_hat            t_hat Q (oplus_rho B_rho^{-1}) Q^T Y  (blockwise solve, then Q).
  diag C^{-1}      the grid diagonal of Q (oplus B^{-1}) Q^T, as for the kurtosis
                   matrix in GridModel.fisher.
  sum (C^{-1})^4   C^{-1} commutes with the hyperoctahedral group (axis reflections
                   and permutations), so the fourth-power sum of row g is constant
                   on the orbit of g.  One row per orbit is formed, at the
                   representatives g_1 <= ... <= g_d <= (G-1)/2, from the parity
                   blocks:
                     C^{-1}[g, h] = sum_p W_p[g] W_p[h] M_c(p)[X_p[g], X_p[h]],
                     M_c = sum_rho sum_i U_i B_rho^{-1} U_i^T   (parity block, class c),
                   where p runs over the 2^d parity patterns, X_p[g] is the reduced
                   canonical index of g in pattern p and W_p[g] the product of the
                   per-axis entries of the parity matrix (+-1/sqrt2, 1 or 0).
Cost per evaluation: one Cholesky inverse of every distinct block (about one
gradient evaluation) plus n * (number of orbits) * 2^d multiply-adds.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
from numba import njit, prange

from .assemble import add_quad_diag
from .likelihood import LOG2PI, chol_logdet_inv

_ROW_CHUNK = 2048


@njit(parallel=True, cache=True)
def _orbit_row_fourth(Rflat, base, wg, XT, WT, out, diag_out, rep_lin):
    """out[r] = sum_h (C^{-1}[g_r, h])^4 and diag_out[r] = C^{-1}[g_r, g_r]."""
    nrep, npat = base.shape
    n = XT.shape[0]
    for r in prange(nrep):
        s = 0.0
        g = rep_lin[r]
        for h in range(n):
            v = 0.0
            for p in range(npat):
                w = wg[r, p]
                if w != 0.0:
                    v += w * WT[h, p] * Rflat[base[r, p] + XT[h, p]]
            if h == g:
                diag_out[r] = v
            v2 = v * v
            s += v2 * v2
        out[r] = s


def grid_orbits(d, G):
    """Representatives (linear grid index) of the hyperoctahedral orbits on the
    midpoint grid, and the orbit sizes."""
    me = (G + 1) // 2
    centre = (G - 1) // 2 if G % 2 else None
    reps, sizes = [], []
    for a in itertools.combinations_with_replacement(range(me), d):
        perms = math.factorial(d)
        for c in Counter(a).values():
            perms //= math.factorial(c)
        refl = 2 ** sum(1 for x in a if x != centre)
        reps.append(np.ravel_multi_index(a, (G,) * d))
        sizes.append(perms * refl)
    sizes = np.array(sizes, dtype=np.int64)
    if sizes.sum() != G**d:
        raise RuntimeError(f"orbit sizes sum to {sizes.sum()} != {G**d}")
    return np.array(reps, dtype=np.int64), sizes


class ResidualCumulant:
    """kappa_tilde of eq. (A.54) for a GridModel; set up once per setting."""

    def __init__(self, model):
        self.model = model
        sym = model.sym
        d, G, n = model.d, model.G, model.n
        g = np.arange(G)
        centre = g == G - 1 - g
        je = np.minimum(g, G - 1 - g)
        jo = np.minimum(je, max(sym.mo - 1, 0))
        s = 1.0 / math.sqrt(2.0)
        we = np.where(centre, 1.0, s)                       # columns of the axis
        wo = np.where(centre, 0.0, np.where(g < G - 1 - g, s, -s))  # parity matrix

        def along(v, q):
            return v.reshape([G if r == q else 1 for r in range(d)])

        pats, X, W = [], [], []
        for ci, cls in enumerate(sym.classes):
            for eps, order in zip(cls.patterns, cls.orders):
                idx = [along(jo if eps[q] else je, q) for q in order]
                lin = np.ravel_multi_index(np.broadcast_arrays(*idx), cls.shape)
                wt = np.ones((1,) * d)
                for q in range(d):
                    wt = wt * along(wo if eps[q] else we, q)
                pats.append(ci)
                X.append(np.broadcast_to(lin, (G,) * d).reshape(-1))
                W.append(np.broadcast_to(wt, (G,) * d).reshape(-1))
        self.pat_class = np.array(pats)
        self.XT = np.ascontiguousarray(np.array(X, dtype=np.int64).T)   # (n, npat)
        self.WT = np.ascontiguousarray(np.array(W).T)                   # (n, npat)

        self.rep_lin, self.orbit_size = grid_orbits(d, G)
        wg = self.WT[self.rep_lin]                                      # (nrep, npat)
        xg = self.XT[self.rep_lin]
        # rows of each parity block M_c needed by some (representative, pattern)
        self.S, offset = [], 0
        base = np.zeros(wg.shape, dtype=np.int64)
        self.offsets = []
        for ci, cls in enumerate(sym.classes):
            cols = np.flatnonzero(self.pat_class == ci)
            use = wg[:, cols] != 0.0
            Sc = np.unique(xg[:, cols][use])
            pos = np.searchsorted(Sc, xg[:, cols])
            pos[~use] = 0
            base[:, cols] = offset + pos * cls.N
            self.S.append(Sc)
            self.offsets.append(offset)
            offset += len(Sc) * cls.N
        self.wg = np.ascontiguousarray(wg)
        self.base = base
        self.R_size = offset
        # rows S_c of every copy of every irrep basis U_i (CSR), and U_i (CSC)
        self.Usub = [[[Ucsr[self.S[ci]] for Ucsr in ir.U_csr] for ir in cls.irreps]
                     for ci, cls in enumerate(sym.classes)]

    def memory_gb(self):
        return 8.0 * self.R_size / 1024**3

    def compute(self, theta, Z):
        """kappa_tilde at theta for block data Z (= Q^T Y); also returns the
        Gaussian log-likelihood at theta, to check that Z is the fitted data."""
        model, sym = self.model, self.model.sym
        t = math.exp(theta[3])
        Rflat = np.zeros(self.R_size)
        red = [np.zeros(cls.N) for cls in sym.classes]
        Wz = {}
        ll = -0.5 * model.n * LOG2PI
        for ci, ri, mult, S in model.iter_blocks(theta, ("K",)):
            cls = sym.classes[ci]
            ir = cls.irreps[ri]
            Zb = Z[(ci, ri)]
            B = S[0].copy()
            B[np.diag_indices_from(B)] += t
            logdet, Binv = chol_logdet_inv(B)
            Wb = Binv @ Zb
            ll -= 0.5 * (mult * logdet + np.sum(Zb * Wb))
            Wz[(ci, ri)] = Wb
            for Ucsr in ir.U_csr:
                add_quad_diag(Ucsr.indptr, Ucsr.indices, Ucsr.data, Binv, red[ci])
            nS = len(self.S[ci])
            Rc = Rflat[self.offsets[ci]: self.offsets[ci] + nS * cls.N].reshape(nS, cls.N)
            for i in range(ir.f):
                Usub, U = self.Usub[ci][ri][i], ir.U[i]
                for a in range(0, nS, _ROW_CHUNK):
                    T = Usub[a:a + _ROW_CHUNK] @ Binv           # rows of U_i B^{-1}
                    Rc[a:a + _ROW_CHUNK] += (U @ T.T).T          # ... U_i^T
            del S, B, Binv
        r_hat = t * sym.inverse(Wz)
        cinv_diag = sym.grid_diagonal(red)
        v_hat = t * t * cinv_diag
        r2 = r_hat * r_hat
        numer = float(np.sum(r2 * r2 - 6.0 * v_hat * r2 + 3.0 * v_hat * v_hat))
        rows4 = np.empty(len(self.rep_lin))
        rdiag = np.empty(len(self.rep_lin))
        _orbit_row_fourth(Rflat, self.base, self.wg, self.XT, self.WT, rows4, rdiag,
                          self.rep_lin)
        del Rflat
        sum4 = float(np.dot(self.orbit_size, rows4))      # sum_ij (C^{-1})_ij^4
        denom = t**4 * sum4
        diag_err = float(np.max(np.abs(rdiag - cinv_diag[self.rep_lin]))
                         / np.max(np.abs(cinv_diag)))
        return dict(kappa=numer / denom, numer=numer, denom=denom, t_hat=t,
                    loglik=float(ll), sum_r4=float(np.sum(r2 * r2)),
                    sum_v=float(np.sum(v_hat)), denom_over_n=denom / model.n,
                    diag_check=diag_err)


def dense_kappa(d, G, theta, y):
    """Reference implementation with a dense inverse (validation only)."""
    from .likelihood import dense_covariance
    C, _ = dense_covariance(d, G, theta)
    t = math.exp(theta[3])
    A = t * np.linalg.inv(C)
    r = A @ y
    v = t * np.diag(A)
    numer = np.sum(r**4 - 6 * v * r**2 + 3 * v**2)
    denom = np.sum(A**4)
    return numer / denom, numer, denom

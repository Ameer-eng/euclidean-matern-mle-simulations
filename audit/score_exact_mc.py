"""FOLLOW-UP Task 1 (2026-11-10): exact Gaussian law of the score at the truth.

Under Gaussian errors Y ~ N(0, C0) exactly, so the score at theta0 in
w = (eta, nu, log a, log t) is the quadratic chaos
  s_p = 1/2 (X' Bt_p X - tr Bt_p),  X ~ N(0, I_n),  Bt_p = L^{-1} dC_p L^{-T}  (C0 = L L'),
computed block by block (the symmetry blocks are independent).  This script draws
the score directly from that law -- no data sets, no fits -- and reports the
distribution of ||S||^2 = s' I_w^{-1} s (invariant to the choice of E_n) against
chi-square(4), with many more draws than the 400 Monte Carlo replicates.

  python -m audit.score_exact_mc --d 5 --nu0 0.5 --G 10 --ndraw 20000
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import chi2, kurtosis, skew

from matern_sim.likelihood import GridModel
from matern_sim.run_task import truth

QS = (0.5, 0.9, 0.95, 0.99)


def draw_scores(d, nu0, G, ndraw, seed=12345, chunk=2000):
    model = GridModel(d, G)
    th0 = truth(d, nu0)
    t0 = math.exp(th0[3])
    rng = np.random.default_rng(seed)
    blocks = []
    for ci, ri, mult, S in model.iter_blocks(th0, ("K", "Knu", "Ka")):
        C = S[0].copy()
        C[np.diag_indices_from(C)] += t0
        L = np.linalg.cholesky(C)
        Bt = []
        for p in range(3):
            X = solve_triangular(L, S[p], lower=True)
            Bt.append(solve_triangular(L, X.T, lower=True).T)       # L^{-1} dC L^{-T}
        Ci = solve_triangular(L, np.eye(len(C)), lower=True)
        Bt.append(t0 * Ci @ Ci.T)                                    # t0 L^{-1} L^{-T}
        Bt = [0.5 * (b + b.T) for b in Bt]
        blocks.append((mult, Bt, np.array([np.trace(b) for b in Bt])))
    I = np.zeros((4, 4))
    for mult, Bt, _ in blocks:
        for p in range(4):
            for q in range(4):
                I[p, q] += 0.5 * mult * np.sum(Bt[p] * Bt[q])
    s = np.zeros((ndraw, 4))
    for c0 in range(0, ndraw, chunk):
        nd = min(chunk, ndraw - c0)
        for mult, Bt, tr in blocks:
            N = Bt[0].shape[0]
            X = rng.standard_normal((N, mult * nd))
            for p in range(4):
                q = np.sum(X * (Bt[p] @ X), axis=0).reshape(mult, nd).sum(0)
                s[c0:c0 + nd, p] += 0.5 * (q - mult * tr[p])
    return model.n, I, s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--nu0", type=float, nargs="+", default=[0.5])
    ap.add_argument("--G", type=int, nargs="+", default=[10, 8, 6])
    ap.add_argument("--ndraw", type=int, default=20000)
    ap.add_argument("--out", default="audit/out/followup")
    a = ap.parse_args(argv)
    rows = []
    crit = chi2.ppf(0.95, 4)
    for nu0 in a.nu0:
        for G in a.G:
            t = time.time()
            n, I, s = draw_scores(a.d, nu0, G, a.ndraw)
            th = np.load(f"results/theory/d{a.d}_nu{nu0:g}_G{G}.npz")
            Q = np.einsum("mi,ij,mj->m", s, np.linalg.inv(I), s)
            cov_ok = np.mean((Q <= crit).astype(float))
            se = math.sqrt(cov_ok * (1 - cov_ok) / len(Q))
            Ls = np.linalg.cholesky(np.linalg.inv(I))
            zs = s @ Ls                                       # a whitened version
            emp = np.cov(s.T)
            rows.append(dict(d=a.d, nu0=nu0, G=G, n=n, ndraw=len(Q),
                             I_rel_diff_theory=float(np.max(np.abs(I - th["I0"]) / np.abs(th["I0"]))),
                             cov_rel_dev_I=float(np.max(np.abs(emp - I) / np.sqrt(np.outer(np.diag(I), np.diag(I))))),
                             mean_Q=float(Q.mean()), var_Q=float(Q.var(ddof=1)),
                             **{f"q{int(100*q)}": float(np.quantile(Q, q)) for q in QS},
                             coverage=cov_ok, coverage_se=se,
                             max_abs_comp_skew=float(np.max(np.abs(skew(zs, axis=0)))),
                             max_abs_comp_exkurt=float(np.max(np.abs(kurtosis(zs, axis=0)))),
                             seconds=time.time() - t))
            print(rows[-1], flush=True)
    ref = dict(d=a.d, nu0=np.nan, G=np.nan, n=np.nan, ndraw=np.nan, mean_Q=4.0, var_Q=8.0,
               **{f"q{int(100*q)}": float(chi2.ppf(q, 4)) for q in QS}, coverage=0.95)
    df = pd.DataFrame([ref] + rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, f"task1_score_exact_mc_d{a.d}.csv"), index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""FOLLOW-UP Task 3 (2026-11-10): numerical error in the latent-field generation factor.

Generation path (run_task / kappa_task / hess0_task / score0_task / refit_task):
  GridModel.signal_eigen(theta0): for every distinct symmetry block rho,
      S_rho = projected signal block (assembled from the Matern kernel table),
      lam, V = numpy.linalg.eigh(S_rho);   lam <- max(lam, 0)      [the only modification]
  draw_field_blocks: F_rho = V diag(sqrt(lam)) Xi,  Xi iid N(0, 1)  (one column per copy).
No jitter, eigenvalue floor other than 0, or truncation is applied.  The covariance the
generation factor represents is  St_rho = V diag(max(lam, 0)) V'.

For the largest grid of each (d, nu0) this script reports, per block and in total,
  * the computed eigenvalue range, the number and size of negative (clipped) eigenvalues,
  * orthogonality ||V'V - I||_max and reconstruction ||V diag(lam) V' - S||_F / ||S||_F,
  * the whitened perturbation W = C^{-1/2} (St - S) C^{-1/2}, C = S + t0 I, through the
    orthogonally similar L^{-1}(St - S)L^{-T} (C = L L'):  max over blocks of ||W||_op and
    the multiplicity-weighted total ||W||_F^2 = sum_rho m_rho ||W_rho||_F^2,
  * the field-whitened S^{-1/2}(St - S)S^{-1/2} (generalized eigenvalues), and the
    total-variation bounds TV <= 1.5 ||.||_F (Devroye, Mehrabian and Reddad, 2018):
    with C-whitening for Gaussian errors, S-whitening for every error law (adding
    the same independent errors cannot increase TV),
  * a floor for the rounding error committed in forming St - S itself,
    eps * N_rho * ||S_rho||_op / t0, and the simple bound ||St - S||_F / t0,
  * global checks of the block assembly against the exact covariance (no n x n matrix):
      sum_rho m_rho tr S_rho = n K(0),
      sum_rho m_rho ||S_rho||_F^2 = sum over lag vectors prod_k (G - |delta_k|) K(lag)^2.

  python -m audit.field_generation_check --cases 5:0.5:10 4:1.5:18 --out audit/out/followup
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time

import numpy as np
import pandas as pd
from scipy.linalg import eigh as geigh, solve_triangular

from matern_sim.kappa_task import cpu_family
from matern_sim.kernel import matern_tables
from matern_sim.likelihood import GridModel
from matern_sim.run_task import truth

EPS = np.finfo(float).eps


def lag_frobenius(theta, d, G):
    """sum_{i,j} K(x_i - x_j)^2 and n K(0) from the lag distribution of the grid."""
    w1 = np.zeros((G - 1) ** 2 + 1)
    for dl in range(-(G - 1), G):
        w1[dl * dl] += G - abs(dl)
    w = np.array([1.0])
    for _ in range(d):
        w = np.convolve(w, w1)
    Dmax = d * (G - 1) ** 2
    K = matern_tables(theta, d, G, Dmax, ("K",))[0]
    return float(np.sum(w * K[: len(w)] ** 2)), float(G**d * K[0])


def check(d, nu0, G):
    t_start = time.time()
    model = GridModel(d, G)
    th0 = truth(d, nu0)
    t0 = math.exp(th0[3])
    rows = []
    trace_sum = fro2_sum = 0.0
    W_op = W_f2 = bound_f2 = floor_op = 0.0
    WS_op = WS_f2 = 0.0
    for ci, ri, mult, S in model.iter_blocks(th0, ("K",)):
        S0 = S[0]
        N = S0.shape[0]
        lam, V = np.linalg.eigh(S0)                    # exactly as signal_eigen
        lam_c = np.maximum(lam, 0.0)
        orth = float(np.max(np.abs(V.T @ V - np.eye(N))))
        recon = (V * lam) @ V.T - S0
        recon_rel = float(np.linalg.norm(recon) / np.linalg.norm(S0))
        D = (V * lam_c) @ V.T - S0                     # St - S as computed
        D = 0.5 * (D + D.T)
        C = S0.copy()
        C[np.diag_indices_from(C)] += t0
        L = np.linalg.cholesky(C)
        X = solve_triangular(L, D, lower=True)
        W = solve_triangular(L, X.T, lower=True).T
        W = 0.5 * (W + W.T)
        wev = np.linalg.eigvalsh(W)
        wop = float(np.max(np.abs(wev)))
        wf2 = float(np.sum(W * W))
        # field-whitened perturbation S^{-1/2}(St - S)S^{-1/2}: generalized eigenvalues
        # of (D, S); bounds TV between generated and exact field laws for every error law
        try:
            gev = geigh(D, S0, eigvals_only=True)
            ws_op, ws_f2 = float(np.max(np.abs(gev))), float(np.sum(gev * gev))
        except np.linalg.LinAlgError:
            ws_op = ws_f2 = math.nan
        neg = lam[lam < 0]
        rows.append(dict(d=d, nu0=nu0, G=G, n=model.n, block=f"{ci}.{ri}", N_rho=N, mult=mult,
                         lam_min=float(lam[0]), lam_max=float(lam[-1]),
                         n_neg=int(neg.size), max_abs_neg=float(-neg.min()) if neg.size else 0.0,
                         sum_abs_neg=float(-neg.sum()) if neg.size else 0.0,
                         n_below_eps_scale=int(np.sum(lam < EPS * N * lam[-1])),
                         orth_err_max=orth, recon_rel_F=recon_rel,
                         pert_F=float(np.linalg.norm(D)), W_op=wop, W_F=math.sqrt(wf2),
                         WS_op=ws_op, WS_F=math.sqrt(ws_f2),
                         floor_op=EPS * N * float(lam[-1]) / t0))
        trace_sum += mult * float(np.trace(S0))
        fro2_sum += mult * float(np.sum(S0 * S0))
        W_op = max(W_op, wop)
        W_f2 += mult * wf2
        WS_op = max(WS_op, ws_op)
        WS_f2 += mult * ws_f2
        bound_f2 += mult * float(np.sum(D * D)) / t0**2
        floor_op = max(floor_op, rows[-1]["floor_op"])
        del S0, V, recon, D, C, L, X, W
    fro2_exact, tr_exact = lag_frobenius(th0, d, G)
    br = pd.DataFrame(rows)
    tot = dict(d=d, nu0=nu0, G=G, n=model.n, host=socket.gethostname(), family=cpu_family(),
               threads=os.environ.get("OPENBLAS_NUM_THREADS", "?"), t0=t0,
               n_blocks=len(rows), n_block_copies=int(br.mult.sum()),
               largest_block=int(br.N_rho.max()),
               lam_min=float(br.lam_min.min()), lam_max=float(br.lam_max.max()),
               n_neg_distinct=int(br.n_neg.sum()), n_neg_total=int((br.n_neg * br.mult).sum()),
               n_blocks_with_neg=int((br.n_neg > 0).sum()),
               max_abs_neg=float(br.max_abs_neg.max()),
               max_orth_err=float(br.orth_err_max.max()), max_recon_rel_F=float(br.recon_rel_F.max()),
               W_op=W_op, W_F=math.sqrt(W_f2), W_F_over_sqrt_n=math.sqrt(W_f2 / model.n),
               WS_op=WS_op, WS_F=math.sqrt(WS_f2),
               TV_bound_gauss_errors=min(1.0, 1.5 * math.sqrt(W_f2)),
               TV_bound_any_errors=min(1.0, 1.5 * math.sqrt(WS_f2)),
               bound_op_pertF_over_t0=float(br.pert_F.max()) / t0,
               bound_F_pertF_over_t0=math.sqrt(bound_f2),
               rounding_floor_op=floor_op,
               trace_rel_err=abs(trace_sum - tr_exact) / tr_exact,
               frob2_rel_err=abs(fro2_sum - fro2_exact) / fro2_exact,
               seconds=time.time() - t_start)
    return br, tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True, help="d:nu0:G ...")
    ap.add_argument("--out", default="audit/out/followup")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    fam = cpu_family()
    for c in a.cases:
        d, nu0, G = c.split(":")
        br, tot = check(int(d), float(nu0), int(G))
        stem = f"task3_fieldgen_d{d}_nu{float(nu0):g}_G{G}_{fam}"
        br.to_csv(os.path.join(a.out, stem + "_blocks.csv"), index=False)
        with open(os.path.join(a.out, stem + "_total.json"), "w") as fh:
            json.dump(tot, fh, indent=1)
        print(json.dumps(tot), flush=True)


if __name__ == "__main__":
    main()

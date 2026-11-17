"""APPENDIX E QUESTIONS (2026-11-16), Q6: coverage and invalidity summaries.

Reads only saved summaries and the per-fit audit table; no fitting.
  (a) d=3 block-interval coverage for log nu and log c at n >= 1e4, for the four
      curvatures (Fisher/Hessian at the fit / at the truth), Gaussian and all laws;
  (b) Table E5 (tab:sim-se): total, valid and covered counts for every entry;
  (c) per (d, nu0, n, law) failure counts of three distinct kinds:
        - indefinite uncorrected full curvature (min eigenvalue <= 0 of the saved
          4x4 matrix in w = (eta, nu, log a, log t); fits_audit.pkl),
        - invalid corrected marginal variance (nfit - n_se_<method>, summary.csv),
        - non-PD / nonfinite corrected joint covariance (joint.csv, log_vector).

  python -m audit.appendixE_q6
Outputs in audit/out/appendixE/.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

OUT = "audit/out/appendixE"
S = pd.read_csv("results/summary/summary.csv")
J = pd.read_csv("results/summary/joint.csv")
F = pd.read_pickle("audit/out/fits_audit.pkl")
S = S[S.variant == "joint"]
J = J[J.variant == "joint"]

E5_METHODS = ["fisher_full_khat", "hess_full_khat", "fisher0_full_khat", "hess0_full_khat",
              "fisher_block_khat", "hess_block_khat", "fisher0_block_khat", "hess0_block_khat"]


def part_a():
    s = S[(S.d == 3) & (S.n >= 10_000) & S.param.isin(["log_nu", "log_c"])]
    rows = []
    for m in ["fisher_block_khat", "fisher0_block_khat", "hess_block_khat", "hess0_block_khat"]:
        for scope, x in [("gauss", s[s.law == "gauss"]), ("all laws", s)]:
            i = x[f"cov_{m}"].idxmax()
            rows.append(dict(method=m, laws=scope, max_cov_pct=100 * x[f"cov_{m}"].max(),
                             at=f"nu0={x.loc[i, 'nu0']} n={x.loc[i, 'n']} {x.loc[i, 'law']} {x.loc[i, 'param']}",
                             min_n_valid=int(x[f"n_se_{m}"].min())))
    t = pd.DataFrame(rows)
    print("(a) d=3, n>=1e4, log nu and log c: largest block-interval coverage")
    print(t.round(2).to_string(index=False))
    g = s[s.law == "gauss"].pivot_table(index=["nu0", "n"], columns="param",
                                        values=["cov_fisher_block_khat", "cov_fisher0_block_khat"]).mul(100)
    print("\nGaussian, by n:\n" + g.round(2).to_string())
    t.to_csv(os.path.join(OUT, "q6a_block_coverage_max.csv"), index=False)


def part_b():
    s = S[(S.law == "gauss")]
    s = s[s.n == s.groupby("d").n.transform("max")]
    rows = []
    for _, r in s.iterrows():
        for m in E5_METHODS:
            if "block" in m and r.d != 3:
                continue
            nv, c = r[f"n_se_{m}"], r[f"cov_{m}"]
            cov_n = c * nv
            rows.append(dict(d=r.d, nu0=r.nu0, n=r.n, param=r.param, method=m, total=int(r.nfit),
                             valid=int(nv), covered=int(round(cov_n)),
                             integer_check=abs(cov_n - round(cov_n)) < 1e-6,
                             coverage_pct=100 * c, invalid_pct=100 * (1 - nv / r.nfit)))
    t = pd.DataFrame(rows)
    assert t.integer_check.all()
    t.drop(columns="integer_check").to_csv(os.path.join(OUT, "q6b_table_E5_counts.csv"), index=False)
    w = t.assign(cell=t.covered.astype(str) + "/" + t.valid.astype(str))
    print("\n(b) Table E5 entries as covered/valid (total = 400 throughout:",
          bool((t.total == 400).all()), ")")
    print(w.pivot_table(index=["d", "nu0", "param"], columns="method", values="cell", aggfunc="first")
          [E5_METHODS].to_string())
    nonfull = t[(t.valid < t.total)]
    print("\nentries with fewer than 400 valid:")
    print(nonfull.groupby("method").size().to_string())


def part_c():
    f = F.assign(indef_H=F.minev_negH <= 0, indef_F=F.minev_fisher <= 0, indef_H0=F.minev_negH0 <= 0)
    a = f.groupby(["d", "nu0", "n", "law"]).agg(nfit=("rep", "size"), indef_H=("indef_H", "sum"),
                                               indef_F=("indef_F", "sum"), indef_H0=("indef_H0", "sum"))
    marg = {}
    for m in ["fisher_full_khat", "fisher_full_none", "hess_full_khat", "hess_full_none"]:
        # a fit is "invalid" for the marginal summary if any identifiable parameter's variance is
        # invalid; summary.csv only stores per-parameter counts, so report the max over params
        # (lower bound on the per-fit count) and the per-parameter sum (upper bound).
        g = S.assign(bad=S.nfit - S[f"n_se_{m}"]).groupby(["d", "nu0", "n", "law"]).bad
        marg[f"marg_invalid_{m}_maxparam"] = g.max()
        marg[f"marg_invalid_{m}_tau2"] = S[S.param == "log_tau2"].set_index(
            ["d", "nu0", "n", "law"]).pipe(lambda x: x.nfit - x[f"n_se_{m}"])
    a = a.join(pd.DataFrame(marg))
    jj = J[J.vector == "log_vector"]
    for m in ["fisher_full_khat", "hess_full_khat", "fisher0_full_khat", "hess0_full_khat"]:
        x = jj[jj.method == m].set_index(["d", "nu0", "n", "law"])
        a[f"joint_notpd_{m}"] = x.n_invalid_not_pd
        a[f"joint_nonfinite_{m}"] = x.n_invalid_nonfinite
    a = a.reset_index()
    a.to_csv(os.path.join(OUT, "q6c_failure_types.csv"), index=False)
    print("\n(c) totals over all 168 (d, nu0, n, law) cells:")
    tot = a.drop(columns=["d", "nu0", "n", "law"]).sum()
    print(tot.to_string())
    print("\nlargest n with any failure, per column:")
    for c in a.columns[4:]:
        x = a[a[c] > 0]
        print(f"  {c:42s} {'none' if x.empty else int(x.n.max())}")
    print("\nby (d, nu0, n), summed over laws, rows with any failure (fitted curvature and Fisher-based):")
    cols = ["nfit", "indef_H", "indef_F", "marg_invalid_hess_full_none_maxparam",
            "marg_invalid_hess_full_khat_maxparam", "marg_invalid_fisher_full_none_maxparam",
            "marg_invalid_fisher_full_khat_maxparam", "joint_notpd_fisher_full_khat",
            "joint_nonfinite_fisher_full_khat", "joint_notpd_hess_full_khat",
            "joint_nonfinite_hess_full_khat"]
    b = a.groupby(["d", "nu0", "n"])[cols].sum()
    print(b[(b[cols[1:]] > 0).any(axis=1)].to_string())
    # overlap: per-fit, indefinite H vs fisher-based validity is not stored per fit in summary.csv;
    # replicates.csv has per-fit SEs, so compute per-fit invalid marginal flags there.
    R = pd.read_csv("results/summary/replicates.csv")
    R = R[R.variant == "joint"]
    for m in ["fisher_full_khat", "fisher_full_none", "hess_full_khat", "hess_full_none"]:
        bad = np.zeros(len(R), bool)
        for p in ["log_nu", "log_c", "log_sigma2", "log_ell", "log_tau2"]:
            use = R.d.isin([3]) if p == "log_c" else (R.d >= 4 if p in ("log_sigma2", "log_ell") else True)
            se = R[f"{p}_se_{m}"].to_numpy(float)
            bad |= np.asarray(use) & ~(np.isfinite(se) & (se > 0))
        R[f"anybad_{m}"] = bad
    key = ["d", "nu0", "n", "law", "rep"]
    M = R[key + [c for c in R.columns if c.startswith("anybad_")]].merge(
        f[key + ["indef_H", "indef_F"]], on=key, how="inner")
    assert len(M) == 67_200, len(M)
    print("\nper-fit cross-tabulation (all 67,200 fits):")
    print("  fits with indefinite fitted Hessian:", int(M.indef_H.sum()),
          "| of these with every uncorrected Hessian marginal valid:",
          int((M.indef_H & ~M.anybad_hess_full_none).sum()))
    print("  fits with any invalid uncorrected Hessian marginal:", int(M.anybad_hess_full_none.sum()),
          "| of these with PD full Hessian:", int((~M.indef_H & M.anybad_hess_full_none).sum()))
    print("  fits with any invalid corrected Hessian marginal:", int(M.anybad_hess_full_khat.sum()))
    print("  fits with indefinite fitted Fisher:", int(M.indef_F.sum()))
    print("  fits with any invalid uncorrected Fisher marginal:", int(M.anybad_fisher_full_none.sum()))
    print("  fits with any invalid corrected Fisher marginal:", int(M.anybad_fisher_full_khat.sum()))
    inv = M[M.anybad_fisher_full_khat]
    print("    by (d, nu0, n, law):")
    print(inv.groupby(["d", "nu0", "n", "law"]).size().to_string())
    M.to_csv(os.path.join(OUT, "q6c_per_fit_flags.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
    part_a()
    part_b()
    part_c()

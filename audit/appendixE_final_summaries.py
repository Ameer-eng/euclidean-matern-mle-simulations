"""APPENDIX E FINAL REQUESTS (2026-11-16), item 2(b)-(d): exact scope of three
numerical statements in E.5.1. Reads saved outputs only.

Sources
  results/summary/summary.csv     one row per (d, nu0, n, law, param); for each
                                  standard-error method m:
                                    cov_m      = covered / valid  (nanmean over the
                                                 replicates with a finite SE)
                                    n_se_m     = # replicates whose SE is finite and > 0
                                    se_ratio_m = median(valid SEs) / RMSE
  audit/out/appendixE/q6c_perfit_small_n.csv
                                  per-fit validity flags (audit/appendixE_q6_perfit.py)

Method names: <curvature>_<inverse>_<correction>
  curvature  fisher  = Fisher information at theta_hat,  hess  = negative Hessian at theta_hat,
             fisher0 = Fisher information at theta_0,    hess0 = negative Hessian at theta_0;
  inverse    full  = identifiable block of the inverse of the full 4x4 curvature in
                     w = (eta, nu, log a, log t), mapped to the reported coordinates by
                     the delta method (reporting Jacobian);
  correction khat  = fitted cumulant correction (kappa-tilde) added to the nugget block,
             none  = no correction.
An SE is invalid if the delta-mapped variance is non-finite or <= 0.

  python -m audit.appendixE_final_summaries
Outputs: audit/out/appendixE/final_*.csv and stdout (final_summaries.txt).
"""
import os

import numpy as np
import pandas as pd

OUT = "audit/out/appendixE"
LAWS = ["gauss", "unif", "mixnorm", "bern"]


def counts(S, m):
    """covered, valid and total counts per summary row for method m."""
    valid = S[f"n_se_{m}"].astype(int)
    covered = np.rint(S[f"cov_{m}"] * valid).astype(int)
    assert np.allclose(covered / valid.where(valid > 0), S[f"cov_{m}"], atol=1e-12, equal_nan=True)
    return covered, valid, S.nfit.astype(int)


def item_b(S):
    print("=" * 78)
    print("2(b)  d=5, nu0=1/2, n=1024, interval for log nu: Fisher vs Hessian at the fit")
    print("      (full inverse in w, delta-mapped; fitted cumulant correction khat;")
    print("       the correction touches only the nugget block, so log nu is the same")
    print("       with and without it)")
    x = S[(S.d == 5) & (S.nu0 == 0.5) & (S.n == 1024) & (S.param == "log_nu")].set_index("law").loc[LAWS]
    rows = []
    for m in ("fisher_full_khat", "hess_full_khat", "fisher_full_none", "hess_full_none"):
        c, v, t = counts(x, m)
        for law in LAWS:
            rows.append(dict(method=m, law=law, covered=c[law], valid=v[law], nfit=t[law],
                             coverage_pct=100 * c[law] / v[law]))
    r = pd.DataFrame(rows)
    print(r.to_string(index=False, float_format="%.2f"))
    r.to_csv(os.path.join(OUT, "final_2b_d5_nu05_n1024_lognu.csv"), index=False)


def item_c(S, P):
    print("\n" + "=" * 78)
    print("2(c)  invalid marginal variances, Hessian at the fit (full inverse)")
    rows = []
    for corr in ("khat", "none"):
        m = f"hess_full_{corr}"
        c, v, t = counts(S, m)
        x = S[["d", "nu0", "n", "law", "param"]].copy()
        x["invalid"], x["nfit"] = t - v, t
        x["pct"] = 100 * x.invalid / x.nfit
        x["corr"] = corr
        rows.append(x)
    C = pd.concat(rows)
    C.to_csv(os.path.join(OUT, "final_2c_cells.csv"), index=False)
    for corr in ("khat", "none"):
        x = C[C["corr"] == corr]
        print(f"\n  [{corr}] (A) per parameter-and-law cell: invalid / 400 for one parameter,"
              " one law, one (d, nu0, n)")
        for d in (3, 4, 5):
            xd = x[x.d == d]
            i = xd.pct.idxmax() if False else xd.pct.values.argmax()
            top = xd.iloc[i]
            att = xd[xd.pct == xd.pct.max()][["nu0", "n", "law", "param", "invalid"]]
            print(f"    d={d}: max {top.pct:.2f}% ({int(top.invalid)}/400); attained at\n"
                  + att.to_string(index=False).replace("\n", "\n      ").join(["      ", ""]))
        # (B) pooled over parameters and laws within a (d, nu0, n) setting
        g = x.groupby(["d", "nu0", "n"]).agg(invalid=("invalid", "sum"), total=("nfit", "sum"))
        g["pct"] = 100 * g.invalid / g.total
        print(f"  [{corr}] (B) pooled over parameters and laws within (d, nu0, n): max by d")
        print("    " + g.groupby("d").pct.max().round(3).to_string().replace("\n", "\n    "))
    # (C) per data set: any identifiable parameter invalid, per (d, nu0, n, law) and pooled over laws
    print("\n  (C) per data set: share of data sets with >= 1 invalid marginal variance")
    for corr in ("khat", "none"):
        col = f"hess_marg_invalid_{corr}"
        a = P.groupby(["d", "nu0", "n", "law"])[col].agg(["sum", "size"])
        a["pct"] = 100 * a["sum"] / a["size"]
        b = P.groupby(["d", "nu0", "n"])[col].agg(["sum", "size"])
        b["pct"] = 100 * b["sum"] / b["size"]
        print(f"    [{corr}] per (d, nu0, n, law), max by d: "
              f"{a.groupby('d').pct.max().round(2).to_dict()}; pooled over laws, max by d: "
              f"{b.groupby('d').pct.max().round(3).to_dict()}")
        print(f"    [{corr}] total data sets with >= 1 invalid: {int(P[col].sum())} of {len(P)} "
              "(all fits with n <= 343; none at larger n, see summary.csv)")
    big = C[(C.n > 343) & (C.invalid > 0)]
    print(f"  cells with any invalid at n > 343: {len(big)}")
    # nugget share and correction-created invalids
    k = C[(C["corr"] == "khat")]
    print(f"\n  khat: invalid marginal variances by parameter (sum over cells): "
          f"{k.groupby('param').invalid.sum().to_dict()}")
    mn, mk = P.hess_marg_invalid_none, P.hess_marg_invalid_khat
    print(f"  data sets made invalid by the correction (valid uncorrected, invalid corrected): "
          f"{int((mk & ~mn).sum())}; invalid in both: {int((mk & mn).sum())}; "
          f"invalid uncorrected only: {int((mn & ~mk).sum())}")
    # indefinite full Hessian
    ind = P.groupby(["d", "nu0", "n", "law"]).hess_indefinite.agg(["sum", "size"])
    ind["pct"] = 100 * ind["sum"] / ind["size"]
    print(f"  indefinite full Hessian: {int(P.hess_indefinite.sum())} of 67,200 fits "
          f"(all in the {len(P)} fits with n <= 343); max per (d, nu0, n, law): "
          f"{ind.pct.max():.2f}% at {ind.pct.idxmax()}")
    print("    by d: " + ind.groupby("d")["sum"].sum().to_string().replace("\n", ", "))


def item_d(S):
    print("\n" + "=" * 78)
    print("2(d)  median SEs of the three full versions at the largest n, khat correction")
    M = ["fisher_full_khat", "hess_full_khat", "fisher0_full_khat"]
    big = S[S.n == S.groupby("d").n.transform("max")].copy()
    for m in M:
        big[f"med_{m}"] = big[f"se_ratio_{m}"] * big.rmse
    med = big[[f"med_{m}" for m in M]]
    big["max_over_min_minus_1"] = med.max(axis=1) / med.min(axis=1) - 1
    big["H_vs_F"] = big[f"med_hess_full_khat"] / big[f"med_fisher_full_khat"] - 1
    big["F0_vs_F"] = big[f"med_fisher0_full_khat"] / big[f"med_fisher_full_khat"] - 1
    big["cov_F_minus_H_pp"] = 100 * (big.cov_fisher_full_khat - big.cov_hess_full_khat)
    cols = ["d", "nu0", "n", "law", "param", "med_fisher_full_khat", "med_hess_full_khat",
            "med_fisher0_full_khat", "max_over_min_minus_1", "H_vs_F", "F0_vs_F",
            "n_se_fisher_full_khat", "n_se_hess_full_khat", "n_se_fisher0_full_khat",
            "cov_F_minus_H_pp"]
    big[cols].to_csv(os.path.join(OUT, "final_2d_median_se_largest_n.csv"), index=False)
    print(f"  cells: {len(big)} = (d, nu0) x 4 laws x identifiable parameters; every SE valid: "
          f"{bool((big[[f'n_se_{m}' for m in M]] == 400).all().all())}")
    for lab, x in (("all four laws", big), ("Gaussian only", big[big.law == "gauss"])):
        i = x.max_over_min_minus_1.idxmax()
        r = x.loc[i]
        print(f"  {lab}: cellwise max(med)/min(med) - 1 = {100 * r.max_over_min_minus_1:.2f}% at "
              f"d={r.d} nu0={r.nu0} n={r.n} {r.law} {r.param} "
              f"(F {r.med_fisher_full_khat:.5g}, H {r.med_hess_full_khat:.5g}, "
              f"F0 {r.med_fisher0_full_khat:.5g})")
        j = x.H_vs_F.abs().idxmax()
        print(f"     |H/F - 1| max {100 * abs(x.loc[j, 'H_vs_F']):.2f}% at d={x.loc[j, 'd']} "
              f"nu0={x.loc[j, 'nu0']} {x.loc[j, 'law']} {x.loc[j, 'param']}; "
              f"|F0/F - 1| max {100 * x.F0_vs_F.abs().max():.2f}%; "
              f"|cov F - cov H| max {x.cov_F_minus_H_pp.abs().max():.2f} pp")
    print("  per (d, nu0), all laws, max(med)/min(med) - 1 (%):")
    print("    " + (100 * big.groupby(["d", "nu0"]).max_over_min_minus_1.max()).round(2)
          .to_string().replace("\n", "\n    "))


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    S = pd.read_csv("results/summary/summary.csv")
    S = S[S.variant == "joint"]
    P = pd.read_csv(os.path.join(OUT, "q6c_perfit_small_n.csv"))
    item_b(S)
    item_c(S, P)
    item_d(S)


if __name__ == "__main__":
    main()

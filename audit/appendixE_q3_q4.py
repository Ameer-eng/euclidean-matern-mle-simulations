"""APPENDIX E QUESTIONS (2026-11-16), Q3 and Q4. Reads saved outputs only.

Q3 (initialization audit): every run in audit/out/ms_local/multistart_{A..E}.csv, which
is 152 data sets x 6 starts, written by audit/multistart.py on 2026-11-09. The script
reports:
  * KKT-qualifying runs, max_pg < 1e-3, where max_pg is the projected gradient in the
    start's own scaled optimizer coordinates;
  * the gap of every run below the best log-likelihood found for its data set
    (best = max over all 6 runs);
  * stopping messages, and the current saved (post-refit) fits against the audit best.

Q4 (joint coverage): exact numerators, denominators and ranges behind Table
tab:sim-joint and the "Log (fit)" / "Log (truth)" columns of Table tab:sim-joint-chain.
Source: results/summary/joint.csv (variant=joint, vector=log_vector).

  python -m audit.appendixE_q3_q4
Outputs: audit/out/appendixE/q3_*.csv, q4_*.csv, and stdout (q3_q4.txt).
"""
import glob
import os

import numpy as np
import pandas as pd

OUT = "audit/out/appendixE"
KEY = ["group", "d", "nu0", "G", "n", "law", "rep"]


def q3():
    ms = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("audit/out/ms_local/multistart_*.csv"))],
                   ignore_index=True)
    ms["qualify"] = ms.max_pg < 1e-3
    ms["best"] = ms.groupby(KEY).loglik.transform("max")
    ms["gap"] = ms.best - ms.loglik
    nds = ms.groupby(KEY).ngroups
    print(f"Q3: {len(ms)} runs on {nds} data sets; families: {ms.family.value_counts().to_dict()}")
    print(f"runs per data set: {sorted(ms.groupby(KEY).size().unique())}; groups: "
          f"{ms.drop_duplicates(KEY).group.value_counts().sort_index().to_dict()}")
    print(f"qualifying (max_pg < 1e-3): {int(ms.qualify.sum())} of {len(ms)}")
    by = ms.groupby("start").qualify.agg(["sum", "size"])
    print("by start:\n" + by.to_string())
    has = ms.groupby(KEY).qualify.any()
    print(f"data sets with >= 1 qualifying run: {int(has.sum())} of {nds}")
    print(f"max gap among qualifying runs: {ms.loc[ms.qualify, 'gap'].max():.3e}")
    qb = ms[ms.qualify].groupby(KEY).loglik.max()
    ab = ms.groupby(KEY).loglik.max()
    print(f"max (overall best - best qualifying) per data set: {(ab - qb.reindex(ab.index)).max():.3e}")
    nq = ms[~ms.qualify].sort_values(["group", "d", "nu0", "G", "law", "rep"])
    cols = KEY + ["start", "success", "message", "any_bound", "fit_in_Theta", "nfev", "max_pg", "gap"]
    print("\nnon-qualifying runs:\n" + nq[cols].to_string(index=False))
    nq[cols].to_csv(os.path.join(OUT, "q3_nonqualifying_runs.csv"), index=False)
    print("\nstopping messages, all runs:\n" + ms.message.str.strip().value_counts().to_string())
    print("stopping messages, non-qualifying runs:\n" + nq.message.str.strip().value_counts().to_string())
    print(f"success=False runs: {int((~ms.success).sum())}; fits outside Theta: {int((~ms.fit_in_Theta).sum())}")
    # data regeneration check
    dm = ms.groupby(KEY).agg(data_match=("data_match", "all"), ll_stored=("ll_stored", "first"),
                             ll_at_stored=("ll_at_stored", "first"), best=("best", "first"))
    dm["abs_diff_at_stored"] = (dm.ll_at_stored - dm.ll_stored).abs()
    print(f"\ndata sets failing the 1e-8 relative data check: {int((~dm.data_match).sum())}; "
          f"|ll(stored theta, regenerated data) - stored ll| > 1e-9: {int((dm.abs_diff_at_stored > 1e-9).sum())}, "
          f"max {dm.abs_diff_at_stored.max():.3e}")
    # current saved fits (post-refit) vs audit best
    cur = []
    for k in dm.index:
        g, d, nu0, G, n, law, rep = k
        f = f"results/fits/d{d}_nu{nu0:g}_G{G}/joint/{law}/rep{rep:04d}.npz"
        cur.append(float(np.load(f)["loglik"]))
    dm["ll_current"] = cur
    dm["current_minus_best"] = dm.ll_current - dm.best
    print(f"current saved fit - audit best: max {dm.current_minus_best.max():+.3e}, "
          f"min {dm.current_minus_best.min():+.3e}; below best by > 1e-6: "
          f"{int((dm.current_minus_best < -1e-6).sum())}; |diff| > 1e-6: "
          f"{int((dm.current_minus_best.abs() > 1e-6).sum())}")
    print(dm[dm.current_minus_best.abs() > 1e-6][["ll_stored", "ll_current", "best", "current_minus_best"]]
          .to_string())
    dm.reset_index().to_csv(os.path.join(OUT, "q3_datasets.csv"), index=False)
    ms[cols + ["family", "data_match"]].to_csv(os.path.join(OUT, "q3_all_runs.csv"), index=False)


def q4():
    J = pd.read_csv("results/summary/joint.csv")
    J = J[(J.variant == "joint") & (J.vector == "log_vector")]
    big = J[J.n == J.groupby("d").n.transform("max")]
    m = big[big.method == "fisher_full_khat"].copy()
    m["cell"] = (m.n_cover_valid.astype(str) + "/" + m.n_valid.astype(str) + " = "
                 + (100 * m.coverage).map("{:.2f}".format))
    print("\nQ4: tab:sim-joint entries (fisher_full_khat, log vector, largest n): covered/valid = %")
    print(m.pivot_table(index=["d", "nu0", "n"], columns="law", values="cell", aggfunc="first").to_string())
    print(f"all nfit = 400: {bool((m.nfit == 400).all())}; all n_valid = 400: {bool((m.n_valid == 400).all())}; "
          f"invalid of any kind: {int((m.nfit - m.n_valid).sum())}")
    rows = []
    for (d, nu0), g in m.groupby(["d", "nu0"]):
        rows.append(dict(row=f"d={d} nu0={nu0}", lo=100 * g.coverage.min(), hi=100 * g.coverage.max()))
    g = m[m.d == 3]
    rows.append(dict(row="d=3 both nu0 (table row 1)", lo=100 * g.coverage.min(), hi=100 * g.coverage.max()))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    m.to_csv(os.path.join(OUT, "q4_table_sim_joint.csv"), index=False)
    # Table E3 columns
    d5 = J[(J.d == 5) & (J.law == "gauss")]
    t = d5.pivot_table(index=["nu0", "n"], columns="method", values="coverage").mul(100)
    print("\ntab:sim-joint-chain, d=5 Gaussian: Log (fit) = fisher_full_khat, Log (truth) = fisher0_full_none")
    print(t[["fisher_full_khat", "fisher0_full_none", "fisher0_full_khat"]].round(2).to_string())
    # other quoted joint ranges
    fk = J[J.method == "fisher_full_khat"]
    small = fk[fk.n.isin([1000, 1296, 1024])]
    print(f"\njoint coverage at n ~ 1e3 (fisher_full_khat): {100 * small.coverage.min():.2f}-"
          f"{100 * small.coverage.max():.2f}; all valid: {bool((small.n_valid == small.nfit).all())}")
    J2 = pd.read_csv("results/summary/joint.csv")
    rp = J2[(J2.variant == "joint") & (J2.vector == "raw_public") & (J2.method == "fisher_full_khat")]
    rp = rp[rp.n == rp.groupby("d").n.transform("max")]
    print(f"raw public (d >= 4), largest n: {100 * rp.coverage.min():.2f}-{100 * rp.coverage.max():.2f}")
    un = big[(big.method == "fisher_full_none")]
    i = un.coverage.idxmin()
    print(f"uncorrected (fisher_full_none), largest n: min {100 * un.coverage.min():.2f} at "
          f"d={un.loc[i, 'd']} nu0={un.loc[i, 'nu0']} {un.loc[i, 'law']}; laws iii/iv min "
          f"{100 * un[un.law.isin(['mixnorm', 'bern'])].coverage.min():.2f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    q3()
    q4()

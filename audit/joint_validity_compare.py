"""FOLLOW-UP Task 2 (2026-11-10): before/after comparison of the joint-region validity fix.

Before: audit/out/followup/before_pdfix/joint.csv (old rule: e' solve(S, e) for every
finite S, so an indefinite S could count as covered).  After: results/summary/joint.csv
(Cholesky positive-definiteness rule of aggregate.joint_wald).  Also confirms that the
marginal summaries (summary.csv: Tables 3, D4 and every marginal number) are unchanged.

  python -m audit.joint_validity_compare
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

B = "audit/out/followup/before_pdfix"
OUT = "audit/out/followup"
KEY = ["d", "nu0", "G", "n", "law", "variant", "vector", "method"]


def main():
    pd.set_option("display.width", 250, "display.max_rows", 500)
    old = pd.read_csv(os.path.join(B, "joint.csv"))
    new = pd.read_csv("results/summary/joint.csv")
    m = old.merge(new, on=KEY, suffixes=("_old", "_new"), how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), m["_merge"].value_counts()
    m["delta_pp"] = 100 * (m.coverage_new - m.coverage_old)
    m["changed"] = ~np.isclose(m.coverage_old, m.coverage_new, rtol=0, atol=1e-12, equal_nan=True)
    cols = KEY + ["nfit_new", "n_valid", "n_invalid_not_pd", "n_invalid_nonfinite",
                  "coverage_old", "coverage_new", "prob_valid_and_cover", "delta_pp"]
    m[cols + ["changed", "max_rel_asym"]].to_csv(os.path.join(OUT, "task2_joint_before_after.csv"), index=False)
    ch = m[m.changed]
    print(f"{len(m)} joint rows; {int(m.changed.sum())} changed coverage; "
          f"{int((m.n_invalid_not_pd > 0).sum())} rows with non-PD covariances "
          f"({int(m.n_invalid_not_pd.sum())} region-replicates); "
          f"{int(m.n_invalid_asymmetric.sum())} asymmetric; max relative asymmetry "
          f"{np.nanmax(m.max_rel_asym):.2e}")
    print("\nchanged rows by method:")
    print(ch.groupby(["vector", "method"]).agg(rows=("delta_pp", "size"),
                                                nonPD=("n_invalid_not_pd", "sum"),
                                                max_abs_delta_pp=("delta_pp", lambda x: x.abs().max()))
          .to_string())
    print("\nnon-PD counts by method (all rows):")
    print(m.groupby(["vector", "method"]).n_invalid_not_pd.sum().loc[lambda s: s > 0].to_string())
    # rows that feed manuscript statements
    j = m[(m.variant == "joint")]
    lv = j[j.vector == "log_vector"]
    big = lv[lv.n == lv.groupby("d").n.transform("max")]
    print("\nTable 4 (fisher_full_khat, log vector, largest n): any change?",
          bool(big[big.method == "fisher_full_khat"].changed.any()))
    print(big[big.method.isin(["fisher_full_khat", "fisher0_full_khat", "hess_full_khat",
                               "hess0_full_khat", "fisher_full_none"])]
          .pivot_table(index=["d", "nu0", "law"], columns="method",
                       values=["coverage_old", "coverage_new"]).mul(100).round(2).to_string())
    small = lv[(lv.method == "fisher_full_khat") & lv.n.isin([1000, 1296, 1024])]
    print("\nD.4.1 joint coverage at n~1e3 (fisher_full_khat): old "
          f"{100*small.coverage_old.min():.2f}-{100*small.coverage_old.max():.2f}%, new "
          f"{100*small.coverage_new.min():.2f}-{100*small.coverage_new.max():.2f}%; "
          f"non-PD {int(small.n_invalid_not_pd.sum())}")
    rp = j[(j.vector == "raw_public") & (j.method == "fisher_full_khat")]
    rpb = rp[rp.n == rp.groupby("d").n.transform("max")]
    print(f"raw public joint (fisher_full_khat, largest n): old {100*rpb.coverage_old.min():.1f}-"
          f"{100*rpb.coverage_old.max():.1f}%, new {100*rpb.coverage_new.min():.1f}-{100*rpb.coverage_new.max():.1f}%")
    un = big[(big.method == "fisher_full_none") & big.law.isin(["mixnorm", "bern"])]
    print(f"uncorrected joint, laws iii/iv, largest n: old min {100*un.coverage_old.min():.2f}%, "
          f"new min {100*un.coverage_new.min():.2f}%")
    o = big[(big.d == 5) & (big.nu0 == 0.5) & (big.law == "gauss")].set_index("method")
    print(f"d5 nu0.5 n=1e5 gauss: fisher0 {100*o.loc['fisher0_full_khat','coverage_new']:.2f}%, "
          f"fisher {100*o.loc['fisher_full_khat','coverage_new']:.2f}%")
    # hess0 at the largest n: conditional vs valid-and-cover
    h0 = big[big.method == "hess0_full_khat"][["d", "nu0", "law", "n_valid", "n_invalid_not_pd",
                                              "coverage_old", "coverage_new", "prob_valid_and_cover"]]
    print("\nhess0_full_khat (oracle Hessian), largest n:")
    print(h0.round(4).to_string(index=False))
    # marginal summaries unchanged?
    so = pd.read_csv(os.path.join(B, "summary.csv"))
    sn = pd.read_csv("results/summary/summary.csv")
    assert list(so.columns) == list(sn.columns) and len(so) == len(sn)
    num = so.select_dtypes("number").columns
    diff = np.nanmax(np.abs(so[num].to_numpy(float) - sn[num].to_numpy(float)))
    same_nan = (so[num].isna().to_numpy() == sn[num].isna().to_numpy()).all()
    print(f"\nsummary.csv: max |difference| over numeric columns = {diff:.3g}; NaN pattern identical: {same_nan}")
    for f in ("nugget.csv", "theory.csv"):
        a, b = pd.read_csv(os.path.join(B, f)), pd.read_csv(os.path.join("results/summary", f))
        num = a.select_dtypes("number").columns
        print(f"{f}: max |difference| = {np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float))):.3g}")
    # smaller-n joint changes for the methods that appear in the paper
    paper = ch[ch.method.isin(["fisher_full_khat", "hess_full_khat", "fisher0_full_khat",
                               "hess0_full_khat", "fisher_full_none"])]
    print("\nchanged rows among methods quoted in the paper:")
    print(paper[["d", "nu0", "n", "law", "vector", "method", "n_invalid_not_pd", "coverage_old",
                 "coverage_new", "prob_valid_and_cover"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

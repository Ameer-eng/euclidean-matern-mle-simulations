"""APPENDIX E FINAL REQUESTS (2026-11-16), item 3: Table E5 (tab:sim-se) exported as
TeX from the saved integer counts.

Input  audit/out/appendixE/q6b_table_E5_counts.csv  (audit/appendixE_q6.py, part b;
       one row per (d, nu0, param, method) at the largest n, Gaussian errors), checked
       here against results/summary/summary.csv.
Columns (method names as in summary.csv; khat = fitted cumulant correction):
  Full  F, theta_hat   fisher_full_khat     Block  F, theta_hat   fisher_block_khat
        H, theta_hat   hess_full_khat              H, theta_hat   hess_block_khat
        F, theta_0     fisher0_full_khat           F, theta_0     fisher0_block_khat
        H, theta_0     hess0_full_khat             H, theta_0     hess0_block_khat
        Inv.           100 (total - valid) / total for hess0_full_khat
Coverage = 100 covered / valid (conditional on a finite, positive variance).
Rounding: each entry is an exact ratio of integers, rounded to one decimal with
round-half-to-even on the exact rational value (decimal.ROUND_HALF_EVEN), so the
result does not depend on floating-point representation.

  python -m audit.appendixE_table_e5_tex [--compare path/to/manuscript.tex]
Outputs: audit/out/appendixE/table_E5_body.tex, table_E5_cells.csv, stdout
(table_E5_tex.txt). With --compare, every cell is compared with the manuscript's
table body.
"""
import argparse
import os
import re
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction

import numpy as np
import pandas as pd

OUT = "audit/out/appendixE"
FULL = ["fisher_full_khat", "hess_full_khat", "fisher0_full_khat", "hess0_full_khat"]
BLOCK = ["fisher_block_khat", "hess_block_khat", "fisher0_block_khat", "hess0_block_khat"]
PARAMS = {3: ["log_nu", "log_c", "log_tau2"], 4: ["log_nu", "log_sigma2", "log_ell", "log_tau2"],
          5: ["log_nu", "log_sigma2", "log_ell", "log_tau2"]}
PTEX = {"log_nu": r"$\log\nu$", "log_c": r"$\log c$", "log_tau2": r"$\log\tau^2$",
        "log_sigma2": r"$\log\sigma^2$", "log_ell": r"$\log l$"}
NUTEX = {0.5: r"$\nu_0=1/2$", 1.5: r"$\nu_0=3/2$"}


def r1(num, den):
    """100 num / den rounded to one decimal, half-to-even on the exact rational."""
    q = Fraction(100 * int(num), int(den))
    return str((Decimal(q.numerator) / Decimal(q.denominator)).quantize(Decimal("0.1"), ROUND_HALF_EVEN)) \
        if den else ""


def exact(num, den):
    q = Fraction(100 * int(num), int(den))
    return f"{float(q):.4f}"


def build():
    C = pd.read_csv(os.path.join(OUT, "q6b_table_E5_counts.csv"))
    # cross-check with the production summary
    S = pd.read_csv("results/summary/summary.csv")
    S = S[(S.variant == "joint") & (S.law == "gauss")]
    S = S[S.n == S.groupby("d").n.transform("max")].set_index(["d", "nu0", "param"])
    for _, c in C.iterrows():
        s = S.loc[(c.d, c.nu0, c.param)]
        assert int(s[f"n_se_{c.method}"]) == c.valid and int(s.nfit) == c.total
        assert np.isclose(s[f"cov_{c.method}"] * c.valid, c.covered, atol=1e-9)
    C = C.set_index(["d", "nu0", "param", "method"])
    rows, cells = [], []
    for d in (3, 4, 5):
        if d > 3:
            rows.append(r"\midrule")
        for k, nu0 in enumerate((0.5, 1.5)):
            for j, p in enumerate(PARAMS[d]):
                lead = [f"$d={d}$" if (k == 0 and j == 0) else "", NUTEX[nu0] if j == 0 else "", PTEX[p]]
                vals = []
                for m in FULL:
                    c = C.loc[(d, nu0, p, m)]
                    vals.append(r1(c.covered, c.valid))
                    cells.append(dict(d=d, nu0=nu0, param=p, column=m, num=c.covered, den=c.valid,
                                      exact=exact(c.covered, c.valid), tex=vals[-1]))
                c = C.loc[(d, nu0, p, "hess0_full_khat")]
                vals.append(r1(c.total - c.valid, c.total))
                cells.append(dict(d=d, nu0=nu0, param=p, column="Inv.(hess0_full_khat)",
                                  num=c.total - c.valid, den=c.total, exact=exact(c.total - c.valid, c.total),
                                  tex=vals[-1]))
                for m in BLOCK:
                    if d == 3:
                        c = C.loc[(d, nu0, p, m)]
                        vals.append(r1(c.covered, c.valid))
                        cells.append(dict(d=d, nu0=nu0, param=p, column=m, num=c.covered, den=c.valid,
                                          exact=exact(c.covered, c.valid), tex=vals[-1]))
                    else:
                        vals.append("")
                body = " & ".join(lead + [f"${v}$" if v else "" for v in vals])
                rows.append(body + r"\\")
    return rows, pd.DataFrame(cells)


def manuscript_cells(path):
    """Numbers of the tab:sim-se body rows in the manuscript, in reading order."""
    tex = open(path, encoding="utf-8").read()
    i = tex.index(r"\label{tab:sim-se}")
    body = tex[tex.index(r"\midrule", i):tex.index(r"\bottomrule", i)]
    out = []
    for line in body.splitlines():
        if "&" not in line:
            continue
        f = [x.strip() for x in line.replace(r"\\", "").split("&")][3:]
        out.append([x.strip("$") for x in f])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", default=None)
    args = ap.parse_args(argv)
    rows, cells = build()
    head = [r"% Table E5 (tab:sim-se) body, generated by audit/appendixE_table_e5_tex.py from",
            r"% audit/out/appendixE/q6b_table_E5_counts.csv. Columns: fisher_full_khat,",
            r"% hess_full_khat, fisher0_full_khat, hess0_full_khat, Inv. (= invalid % of",
            r"% hess0_full_khat), fisher_block_khat, hess_block_khat, fisher0_block_khat,",
            r"% hess0_block_khat. Exact ratios of integers, one decimal, round half to even."]
    with open(os.path.join(OUT, "table_E5_body.tex"), "w") as f:
        f.write("\n".join(head + rows) + "\n")
    cells.to_csv(os.path.join(OUT, "table_E5_cells.csv"), index=False)
    print("\n".join(rows))
    ties = cells[[(Fraction(100 * int(n), int(d)) * 10).denominator == 2 for n, d in zip(cells.num, cells.den)]]
    print(f"\n{len(cells)} cells; {len(ties)} exact ties at the second decimal (x.x5), rounded half to even")
    if args.compare:
        ms = manuscript_cells(args.compare)
        ours = [[x.strip("$") for x in r.replace(r"\\", "").split(" & ")][3:] for r in rows if "&" in r]
        assert len(ms) == len(ours), (len(ms), len(ours))
        it = iter(cells.itertuples())
        diffs = 0
        names = FULL + ["Inv."] + BLOCK
        for i, (a, b) in enumerate(zip(ms, ours)):
            for j, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    diffs += 1
                    print(f"  row {i + 1} column {names[j]}: manuscript {x!r} -> exported {y!r}")
        print(f"cells differing from the manuscript: {diffs}")


if __name__ == "__main__":
    main()

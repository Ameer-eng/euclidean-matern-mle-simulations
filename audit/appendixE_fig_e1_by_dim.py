"""APPENDIX E (2026-11-16): array and text check for Figure E1.

Figure E1 is drawn by matern_sim/plots.py:
  fig_se      -> matern_mle_se.pdf          (combined 3 x 6 figure; row-3 label
                                             "invalid variance (%)" since 2026-11-16)
  fig_se_dim  -> matern_mle_se_d{3,4,5}.pdf (one 3 x 2 figure per dimension, 6.5 in wide)
Both take their statistics from plots.se_curves (khat-corrected SEs):
  coverage  arithmetic mean of cov_* over the (law x identifiable parameter) rows per n;
  ratio     geometric mean of madz_* over the same rows;
  invalid   100 (1 - sum n_se / sum nfit), pooled over the same rows.

This script
  1. draws fig_se (plt.close is intercepted so that its plotted arrays can be read back)
     and fig_se_dim for d = 3, 4, 5 into OUT;
  2. compares, curve by curve with np.array_equal on x and y, the arrays of
       (a) the combined figure, (b) the per-dimension figures, and
       (c) the arrays frozen from the combined figure before the relabel
           (audit/out/appendixE/fig_e1_arrays_before_relabel.npz, written 2026-11-16 by
           the unmodified fig_se, whose PDF text equals paper/figures/matern_mle_se.pdf);
  3. compares every non-empty text object of the combined figure with the frozen list
     (fig_e1_texts_before_relabel.json): the only expected change is the row-3 label.

  PYTHONPATH=repro/pylib python -m audit.appendixE_fig_e1_by_dim [--out DIR] [--results DIR]
Outputs: DIR/matern_mle_se.pdf, DIR/matern_mle_se_d{3,4,5}.pdf (+ .png previews),
         DIR/fig_e1_array_check.txt (DIR defaults to audit/out/appendixE; --results defaults
         to results, i.e. results/summary/*.csv; repro/regenerate.sh passes its OUTDIR).
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402
import numpy as np  # noqa: E402

from matern_sim import plots as P  # noqa: E402

FROZEN = "audit/out/appendixE"
ROWS = ("coverage", "madz_ratio", "invalid")


def combined(D, path):
    """Run fig_se and read back every data curve (in drawing order) and every text."""
    captured = {}
    real_close = P.plt.close
    P.plt.close = lambda fig=None: captured.setdefault("fig", fig)
    try:
        P.fig_se(D, path)
    finally:
        P.plt.close = real_close
    fig = captured["fig"]
    axes = np.array(fig.axes[:18]).reshape(3, 6)
    arr = {}
    for k, (d, nu0) in enumerate(P.SETTINGS):
        for row in range(3):
            lines = [ln for ln in axes[row, k].get_lines() if ln.get_marker() not in (None, "None", "")]
            arr[(d, nu0, row)] = [(np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float))
                                  for ln in lines]
    texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text()]
    plt.close(fig)
    return arr, texts


def frozen_arrays():
    z = np.load(os.path.join(FROZEN, "fig_e1_arrays_before_relabel.npz"))
    arr = {}
    for (d, nu0) in P.SETTINGS:
        for row in range(3):
            c, lst = 0, []
            while f"d{d}_nu{nu0}_r{row}_c{c}_x" in z:
                lst.append((z[f"d{d}_nu{nu0}_r{row}_c{c}_x"], z[f"d{d}_nu{nu0}_r{row}_c{c}_y"]))
                c += 1
            arr[(d, nu0, row)] = lst
    return arr


def same(a, b):
    return len(a) == len(b) and all(np.array_equal(xa, xb) and np.array_equal(ya, yb, equal_nan=True)
                                    for (xa, ya), (xb, yb) in zip(a, b))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=FROZEN)
    ap.add_argument("--results", default="results", help="directory holding summary/*.csv")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    D = P.Data(args.results)
    comb, texts = combined(D, os.path.join(args.out, "matern_mle_se.pdf"))
    dim = {}
    for d in (3, 4, 5):
        base = os.path.join(args.out, f"matern_mle_se_d{d}")
        dim.update(P.fig_se_dim(D, d, base + ".pdf", png=base + ".png"))
    froz = frozen_arrays()
    lines, ok = [], True
    for key in sorted(froz):
        s1, s2, s3 = same(froz[key], comb[key]), same(froz[key], dim[key]), same(comb[key], dim[key])
        ok &= s1 and s2 and s3
        npts = sum(len(x) for x, _ in froz[key])
        lines.append(f"d={key[0]} nu0={key[1]} row={ROWS[key[2]]:10s} curves={len(froz[key])} points={npts:3d} "
                     f"frozen==combined {s1}  frozen==per-dim {s2}  combined==per-dim {s3}")
    old_texts = json.load(open(os.path.join(FROZEN, "fig_e1_texts_before_relabel.json")))
    diff = [(a, b) for a, b in zip(old_texts, texts) if a != b]
    text_ok = len(old_texts) == len(texts) and diff == [("negative variance (%)", "invalid variance (%)")]
    lines.append(f"combined-figure texts: {len(texts)} (before relabel {len(old_texts)}); changed: {diff}")
    lines.append(f"ALL PLOTTED ARRAYS IDENTICAL: {ok}")
    lines.append(f"ONLY TEXT CHANGE IS THE ROW-3 LABEL: {text_ok}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out, "fig_e1_array_check.txt"), "w") as f:
        f.write(txt + "\n")
    assert ok and text_ok


if __name__ == "__main__":
    main()

"""APPENDIX E QUESTIONS (2026-11-16), Q1: block multiplicities on every grid.

For every Monte Carlo grid (config.GRIDS) and every theory grid (config.THEORY_GRIDS),
this builds symmetry.GridSymmetry(d, G). The constructor runs _check_completeness,
which raises unless sum_rho f_rho N_rho = N_class in each parity class and
sum_class n_patterns N_class = n. The script then records:
  * the number of distinct blocks, the largest block size N_rho, and
    sum_rho m_rho N_rho with m_rho = n_patterns(class) * f_rho (GridSymmetry.blocks),
    checked against n;
  * an orthogonality check of Q on two random vectors y1, y2: the round-trip error
    max|Q Q^T y - y| and the inner-product error |<Q^T y1, Q^T y2> - <y1, y2>|
    (sym.forward = Q^T, sym.inverse = Q).

  python -m audit.appendixE_q1
Output: audit/out/appendixE/q1_block_multiplicities.csv
"""
import os
import time

import numpy as np
import pandas as pd

from matern_sim import config
from matern_sim.symmetry import GridSymmetry


def main():
    rng = np.random.default_rng(0)
    rows = []
    grids = sorted({(d, G) for g in (config.GRIDS, config.THEORY_GRIDS) for d, Gs in g.items() for G in Gs})
    for d, G in grids:
        t0 = time.time()
        sym = GridSymmetry(d, G)
        info = sym.describe()
        b = sym.blocks
        y1, y2 = rng.standard_normal(sym.n), rng.standard_normal(sym.n)
        Z1, Z2 = sym.forward(y1), sym.forward(y2)
        rt = float(np.max(np.abs(sym.inverse(Z1) - y1)))
        ip = sum(float(np.sum(Z1[k] * Z2[k])) for k in Z1)
        rows.append(dict(d=d, G=G, n=sym.n, odd_G=G % 2 == 1,
                         monte_carlo=G in config.GRIDS[d], theory=G in config.THEORY_GRIDS[d],
                         n_classes=len(sym.classes), n_distinct_blocks=info["n_distinct_blocks"],
                         largest_block=info["largest_block"],
                         sum_mult_N=info["sum_mult_N"], equals_n=info["sum_mult_N"] == sym.n,
                         sum_mult=sum(x[3] for x in b),
                         roundtrip_err=rt, inner_product_err=abs(ip - float(y1 @ y2)) / sym.n,
                         seconds=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    os.makedirs("audit/out/appendixE", exist_ok=True)
    t.to_csv("audit/out/appendixE/q1_block_multiplicities.csv", index=False)
    print(f"\n{len(t)} grids; sum m N = n on all: {bool(t.equals_n.all())}; "
          f"max roundtrip {t.roundtrip_err.max():.1e}; max inner-product error/n {t.inner_product_err.max():.1e}")


if __name__ == "__main__":
    main()

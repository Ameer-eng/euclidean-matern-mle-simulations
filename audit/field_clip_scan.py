"""FOLLOW-UP Task 3 (2026-11-10): how often the generation clips an eigenvalue.

For every Monte Carlo grid (a directory results/fits/d*_nu*_G*), computes the
eigenvalues of each symmetry block of the signal covariance at theta0 exactly as
GridModel.signal_eigen does (numpy.linalg.eigh, then max(lam, 0)) and counts the
negative computed eigenvalues, i.e. the activations of the only modification.
Writes audit/out/followup/task3_clip_scan_<family>.csv.

  python -m audit.field_clip_scan [--skip-largest]
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

from matern_sim.kappa_task import cpu_family
from matern_sim.likelihood import GridModel
from matern_sim.run_task import truth

LARGEST = {(5, 10), (4, 18), (3, 46)}


def main():
    skip = "--skip-largest" in sys.argv
    rows = []
    for p in sorted(glob.glob("results/fits/d*_nu*_G*")):
        d, nu0, G = re.match(r".*/d(\d)_nu([\d.]+)_G(\d+)$", p).groups()
        d, nu0, G = int(d), float(nu0), int(G)
        if skip and (d, G) in LARGEST:
            continue
        t = time.time()
        m = GridModel(d, G)
        nneg = nneg_copies = nb = 0
        lmin, lmax, nmax = np.inf, 0.0, 0
        for ci, ri, mult, S in m.iter_blocks(truth(d, nu0), ("K",)):
            lam = np.linalg.eigh(S[0])[0]
            k = int(np.sum(lam < 0))
            nneg += k
            nneg_copies += k * mult
            nb += 1
            lmin, lmax, nmax = min(lmin, lam[0]), max(lmax, lam[-1]), max(nmax, len(lam))
        rows.append(dict(d=d, nu0=nu0, G=G, n=m.n, n_blocks=nb, largest_block=nmax,
                         lam_min=lmin, lam_max=lmax, n_neg_distinct=nneg,
                         n_neg_with_multiplicity=nneg_copies, seconds=time.time() - t))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    os.makedirs("audit/out/followup", exist_ok=True)
    df.to_csv(f"audit/out/followup/task3_clip_scan_{cpu_family()}.csv", index=False)
    print(f"{len(df)} grids; clipped eigenvalues: {int(df.n_neg_distinct.sum())} distinct, "
          f"{int(df.n_neg_with_multiplicity.sum())} with multiplicity; smallest computed "
          f"eigenvalue {df.lam_min.min():.3g}")


if __name__ == "__main__":
    main()

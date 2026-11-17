"""FOLLOW-UP Task 4 (2026-11-10): the data-based (DATA) restart of refit_task.

For every refitted fit (fit file with a 'refit' field) the replicate's data are
regenerated from its seeds (as in refit_task), and the DATA start is rebuilt exactly
as refit_task builds it:
    y2      = (1/n) sum_i Y_i^2           (= ||Q'Y||^2 / n, Q orthogonal)
    t_tilde = sum D^2 / (2 m)             (stored t_pair; D = disjoint adjacent-pair
                                           differences, fit.adjacent_pair_differences)
    sigma2  = max(y2 - t_tilde, 0.1 y2),  l = 0.25,  nu = 1,  tau2 = min(max(t_tilde, 1.1e-3), 9.9)
Reports whether the start lies in Theta and in the optimiser box (scipy's L-BFGS-B
would clip x0 componentwise to the box otherwise), and whether the regenerated data
reproduce the stored fit (data_rel <= 1e-8); y2 is reported in either case.

  python -m audit.data_start_check
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pandas as pd

from matern_sim.fit import Fitter, adjacent_pair_differences
from matern_sim.kappa_task import data_blocks, field_blocks
from matern_sim.kernel import theta_from_public
from matern_sim.likelihood import GridModel
from matern_sim.run_task import truth
from audit.multistart import in_box


def main():
    rows, cache = [], {}
    files = sorted(glob.glob("results/fits/*/joint/*/rep*.npz"))
    for f in files:
        with np.load(f) as z:
            if "refit" not in z.files:
                continue
            d, nu0, G, rep, law = int(z["d"]), float(z["nu0"]), int(z["G"]), int(z["rep"]), str(z["law"])
            t_pair, ll_pre = float(z["t_pair"]), float(z["loglik_prerefit"])
            th_pre = np.array(z["theta_hat_prerefit"])
            cands = json.loads(str(z["refit_candidates"]))
            best = str(z["refit_start"])
        key = (d, nu0, G)
        if key not in cache:
            cache.clear()
            m = GridModel(d, G)
            th0 = truth(d, nu0)
            I0 = np.load(f"results/theory/d{d}_nu{nu0:g}_G{G}.npz")["I0"]
            cache[key] = (m, m.signal_eigen(th0), Fitter(m, th0, nu0, I0))
        m, eig, F = cache[key]
        Z = data_blocks(m, field_blocks(eig, d, nu0, G, rep), d, nu0, G, rep, law)
        rel = abs(m.loglik_grad(th_pre, Z, grad=False) - ll_pre) / max(1.0, abs(ll_pre))
        y2 = sum(float(np.sum(v * v)) for v in Z.values()) / m.n
        s2 = max(y2 - t_pair, 0.1 * y2)
        tt = min(max(t_pair, 1.1e-3), 9.9)
        th = theta_from_public(d, s2, 0.25, 1.0, tt)
        x0 = F.x_of(th)
        rows.append(dict(d=d, nu0=nu0, G=G, n=m.n, law=law, rep=rep, data_rel=rel,
                         data_reproduced=rel <= 1e-8, y2=y2, t_pair=t_pair,
                         sigma2_start=s2, sigma2_branch="y2-t" if y2 - t_pair >= 0.1 * y2 else "0.1*y2",
                         tau2_clipped=tt != t_pair, eta_start=th[0],
                         start_in_Theta=in_box(th),
                         start_in_xbox=bool(np.all(x0 >= F.xlo) and np.all(x0 <= F.xhi)),
                         loglik_DATA=cands["DATA"]["loglik"], kept=best))
    df = pd.DataFrame(rows)
    os.makedirs("audit/out/followup", exist_ok=True)
    df.to_csv("audit/out/followup/task4_data_start_refits.csv", index=False)
    print(f"{len(df)} refitted fits; data reproduced: {int(df.data_reproduced.sum())}")
    print(f"start in Theta: {int(df.start_in_Theta.sum())}, in optimiser box: {int(df.start_in_xbox.sum())}")
    print(f"sigma2 branch 0.1*y2 used: {int((df.sigma2_branch == '0.1*y2').sum())}; "
          f"tau2 start clipped: {int(df.tau2_clipped.sum())}")
    print(df[["y2", "t_pair", "sigma2_start", "eta_start"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()

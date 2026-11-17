"""AUDIT (2026-11-09): sensitivity of the fitted global maximum to initialisation.

For each data set (regenerated from its seeds) the production fit (start theta0,
truth-based Fisher scaling) is repeated, then refitted from
  S1 (sigma2, l, nu, tau2) = (4, 0.1, 2.5, 0.02)
  S2                        = (0.25, 1.5, 0.3, 2)
  S3                        = (2, 0.8, 1, 0.05)      (the three starts of App. D.2.4)
  DATA   nu = 1, tau2 = t_pair (adjacent-pair nugget estimate), l = 0.25,
         sigma2 = max(var(y) - t_pair, 0.1 var(y))     (same truth-based scaling)
  DATAPC the DATA start with data-based scaling: Fisher information at the DATA
         start and L_n computed with nu = 1 (nothing from theta0 is used).
Records feasibility of each start in the exact box Theta and in the optimiser box,
the final log-likelihood, message, projected gradient and estimates.

  python -m audit.multistart --group A --out audit/out
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

from matern_sim import config
from matern_sim.fit import Fitter
from matern_sim.kappa_task import cpu_family, data_blocks, field_blocks
from matern_sim.kernel import theta_from_public
from matern_sim.likelihood import GridModel
from matern_sim.run_task import fit_path, log, truth

FAR = {"S1": (4.0, 0.1, 2.5, 0.02), "S2": (0.25, 1.5, 0.3, 2.0), "S3": (2.0, 0.8, 1.0, 0.05)}
LAWS3 = ["unif", "mixnorm", "bern"]


def cases(group):
    if group == "A":   # the 16 data sets described in App. D.2.4
        return [(d, nu, G, "gauss", r) for d, nu, G in
                [(3, .5, 10), (3, 1.5, 10), (4, .5, 6), (5, 1.5, 4)] for r in range(4)]
    if group == "B":   # same designs, non-Gaussian laws
        return [(d, nu, G, law, r) for d, nu, G in
                [(3, .5, 10), (3, 1.5, 10), (4, .5, 6), (5, 1.5, 4)]
                for law in LAWS3 for r in range(4)]
    if group == "C":   # boundary-prone smallest designs + the worst-KKT fits
        c = [(d, nu, G, law, r) for d, nu, G in
             [(3, .5, 5), (3, 1.5, 5), (4, .5, 3), (4, 1.5, 3), (5, .5, 3), (5, 1.5, 3)]
             for law in ("gauss", "bern") for r in range(4)]
        with open("audit/out/hi_pg_cases.json") as fh:
            c += [tuple(x) for x in json.load(fh)]
        return c
    if group == "D":   # n ~ 1e4 - 3e4
        return [(d, nu, G, law, r) for d, nu, G in
                [(3, .5, 22), (3, 1.5, 22), (4, .5, 13), (4, 1.5, 13), (5, .5, 8), (5, 1.5, 8)]
                for law in ("gauss", "mixnorm") for r in range(2)]
    if group == "E":   # n ~ 1e5 (d = 4, 5)
        return [(d, nu, G, "gauss", 0) for d, nu, G in
                [(4, .5, 18), (4, 1.5, 18), (5, .5, 10), (5, 1.5, 10)]]
    raise ValueError(group)


def in_box(theta):
    eta, nu, la, lt = theta
    B = config.BOX
    a_lo, a_hi = B["ell"][1] ** -2, B["ell"][0] ** -2
    return bool(B["eta"][0] <= eta <= B["eta"][1] and B["nu"][0] <= nu <= B["nu"][1]
                and a_lo <= math.exp(la) <= a_hi and B["tau2"][0] <= math.exp(lt) <= B["tau2"][1])


def proj_grad(F, x, g):
    gx = -(F.T.T @ g)
    lo, hi = x <= F.xlo + 1e-10, x >= F.xhi - 1e-10
    pg = gx.copy()
    pg[lo] = np.minimum(gx[lo], 0)
    pg[hi] = np.maximum(gx[hi], 0)
    return float(np.max(np.abs(pg)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", default="audit/out")
    a = ap.parse_args(argv)
    rows, models = [], {}
    for (d, nu0, G, law, rep) in cases(a.group):
        key = (d, nu0, G)
        if key not in models:
            m = GridModel(d, G)
            th0 = truth(d, nu0)
            I0 = np.load(f"results/theory/d{d}_nu{nu0:g}_G{G}.npz")["I0"]
            models[key] = (m, th0, I0, m.signal_eigen(th0), Fitter(m, th0, nu0, I0))
        m, th0, I0, eig, F = models[key]
        Z = data_blocks(m, field_blocks(eig, d, nu0, G, rep), d, nu0, G, rep, law)
        with np.load(fit_path("results", d, nu0, G, "joint", law, rep)) as z:
            th_st, ll_st, t_pair = np.array(z["theta_hat"]), float(z["loglik"]), float(z["t_pair"])
        ll_at_stored = m.loglik_grad(th_st, Z, grad=False)
        data_match = abs(ll_at_stored - ll_st) / max(1, abs(ll_st)) < 1e-8
        y2 = sum(float(np.sum(v * v)) for v in Z.values()) / m.n
        s2 = max(y2 - t_pair, 0.1 * y2)
        th_data = theta_from_public(d, s2, 0.25, 1.0, min(max(t_pair, 1.1e-3), 9.9))
        starts = {"truth": (F, np.zeros(4), th0)}
        for k, (s, l, nu, t) in FAR.items():
            th = theta_from_public(d, s, l, nu, t)
            starts[k] = (F, F.x_of(th), th)
        starts["DATA"] = (F, F.x_of(th_data), th_data)
        Fd = Fitter(m, th_data, 1.0, m.fisher(th_data))
        starts["DATAPC"] = (Fd, np.zeros(4), th_data)
        for name, (FF, x0, thst) in starts.items():
            t = time.time()
            x_in = bool(np.all(x0 >= FF.xlo - 1e-12) and np.all(x0 <= FF.xhi + 1e-12))
            res = FF.fit(Z, x_start=x0)
            th = res["theta"]
            rows.append(dict(group=a.group, d=d, nu0=nu0, G=G, n=m.n, law=law, rep=rep,
                             start=name, start_in_Theta=in_box(thst), start_in_xbox=x_in,
                             data_match=data_match, ll_stored=ll_st,
                             ll_at_stored=ll_at_stored, loglik=res["loglik"],
                             eta=th[0], nu=th[1], loga=th[2], logt=th[3],
                             fit_in_Theta=in_box(th), success=res["success"],
                             message=res["message"][:40], nfev=res["nfev"],
                             any_bound=bool(np.any(res["at_bound"])),
                             max_pg=proj_grad(FF, res["x"], res["grad"]),
                             seconds=time.time() - t, family=cpu_family()))
            log(f"{key} {law} rep {rep} {name:6s} ll={res['loglik']:.6f} nfev={res['nfev']} "
                f"{res['message'][:30]}")
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"multistart_{a.group}.csv"), index=False)


if __name__ == "__main__":
    main()

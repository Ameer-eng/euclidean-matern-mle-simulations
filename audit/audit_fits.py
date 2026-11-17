"""AUDIT (2026-11-09): per-fit table over all 67,200 saved fits -- original-constraint
checks, optimizer status, projected gradient (KKT residual) in the scaled optimizer
coordinates, min eigenvalues of the curvatures, kappa-tilde and hess0 data checks.
  python audit/audit_fits.py audit/out/fits_audit.pkl"""
import glob, math, os, sys, types
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from matern_sim import config
from matern_sim.fit import Fitter

OUT = sys.argv[1]
th = {}
for f in glob.glob("results/theory/*.npz"):
    z = np.load(f); th[(int(z["d"]), float(z["nu0"]), int(z["G"]))] = dict(z)
fitters = {}
rows = []
for f in sorted(glob.glob("results/fits/*/joint/*/rep*.npz")):
    z = np.load(f)
    d, nu0, G, n = int(z["d"]), float(z["nu0"]), int(z["G"]), int(z["n"])
    key = (d, nu0, G)
    if key not in fitters:
        t = th[key]
        fitters[key] = Fitter(types.SimpleNamespace(d=d, n=n), t["theta0"], nu0, t["I0"])
    F = fitters[key]
    th_hat = z["theta_hat"]; x = z["x_hat"]; g = z["grad"]
    gx = -(F.T.T @ g)                         # gradient of the objective (-ell) in x
    # projected gradient (L-BFGS-B convention)
    pg = gx.copy()
    lo_act = x <= F.xlo + 1e-10; hi_act = x >= F.xhi - 1e-10
    pg[lo_act] = np.minimum(gx[lo_act], 0.0)  # at lower bound only descent into interior matters
    pg[hi_act] = np.maximum(gx[hi_act], 0.0)
    k = np.load(f.replace("/fits/", "/kappa/"))
    h = np.load(f.replace("/fits/", "/hess0/"))
    negH, fis, negH0 = z["negH"], z["fisher_hat"], h["negH0"]
    eta, nu, la, lt = th_hat
    rows.append(dict(
        d=d, nu0=nu0, G=G, n=n, law=str(z["law"]), rep=int(z["rep"]),
        eta=eta, nu=nu, loga=la, logt=lt, u=eta - 2 * F.Ln * nu, Ln=F.Ln,
        x0=x[0], x1=x[1], x2=x[2], x3=x[3],
        xlo0=F.xlo[0], xhi0=F.xhi[0],
        u_lo=(x[0] <= F.xlo[0] + 1e-8), u_hi=(x[0] >= F.xhi[0] - 1e-8),
        x_lo=lo_act.tolist(), x_hi=hi_act.tolist(),
        loglik=float(z["loglik"]), success=bool(z["success"]), message=str(z["message"]),
        nit=int(z["nit"]), nfev=int(z["nfev"]),
        at_bound=z["at_bound"].tolist(),
        max_abs_gx=float(np.max(np.abs(gx))), max_abs_pg=float(np.max(np.abs(pg))),
        g_eta=g[0], g_nu=g[1], g_la=g[2], g_lt=g[3],
        minev_negH=float(np.linalg.eigvalsh(negH).min() / np.linalg.eigvalsh(negH).max()),
        minev_fisher=float(np.linalg.eigvalsh(fis).min() / np.linalg.eigvalsh(fis).max()),
        minev_negH0=float(np.linalg.eigvalsh(negH0).min() / np.linalg.eigvalsh(negH0).max()),
        kappa=float(k["kappa_resid"]), k_numer=float(k["numer"]), k_denom=float(k["denom"]),
        k_ll_rel=float(k["loglik_rel_diff"]), k_diag=float(k["diag_check"]),
        k_family=str(k["family"]) if "family" in k.files else "", k_threads=str(k["threads"]) if "threads" in k.files else "",
        h0_rel=float(h["oracle_rel_diff"]), h0_family=str(h["family"]) if "family" in h.files else "",
        kappa_true=float(z["kappa4_true"]), sec_fit=float(z["seconds_fit"]),
        sec_total=float(z["seconds_total"])))
df = pd.DataFrame(rows)
df.to_pickle(OUT)
print(len(df), "fits")

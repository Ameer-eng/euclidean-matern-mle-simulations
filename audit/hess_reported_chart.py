"""AUDIT (2026-11-09), Q5/Q6: observed-Hessian standard errors in the reported log
chart, with the score term, vs the production transformation (no score term).

  production ("impl"):  V_zeta = J V_theta J^T,  V_theta = H_theta^{-1}   (aggregate.feasible_covs)
                        = (G^T H_theta G)^{-1},  G = DF, F: zeta -> theta
  reported chart ("rep"): V_zeta = (G^T H_theta G - sum_j s_j D^2 F_j)^{-1},  s = grad_theta ell
Both: + kappa_tilde / (n t^2) on log tau2 (t = t_hat at theta_hat, t0 at theta0), as in
production.  "full": identifiable entries of the 4x4 inverse;  "block" (d=3): inverse of
the identifiable 3x3 block (log c, log nu, log tau2).  Valid = finite and > 0.

At theta_hat: H_theta = stored negH, s = stored grad (all 67,200 fits).
At theta0:    H_theta = results/hess0 negH0, s = score at theta0 from audit.score0_task
              (audit/out/score0_local/ok or audit/out/score0/ok; data regeneration checked).
Output: audit/out/hess_reported_chart.csv, one row per (d, nu0, n, law, at, mode, param).
"""
import glob, math, os, sys
import numpy as np, pandas as pd
from audit.hessian_chart_check import F_and_derivs, zeta_of
from matern_sim import aggregate as A
from matern_sim.config import GRIDS

LAWS = ["gauss", "unif", "mixnorm", "bern"]
# names of zeta coordinates in hessian_chart_check order
ZNAMES = {3: ["log_c", "log_nu", "log_ell", "log_tau2"], 4: ["log_sigma2", "log_ell", "log_nu", "log_tau2"]}


def znames(d):
    return ZNAMES[3] if d <= 3 else ZNAMES[4]


def score0_file(f):
    for root in ("audit/out/score0_local/ok", "audit/out/score0/ok"):
        p = f.replace("results/fits", root)
        if os.path.exists(p):
            return p
    return None


def variances(H, s, zeta, d, n, t, kap):
    """dict (chart, mode) -> variances of the zeta coordinates (NaN where unavailable)."""
    _, G, D2 = F_and_derivs(zeta, d)
    Himpl = G.T @ H @ G
    out = {}
    for chart in ("impl", "rep"):
        if chart == "rep":
            if s is None:
                continue
            Hz = Himpl - sum(s[j] * D2[j] for j in range(4))
        else:
            Hz = Himpl
        names = znames(d)
        it = names.index("log_tau2")
        modes = {"full": list(range(4))}
        if d <= 3:
            modes["block"] = [names.index(p) for p in ("log_c", "log_nu", "log_tau2")]
        for mode, idx in modes.items():
            v = np.full(4, np.nan)
            if np.all(np.isfinite(Hz)):
                try:
                    Vi = np.linalg.inv(Hz[np.ix_(idx, idx)])
                    v[idx] = np.diag(Vi)
                except np.linalg.LinAlgError:
                    pass
            v[it] += kap / (n * t * t) if np.isfinite(kap) else np.nan
            out[(chart, mode)] = v
    return out


def run_design(d, nu0, G, rows, at0=True):
    tagg = f"d{d}_nu{nu0:g}_G{G}"
    names = znames(d)
    ident = ["log_c", "log_nu", "log_tau2"] if d <= 3 else ["log_nu", "log_sigma2", "log_ell", "log_tau2"]
    for law in LAWS:
        files = sorted(glob.glob(f"results/fits/{tagg}/joint/{law}/rep*.npz"))
        if not files:
            continue
        recs = []
        n_s0 = 0
        for f in files:
            z = np.load(f)
            n = int(z["n"]); th0 = z["theta0"]; thh = z["theta_hat"]
            kap = A.load_kappa(f)
            zeta0, zetah = zeta_of(th0, d), zeta_of(thh, d)
            err = zetah - zeta0
            r = {"err": err}
            r["hat"] = variances(z["negH"], z["grad"], zetah, d, n, math.exp(thh[3]), kap)
            if at0:
                sf = score0_file(f)
                s0 = None
                if sf is not None:
                    with np.load(sf) as q:
                        s0 = np.array(q["score0"], float)
                    n_s0 += 1
                r["0"] = variances(A.load_hess0(f), s0, zeta0, d, n, math.exp(th0[3]), kap)
            recs.append(r)
        errs = np.stack([r["err"] for r in recs])
        for at in (["hat", "0"] if at0 else ["hat"]):
            for key in recs[0][at]:
                chart, mode = key
                V = np.stack([r[at][key] if key in r[at] else np.full(4, np.nan) for r in recs])
                for p in ident:
                    j = names.index(p)
                    v = V[:, j]
                    valid = np.isfinite(v) & (v > 0)
                    if chart == "rep" and at == "0":
                        have = np.array([key in r["0"] for r in recs])
                    else:
                        have = np.ones(len(recs), bool)
                    se = np.sqrt(np.where(valid, v, np.nan))
                    cov = np.abs(errs[:, j]) <= 1.96 * se
                    rows.append(dict(d=d, nu0=nu0, G=G, n=n, law=law, at=at, chart=chart, mode=mode,
                                     param=p, n_avail=int(have.sum()), n_valid=int((valid & have).sum()),
                                     invalid_pct=100 * (1 - (valid & have).sum() / max(have.sum(), 1)),
                                     cov_cond=100 * np.nanmean(np.where(valid & have, cov, np.nan)) if (valid & have).any() else np.nan,
                                     cov_uncond=100 * np.sum(valid & have & cov) / max(have.sum(), 1),
                                     med_se=float(np.nanmedian(np.where(have, se, np.nan))) if (valid & have).any() else np.nan,
                                     se_cv=float(np.nanstd(se[valid & have]) / np.nanmean(se[valid & have]))
                                     if (valid & have).sum() > 1 else np.nan))
        print(f"{tagg} {law}: {len(files)} fits, score0 available for {n_s0}", flush=True)


def main():
    rows = []
    designs = [(d, nu0, G) for d in (3, 4, 5) for nu0 in (0.5, 1.5) for G in GRIDS[d]]
    if len(sys.argv) > 1 and sys.argv[1] == "largest":
        designs = [(d, nu0, GRIDS[d][-1]) for d in (3, 4, 5) for nu0 in (0.5, 1.5)]
    for d, nu0, G in designs:
        run_design(d, nu0, G, rows)
    df = pd.DataFrame(rows)
    out = "audit/out/hess_reported_chart" + ("_largest" if len(sys.argv) > 1 else "") + ".csv"
    df.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()

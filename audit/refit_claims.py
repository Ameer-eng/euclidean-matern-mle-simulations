"""AUDIT (2026-11-10): manuscript claims that involve n <= 10^4, before and after the
KKT-restart refits (matern_sim.refit_task).  Reads results/summary_prerefit and
results/summary.

  python -m audit.refit_claims > audit/out/refit_claims.txt
"""
import glob, math
import numpy as np, pandas as pd
from scipy.stats import spearmanr, skew

TAU_LO = math.log(1e-3)
NU_HI = math.log(4.0)


def load(tag):
    s = pd.read_csv(f"results/{tag}/summary.csv")
    r = pd.read_csv(f"results/{tag}/replicates.csv")
    j = pd.read_csv(f"results/{tag}/joint.csv")
    return s[s.variant == "joint"], r[r.variant == "joint"], j[j.variant == "joint"]


def claims(tag):
    s, r, j = load(tag)
    out = {}
    # --- Table D2 and its caption / Section 4.1 text
    b = r.groupby(["d", "nu0", "n", "law"]).any_bound.mean().mul(100).unstack("law")
    out["TableD2_gauss"] = b["gauss"]
    diff = b[["unif", "mixnorm", "bern"]].sub(b["gauss"], axis=0).abs()
    out["D2 caption: max |law - Gaussian| (pp)"] = (round(diff.max().max(), 2),
                                                    diff.stack().idxmax())
    nb = b.reset_index()
    out["max % on bound, n >= 1e4 (any law)"] = round(nb[nb.n >= 1e4][["gauss", "unif", "mixnorm", "bern"]].max().max(), 2)
    out["max % on bound, n >= 2.5e4 (any law)"] = round(nb[nb.n >= 2.5e4][["gauss", "unif", "mixnorm", "bern"]].max().max(), 2)
    # --- D.4.4: tau2 at the lower bound, d = 3, nu0 = 1/2
    g = r[(r.d == 3) & (r.nu0 == 0.5)]
    lo = g.log_tau2_hat <= TAU_LO + 1e-6
    pct = g.assign(lo=lo).groupby(["n", "law"]).lo.mean().mul(100).unstack("law")
    out["D.4.4 % tau2 at lower bound, d=3 nu0=1/2"] = pct.round(2)
    g1 = g[(g.n == 1000) & (g.law == "gauss")]
    lo1 = g1.log_tau2_hat <= TAU_LO + 1e-6
    nu = np.exp(g1.log_nu_hat)
    out["D.4.4 Gaussian n=1000: count at bound, nu range, median nu at bound / others"] = (
        int(lo1.sum()), (round(nu[lo1].min(), 2), round(nu[lo1].max(), 2)),
        round(nu[lo1].median(), 2), round(nu[~lo1].median(), 2))
    out["D.4.4 other laws, % at lower bound at n=1000"] = (
        round(pct.loc[1000, ["unif", "mixnorm", "bern"]].min(), 2),
        round(pct.loc[1000, ["unif", "mixnorm", "bern"]].max(), 2))
    sk = {}
    for (n, law), h in g.groupby(["n", "law"]):
        z = h.log_tau2_err / h.log_tau2_err.std()
        sk[(n, law)] = round(float(skew(z)), 2)
    out["D.4.4 skewness of nugget errors, d=3 nu0=1/2, Gaussian, n>=1000"] = {
        n: sk[(n, "gauss")] for n in sorted({k[0] for k in sk}) if n >= 1000}
    # --- D.5.1: invalid H(theta_hat) variances
    hv = s.assign(inv=100 * (1 - s["n_se_hess_full_khat"] / s["nfit"]))
    for grp, sel in (("d=3", hv.d == 3), ("d>=4", hv.d >= 4)):
        w = hv[sel]
        k = w.inv.idxmax()
        out[f"D.5.1 max % invalid H(theta_hat) var, {grp} (all laws)"] = (
            round(w.inv.max(), 2), tuple(w.loc[k, ["d", "nu0", "n", "law", "param"]]))
        wg = w[w.law == "gauss"]
        out[f"D.5.1 max % invalid H(theta_hat) var, {grp} (Gaussian)"] = round(wg.inv.max(), 2)
    out["D.5.1 n with any invalid H(theta_hat) var, by d"] = {
        d: sorted(hv[(hv.d == d) & (hv.inv > 0)].n.unique().tolist()) for d in (3, 4, 5)}
    # --- D.5.1: nu interval, d=5 nu0=1/2 n=1024 Gaussian
    w = s[(s.d == 5) & (s.nu0 == 0.5) & (s.n == 1024) & (s.law == "gauss") & (s.param == "log_nu")]
    out["D.5.1 d=5 nu0=1/2 n=1024 log nu coverage, Fisher / Hessian"] = (
        round(100 * float(w.cov_fisher_full_khat.iloc[0]), 2), round(100 * float(w.cov_hess_full_khat.iloc[0]), 2))
    # --- D.5.1: rank correlations, d=3, Gaussian, n <= 2744
    rc = []
    for (nu0, n), h in r[(r.d == 3) & (r.law == "gauss") & (r.n <= 2744)].groupby(["nu0", "n"]):
        for p in ("log_nu", "log_c"):
            se, e = h[f"{p}_se_fisher_full_khat"], h[f"{p}_err"].abs()
            ok = np.isfinite(se)
            rc.append(spearmanr(se[ok], e[ok]).statistic)
    out["D.5.1 rank corr (F theta_hat SE vs |err|), d=3 Gaussian n<=2744"] = (round(min(rc), 2), round(max(rc), 2))
    # --- D.4.1: joint coverage at n ~ 1e3
    jj = j[(j.method == "fisher_full_khat") & (j.vector == "log_vector") & j.n.isin([1000, 1296, 1024])]
    out["D.4.1 joint coverage at n~1e3 (fisher_full_khat), min-max %"] = (
        round(100 * jj.coverage.min(), 2), round(100 * jj.coverage.max(), 2))
    # --- D.2.3: kappa_tilde for n >= 1e4
    kk = r[r.n >= 1e4].groupby(["d", "nu0", "n", "law"]).kappa_hat.mean().reset_index()
    true = {"gauss": 0.0, "unif": -0.075, "mixnorm": 1 / 3, "bern": 0.31944}
    kk["rel"] = [abs(m - true[l]) / abs(true[l]) if true[l] else np.nan for m, l in zip(kk.kappa_hat, kk.law)]
    out["D.2.3 kappa_tilde n>=1e4: max rel dev (non-Gauss), max |mean| (Gauss)"] = (
        round(kk.rel.max(), 4), round(kk[kk.law == "gauss"].kappa_hat.abs().max(), 5))
    # --- D.2.3: evaluation counts (production nfev)
    big = r[r.n >= 1e4].groupby(["d", "nu0", "n"]).nfev
    out["D.2.3 nfev n>=1e4: mean range, median range"] = (
        (round(big.mean().min(), 1), round(big.mean().max(), 1)),
        (big.median().min(), big.median().max()))
    sm = r[r.groupby(["d", "nu0"]).n.transform("min") == r.n].groupby(["d", "nu0"]).nfev.mean()
    out["D.2.3 nfev smallest n: mean range (production)"] = (round(sm.min(), 1), round(sm.max(), 1))
    return out


def main():
    pd.set_option("display.width", 200)
    a, b = claims("summary_prerefit"), claims("summary")
    for k in a:
        if k == "TableD2_gauss":
            t = pd.DataFrame({"before": a[k], "after": b[k]})
            t = t[t.before != t.after]
            print("Table D2 (Gaussian) cells that change:\n" + t.round(2).to_string() + "\n")
            continue
        same = (str(a[k]) == str(b[k]))
        print(f"{k}:\n   before {a[k]}\n   after  {b[k]}{'   (unchanged)' if same else ''}\n")
    # evaluation counts including the restarts
    rf = pd.read_csv("audit/out/refit_per_fit.csv")
    print(f"restart evaluations: {len(rf)} fits, total {rf.refit_nfev.sum()}, "
          f"per fit median {rf.refit_nfev.median():.0f} (range {rf.refit_nfev.min()}-{rf.refit_nfev.max()})")


if __name__ == "__main__":
    main()

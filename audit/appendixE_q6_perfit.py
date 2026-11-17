"""APPENDIX E QUESTIONS (2026-11-16), Q6(c): per-fit cross-tabulation of three distinct
failure criteria for the curvature at the fit, on the grids where any of them occurs
(n <= 343; joint.csv and summary.csv record no failure of the fitted-curvature
methods at larger n).

For each fit, and for each curvature A in {Fisher at theta_hat, Hessian at theta_hat}:
  (i)   indefinite uncorrected full curvature: min eigenvalue of the 4x4 A in
        w = (eta, nu, log a, log t) is <= 0;
  (ii)  invalid marginal variance: some identifiable reported coordinate has a
        nonfinite or nonpositive variance J V J^T. This is checked uncorrected
        ("none") and with the fitted cumulant correction ("khat");
  (iii) invalid corrected joint covariance: aggregate.joint_wald status != 'ok' for the
        log identifiable vector with the khat correction (nonfinite, asymmetric
        beyond 1e-8 relative, or not positive definite by Cholesky).
The covariances are those of aggregate.feasible_covs, and the per-cell counts are
checked against results/summary/{summary,joint}.csv.

  python -m audit.appendixE_q6_perfit
Output: audit/out/appendixE/q6c_perfit_small_n.csv and stdout
(audit/out/appendixE/q6c_perfit.txt).
"""
import glob
import os

import numpy as np
import pandas as pd

from matern_sim import aggregate as A

OUT = "audit/out/appendixE"
SMALL = [(3, 5), (3, 7), (4, 3), (4, 4), (5, 3)]       # n = 125, 343, 81, 256, 243


def load(d, G):
    rows = []
    for f in sorted(glob.glob(f"results/fits/d{d}_nu*_G{G}/joint/*/rep*.npz")):
        z = np.load(f)
        rows.append(dict(kappa_resid=A.load_kappa(f), negH0=A.load_hess0(f),
                         d=int(z["d"]), nu0=float(z["nu0"]), G=int(z["G"]), n=int(z["n"]),
                         law=str(z["law"]), variant=str(z["variant"]), rep=int(z["rep"]),
                         theta_hat=z["theta_hat"], theta0=z["theta0"], negH=z["negH"],
                         fisher_hat=z["fisher_hat"], kappa_pair=float(z["kappa_pair"]),
                         kappa4_true=float(z["kappa4_true"])))
    return rows


def main():
    np.seterr(invalid="ignore", divide="ignore")
    th_all = A.load_theory("results")
    recs = []
    for d, G in SMALL:
        fits = load(d, G)
        groups = {}
        for r in fits:
            groups.setdefault((r["d"], r["nu0"], r["G"], r["law"]), []).append(r)
        for key, grp in sorted(groups.items()):
            grp = sorted(grp, key=lambda r: r["rep"])
            dd, nu0, GG, law = key
            th = th_all.get((dd, nu0, GG))
            covs = A.feasible_covs(grp, dd, "joint", th)
            theta_hat = np.stack([r["theta_hat"] for r in grp])
            g0 = A.gvals(grp[0]["theta0"][None, :], dd)[0]
            err = A.gvals(theta_hat, dd) - g0
            J = A.gjac(theta_hat, dd)
            params = A.params_for(dd, "joint")
            idx = [A.PARAMS.index(p) for p in params]
            for i, r in enumerate(grp):
                rec = dict(d=dd, nu0=nu0, G=GG, n=r["n"], law=law, rep=r["rep"])
                for src, mat in (("fisher", r["fisher_hat"]), ("hess", r["negH"])):
                    ev = np.linalg.eigvalsh(mat)
                    rec[f"{src}_indefinite"] = bool(ev.min() <= 0)
                    for k in ("none", "khat"):
                        V = covs[f"{src}_full_{k}"][i]
                        S = J[i][idx] @ V @ J[i][idx].T
                        var = np.diag(S)
                        rec[f"{src}_marg_invalid_{k}"] = bool(np.any(~(np.isfinite(var) & (var > 0))))
                        if k == "khat":
                            st, _, _ = A.joint_wald(err[i, idx], S)
                            rec[f"{src}_joint_status_khat"] = st
                recs.append(rec)
        print(f"d={d} G={G}: {len(fits)} fits", flush=True)
    t = pd.DataFrame(recs)
    t.to_csv(os.path.join(OUT, "q6c_perfit_small_n.csv"), index=False)

    # consistency with the production summaries
    J_ = pd.read_csv("results/summary/joint.csv")
    J_ = J_[(J_.variant == "joint") & (J_.vector == "log_vector")]
    for src in ("fisher", "hess"):
        c = t.groupby(["d", "nu0", "n", "law"])[f"{src}_joint_status_khat"].apply(
            lambda s: int((s != "ok").sum()))
        ref = J_[J_.method == f"{src}_full_khat"].set_index(["d", "nu0", "n", "law"]).pipe(
            lambda x: x.nfit - x.n_valid).reindex(c.index)
        print(f"{src}: per-cell invalid joint counts match joint.csv: {bool((c == ref).all())}")
    for src in ("fisher", "hess"):
        x = t
        ind, mn, mk = x[f"{src}_indefinite"], x[f"{src}_marg_invalid_none"], x[f"{src}_marg_invalid_khat"]
        jbad = x[f"{src}_joint_status_khat"] != "ok"
        print(f"\n{src.upper()} at theta_hat, {len(x)} fits with n <= 343:")
        print(f"  (i)   indefinite uncorrected full matrix         : {int(ind.sum())}")
        print(f"  (ii)  invalid uncorrected marginal (any param)  : {int(mn.sum())}"
              f"  [of which indefinite: {int((mn & ind).sum())}]")
        print(f"        invalid corrected marginal (any param)    : {int(mk.sum())}"
              f"  [uncorrected also invalid: {int((mk & mn).sum())}; caused by the correction: {int((mk & ~mn).sum())}]")
        print(f"  (iii) invalid corrected joint covariance        : {int(jbad.sum())}"
              f"  by status {x.loc[jbad, f'{src}_joint_status_khat'].value_counts().to_dict()}")
        print(f"        ... with every corrected marginal valid   : {int((jbad & ~mk).sum())}")
        print(f"        ... with PD uncorrected full matrix       : {int((jbad & ~ind).sum())}")
        print(f"        corrected marginal invalid but joint ok   : {int((mk & ~jbad).sum())}")
        print(pd.crosstab([ind.rename("indefinite"), mk.rename("marg_invalid_khat")],
                          jbad.rename("joint_invalid_khat")).to_string())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()

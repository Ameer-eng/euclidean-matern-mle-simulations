"""AUDIT: recompute joint chi-square coverage at the largest n from the fit files,
check positive definiteness of every covariance used, and compare with joint.csv."""
import glob, math, os
import numpy as np, pandas as pd
from scipy.stats import chi2
from matern_sim import aggregate as A

jc = pd.read_csv("results/summary/joint.csv")
th = A.load_theory("results")
out = []
for tagg in ["d3_nu0.5_G46", "d3_nu1.5_G46", "d4_nu0.5_G18", "d4_nu1.5_G18", "d5_nu0.5_G10", "d5_nu1.5_G10"]:
    for law in ["gauss", "unif", "mixnorm", "bern"]:
        files = sorted(glob.glob(f"results/fits/{tagg}/joint/{law}/rep*.npz"))
        grp = []
        for f in files:
            z = np.load(f)
            grp.append(dict(kappa_resid=A.load_kappa(f), negH0=A.load_hess0(f), d=int(z["d"]),
                            nu0=float(z["nu0"]), G=int(z["G"]), n=int(z["n"]), law=law,
                            variant="joint", rep=int(z["rep"]), theta_hat=z["theta_hat"],
                            theta0=z["theta0"], negH=z["negH"], fisher_hat=z["fisher_hat"],
                            kappa_pair=float(z["kappa_pair"]), kappa4_true=float(z["kappa4_true"])))
        d, nu0, G, n = grp[0]["d"], grp[0]["nu0"], grp[0]["G"], grp[0]["n"]
        covs = A.feasible_covs(grp, d, "joint", th[(d, nu0, G)])
        ps = A.params_for(d, "joint"); idx = [A.PARAMS.index(p) for p in ps]
        theta_hat = np.stack([r["theta_hat"] for r in grp]); theta0 = grp[0]["theta0"]
        g_hat = A.gvals(theta_hat, d); g0 = A.gvals(theta0[None], d)[0]
        J = A.gjac(theta_hat, d); J0 = np.repeat(A.gjac(theta0[None], d), len(grp), 0)
        for vec in (["log_vector", "raw_public"] if d >= 4 else ["log_vector"]):
            for meth in ["fisher_full_khat", "fisher0_full_khat", "hess_full_khat", "hess0_full_khat", "fisher_full_none"]:
                V = covs[meth]; at0 = meth.startswith(("fisher0", "hess0"))
                Jv = (J0 if at0 else J)[:, idx, :]
                ev = (g_hat - g0)[:, idx]
                if vec == "raw_public":
                    ev = np.exp(g_hat[:, idx]) - np.exp(g0[idx])
                    sc = np.exp(np.repeat(g0[None, idx], len(ev), 0) if at0 else g_hat[:, idx])
                    Jv = Jv * sc[:, :, None]
                S = np.einsum("mpi,mij,mqj->mpq", np.where(np.isnan(V), 0, V), Jv, Jv) if False else \
                    np.einsum("mpi,mij,mqj->mpq", Jv, np.where(np.isnan(V), 0, V), Jv)
                valid_diag = ~np.any(np.isnan(np.diagonal(V, axis1=1, axis2=2)), axis=1)
                minev = np.array([np.linalg.eigvalsh(s).min() for s in S])
                stat = np.array([e @ np.linalg.solve(s, e) for e, s in zip(ev, S)])
                crit = chi2.ppf(0.95, len(idx))
                used = valid_diag                      # production rule
                cov_prod = np.mean(stat[used] <= crit)
                pd_ = valid_diag & (minev > 0)
                cov_pd = np.mean(stat[pd_] <= crit)
                cov_uncond = np.mean(pd_ & (stat <= crit))   # invalid counted as non-coverage
                ref = jc[(jc.d == d) & (jc.nu0 == nu0) & (jc.n == n) & (jc.law == law) &
                         (jc.vector == vec) & (jc.method == meth)].coverage
                out.append(dict(d=d, nu0=nu0, n=n, law=law, vector=vec, method=meth,
                                cov_csv=float(ref.iloc[0]) if len(ref) else np.nan,
                                cov_recomputed=cov_prod, n_used=int(used.sum()),
                                n_nonPD_used=int((used & (minev <= 0)).sum()),
                                n_nonPD_counted_covered=int((used & (minev <= 0) & (stat <= crit)).sum()),
                                cov_PD_only=cov_pd, cov_invalid_as_miss=cov_uncond))
df = pd.DataFrame(out)
df.to_csv("audit/out/joint_check.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(4).to_string(index=False))

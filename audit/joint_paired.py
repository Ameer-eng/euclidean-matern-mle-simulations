"""AUDIT: paired MC SE of the oracle-vs-fitted joint coverage contrast (d=5, nu0=1/2, n=1e5)."""
import glob, numpy as np
from scipy.stats import chi2
from matern_sim import aggregate as A
th = A.load_theory("results")
for law in ["gauss", "unif", "mixnorm", "bern"]:
    grp = []
    for f in sorted(glob.glob(f"results/fits/d5_nu0.5_G10/joint/{law}/rep*.npz")):
        z = np.load(f)
        grp.append(dict(kappa_resid=A.load_kappa(f), negH0=A.load_hess0(f), n=int(z["n"]), theta_hat=z["theta_hat"],
                        theta0=z["theta0"], negH=z["negH"], fisher_hat=z["fisher_hat"], kappa_pair=float(z["kappa_pair"]),
                        kappa4_true=float(z["kappa4_true"])))
    covs = A.feasible_covs(grp, 5, "joint", th[(5, 0.5, 10)])
    idx = [A.PARAMS.index(p) for p in A.params_for(5, "joint")]
    th_hat = np.stack([r["theta_hat"] for r in grp]); th0 = grp[0]["theta0"]
    e = (A.gvals(th_hat, 5) - A.gvals(th0[None], 5))[:, idx]
    out = {}
    for meth, J in [("fisher_full_khat", A.gjac(th_hat, 5)), ("fisher0_full_khat", np.repeat(A.gjac(th0[None], 5), len(grp), 0))]:
        S = np.einsum("mpi,mij,mqj->mpq", J[:, idx, :], covs[meth], J[:, idx, :])
        out[meth] = np.array([x @ np.linalg.solve(s, x) for x, s in zip(e, S)]) <= chi2.ppf(0.95, 4)
    d = out["fisher0_full_khat"].astype(float) - out["fisher_full_khat"]
    print(f"{law:8s}: theta0 {100*out['fisher0_full_khat'].mean():.2f}%  theta_hat {100*out['fisher_full_khat'].mean():.2f}%  "
          f"diff {100*d.mean():+.2f} pp, paired SE {100*d.std(ddof=1)/np.sqrt(len(d)):.2f} pp ({int(np.sum(d!=0))} discordant)")

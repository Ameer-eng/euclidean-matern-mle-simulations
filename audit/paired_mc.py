"""AUDIT: paired Monte Carlo SEs for small between-method contrasts at the largest n
(same replicates for both methods), and nfev summaries."""
import numpy as np, pandas as pd
r = pd.read_csv("results/summary/replicates.csv")
big = r.groupby(["d", "nu0"]).n.transform("max") == r.n
R = r[big & (r.variant == "joint")]
def cov(df, p, m):
    se = df[f"{p}_se_{m}"]; e = df[f"{p}_err"]
    return np.where(np.isfinite(se) & (se > 0), np.abs(e) <= 1.96 * se, np.nan)
rows = []
for (d, nu0, law), g in R.groupby(["d", "nu0", "law"]):
    g = g.sort_values("rep")
    ps = ["log_nu", "log_c", "log_tau2"] if d == 3 else ["log_nu", "log_sigma2", "log_ell", "log_tau2"]
    for p in ps:
        for a, b in [("fisher_full_khat", "hess_full_khat"), ("fisher_full_khat", "fisher_full_ktrue"),
                     ("fisher_full_khat", "fisher0_full_khat")]:
            ca, cb = cov(g, p, a), cov(g, p, b)
            ok = np.isfinite(ca) & np.isfinite(cb)
            dlt = ca[ok] - cb[ok]
            rows.append(dict(d=d, nu0=nu0, law=law, p=p, a=a, b=b, diff_pp=100 * dlt.mean(),
                             paired_se_pp=100 * dlt.std(ddof=1) / np.sqrt(ok.sum()),
                             discordant=int(np.sum(dlt != 0)), n=int(ok.sum())))
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for (a, b), x in t.groupby(["a", "b"]):
    i = x.diff_pp.abs().idxmax()
    print(f"{a} vs {b}: max |diff| {abs(t.diff_pp[i]):.2f} pp at {t.loc[i, ['d','nu0','law','p']].to_dict()}, "
          f"paired SE there {t.paired_se_pp[i]:.2f} pp ({t.discordant[i]} discordant replicates); "
          f"max paired SE over cells {x.paired_se_pp.max():.2f} pp")
t.to_csv("audit/out/paired_mc_marginal.csv", index=False)
# nfev
print("\nnfev (mean, median) for n>=1e4 and at the smallest n, by (d, nu0), all laws:")
for (d, nu0), g in r.groupby(["d", "nu0"]):
    a = g[g.n >= 1e4].nfev; b = g[g.n == g.n.min()].nfev
    print(f"  d={d} nu0={nu0}: n>=1e4 mean {a.mean():.1f} median {a.median():.0f};  smallest n mean {b.mean():.1f} median {b.median():.0f}")

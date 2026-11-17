"""AUDIT: summarise audit/out/hess_reported_chart.csv (production 'impl' vs reported-chart
'rep' observed-Hessian SEs)."""
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
df = pd.read_csv("audit/out/hess_reported_chart.csv", dtype={"at": str}).rename(columns={"at": "where"})
big = df.groupby(["d", "nu0"]).n.transform("max") == df.n
idx = ["d", "nu0", "param"]

def side(x, at, mode="full"):
    a = x[(x["where"] == at) & (x["mode"] == mode)]
    p = a.pivot_table(index=idx + (["law"] if "law" in x and x.law.nunique() > 1 else []), columns="chart",
                      values=["invalid_pct", "cov_cond", "cov_uncond", "se_cv", "n_avail"], aggfunc="first")
    return p

print("=== Largest n, Gaussian, H at theta0, FULL: production (impl) vs reported chart (rep) ===")
g = df[big & (df.law == "gauss") & (df["where"] == "0") & (df["mode"] == "full")]
t = g.pivot_table(index=idx, columns="chart", values=["invalid_pct", "cov_cond", "se_cv", "n_avail"])
print(t.round(2).to_string())
print("\n=== Largest n, Gaussian, H at theta0, BLOCK (d=3) ===")
g = df[big & (df.law == "gauss") & (df["where"] == "0") & (df["mode"] == "block")]
print(g.pivot_table(index=idx, columns="chart", values=["invalid_pct", "cov_cond", "n_avail"]).round(2).to_string())
print("\n=== Largest n, all laws: max invalid % over params, H theta0 full ===")
g = df[big & (df["where"] == "0") & (df["mode"] == "full")]
print(g.groupby(["d", "nu0", "law", "chart"]).agg(max_invalid=("invalid_pct", "max"), min_avail=("n_avail", "min"))
      .unstack("chart").round(2).to_string())
print("\n=== Fig. D1 (m)-(r) analogue: mean invalid % over params and laws, H theta0 full, by n ===")
g = df[(df["where"] == "0") & (df["mode"] == "full")]
print(g.groupby(["d", "nu0", "n", "chart"]).agg(mean_invalid=("invalid_pct", "mean"), min_avail=("n_avail", "min"))
      .unstack("chart").round(1).to_string())
print("\n=== H at theta_hat: rows where impl and rep differ (coverage or validity), FULL ===")
h = df[(df["where"] == "hat") & (df["mode"] == "full")]
w = h.pivot_table(index=["d", "nu0", "n", "law", "param"], columns="chart", values=["invalid_pct", "cov_cond"])
dcov = (w[("cov_cond", "rep")] - w[("cov_cond", "impl")]).abs()
dinv = (w[("invalid_pct", "rep")] - w[("invalid_pct", "impl")]).abs()
print("max |coverage diff| (pp):", dcov.max().round(2), " at", dcov.idxmax())
print("max |invalid diff| (pp):", dinv.max().round(2), " at", dinv.idxmax())
print("largest n with any coverage diff > 0.01 pp:", dcov[dcov > 1e-6].reset_index().n.max())
print(w[(dcov > 0.5) | (dinv > 0.5)].round(2).to_string())

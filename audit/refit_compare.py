"""AUDIT (2026-11-10): effect of the KKT-restart rule (matern_sim.refit_task) on the
summaries.  Compares results/summary_prerefit/*.csv with results/summary/*.csv
and the per-fit records in results/fits_prerefit vs results/fits.

  python -m audit.refit_compare > audit/out/refit_compare.txt
"""
import glob, json, os
import numpy as np, pandas as pd

KEY = ["d", "nu0", "G", "n", "law", "variant", "param"]


def per_fit():
    rows = []
    for f in sorted(glob.glob("results/fits_prerefit/*/joint/*/rep*.npz")):
        g = f.replace("fits_prerefit", "fits", 1)
        with np.load(f) as o, np.load(g) as z:
            c = json.loads(str(z["refit_candidates"]))
            rows.append(dict(d=int(z["d"]), nu0=float(z["nu0"]), G=int(z["G"]), n=int(z["n"]),
                             law=str(z["law"]), rep=int(z["rep"]),
                             gain=float(z["loglik"]) - float(o["loglik"]),
                             pg_before=float(z["refit_pg_before"]),
                             pg_after=float(z["refit_pg_after"]),
                             best=str(z["refit_start"]), polish=int(z["refit_polish"]),
                             nu_old=float(o["theta_hat"][1]), nu_new=float(z["theta_hat"][1]),
                             dtheta=float(np.max(np.abs(z["theta_hat"] - o["theta_hat"]))),
                             bound_old="".join("1" if b else "0" for b in o["at_bound"]),
                             bound_new="".join("1" if b else "0" for b in z["at_bound"]),
                             spread=max(v["loglik"] for k, v in c.items() if k != "PRODUCTION")
                             - min(v["loglik"] for k, v in c.items() if k != "PRODUCTION"),
                             n_starts_at_best=sum(abs(v["loglik"] - float(z["loglik"])) < 1e-6
                                                  for k, v in c.items() if k != "PRODUCTION"),
                             refit_nfev=int(z["refit_nfev"])))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    pf = per_fit()
    print(f"refitted fits: {len(pf)}")
    print("log-likelihood gain: ",
          {t: int((pf.gain > t).sum()) for t in (1e-9, 1e-6, 1e-3, 1e-2, 1e-1)},
          f"max {pf.gain.max():.4f}")
    print(f"KKT residual after: max {pf.pg_after.max():.2e}; > 1e-3: {(pf.pg_after > 1e-3).sum()}")
    print("kept start:", pf.best.value_counts().to_dict(), " polish rounds:", pf.polish.value_counts().to_dict())
    print(f"max |theta change|: {pf.dtheta.max():.3f};  > 0.01: {(pf.dtheta > 0.01).sum()}")
    print("bound pattern changed:", int((pf.bound_old != pf.bound_new).sum()))
    print(pd.crosstab(pf.bound_old, pf.bound_new))
    print("spread of restart log-likelihoods (max - min over RESTART,S1-S3,DATA): "
          f"median {pf.spread.median():.1e}, > 1e-6: {(pf.spread > 1e-6).sum()}")
    print("number of the five restarts at the kept maximum (1e-6):",
          pf.n_starts_at_best.value_counts().sort_index().to_dict())
    print("by design:")
    print(pf.groupby(["d", "nu0", "n"]).agg(fits=("gain", "size"), max_gain=("gain", "max"),
                                            gain_gt_1e3=("gain", lambda g: int((g > 1e-3).sum())),
                                            max_dtheta=("dtheta", "max")))
    big = pf[pf.gain > 1e-3].sort_values("gain", ascending=False)
    print("\nfits with gain > 1e-3:")
    print(big[["d", "nu0", "n", "law", "rep", "gain", "pg_before", "pg_after", "best",
               "nu_old", "nu_new", "bound_old", "bound_new"]].to_string(index=False))
    pf.to_csv("audit/out/refit_per_fit.csv", index=False)

    for name in ("summary", "joint", "nugget"):
        o = pd.read_csv(f"results/summary_prerefit/{name}.csv")
        z = pd.read_csv(f"results/summary/{name}.csv")
        key = [k for k in KEY if k in o.columns and k in z.columns]
        if "region" in o.columns:
            key = key + ["region"]
        m = o.merge(z, on=key, suffixes=("_old", "_new"), how="outer", indicator=True)
        assert (m["_merge"] == "both").all(), name
        num = [c for c in o.columns if c not in key and pd.api.types.is_numeric_dtype(o[c])]
        changed = []
        for c in num:
            a, b = m[c + "_old"].to_numpy(float), m[c + "_new"].to_numpy(float)
            diff = ~((a == b) | (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-12 * np.maximum(np.abs(a), 1)))
            if diff.any():
                changed.append((c, int(diff.sum()), float(np.nanmax(np.abs(a - b)[diff])),
                                sorted(set(m.loc[diff, "n"].astype(int)))))
        rows_changed = 0
        print(f"\n{name}.csv: {len(m)} rows; columns with any change: {len(changed)}")
        for c, k, mx, ns in changed:
            print(f"  {c:40s} {k:5d} cells, max |diff| {mx:.4g}, n = {ns}")


if __name__ == "__main__":
    main()

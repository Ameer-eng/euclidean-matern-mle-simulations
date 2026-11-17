"""PROCESSOR AND THREAD PROVENANCE (2026-11-16) of the post-fit calculations that
regenerate each replicate's data from its seeds, compared with the fit's own
processor family and thread count (results/kappa/fit_hosts.csv).

Passes (every one regenerates the data and checks it against a stored value):
  kappa   results/kappa/**/rep*.npz        kappa-tilde (eq. A.54); check: relative
                                           difference of l(theta_hat) on the regenerated
                                           data vs the stored fit (loglik_rel_diff)
  hess0   results/hess0/**/rep*.npz        Hessian at theta_0; check: oracle nugget
                                           criterion vs the stored fit (oracle_rel_diff)
  score0  audit/out/score0{,_local}/{ok,mismatch}/**/rep*.npz
                                           score at theta_0 (Table E3 diagnostics);
                                           same check; 'mismatch' files failed it
The initialization audit (audit/multistart.py -> audit/out/ms_local/) is separate: it
regenerated 152 data sets on this AMD login node, whatever the fit's family, and
records data_match per data set (see audit/out/appendixE/q3_q4.txt).

  python -m audit.postfit_processor_audit
Outputs: audit/out/postfit_processor.csv.gz (one row per file) and stdout
(audit/out/postfit_processor.txt).
"""
import glob
import os

import numpy as np
import pandas as pd

KEY = ["d", "nu0", "G", "rep", "law", "variant"]


def scan(pattern, pass_, check):
    rows = []
    for f in glob.glob(pattern, recursive=True):
        z = np.load(f)
        rows.append(dict(pass_=pass_, file=f, d=int(z["d"]), nu0=float(z["nu0"]), G=int(z["G"]),
                         rep=int(z["rep"]), law=str(z["law"]), variant=str(z["variant"]),
                         host=str(z["host"]) if "host" in z.files else "unrecorded",
                         family=str(z["family"]) if "family" in z.files else "unrecorded",
                         threads=int(str(z["threads"])) if "threads" in z.files else -1,
                         check=float(z[check]) if check in z.files else np.nan,
                         subdir=f.split("/")[3] if pass_.startswith("score0") else ""))
    return rows


def main():
    H = pd.read_csv("results/kappa/fit_hosts.csv").rename(
        columns={"family": "fit_family", "cpus": "fit_threads", "host": "fit_host"})
    rows = scan("results/kappa/**/rep*.npz", "kappa", "loglik_rel_diff")
    rows += scan("results/hess0/**/rep*.npz", "hess0", "oracle_rel_diff")
    rows += scan("audit/out/score0/**/rep*.npz", "score0", "oracle_rel_diff")
    rows += scan("audit/out/score0_local/**/rep*.npz", "score0_local", "oracle_rel_diff")
    T = pd.DataFrame(rows).merge(H[KEY + ["fit_host", "fit_family", "fit_threads"]], on=KEY, how="left")
    T["same_family"] = T.family == T.fit_family
    T["same_threads"] = T.threads == T.fit_threads
    T["matched"] = T.same_family & T.same_threads
    T.to_csv("audit/out/postfit_processor.csv.gz", index=False)
    pd.set_option("display.width", 200)
    print(f"fits in fit_hosts.csv: {len(H)}; fit families x threads:")
    print(H.groupby(["fit_family", "fit_threads"]).size().unstack(fill_value=0).to_string())
    for p, g in T.groupby("pass_"):
        sub = [("all", g)] if not p.startswith("score0") else [(s, x) for s, x in g.groupby("subdir")]
        for s, x in sub:
            print(f"\n{p} [{s}]: {len(x)} files; matched family and threads: {int(x.matched.sum())}; "
                  f"same family only: {int((x.same_family & ~x.same_threads).sum())}; "
                  f"different family: {int((~x.same_family).sum())}; fit host unknown: "
                  f"{int(x.fit_family.isna().sum())}; processor not recorded in the file: "
                  f"{int((x.family == 'unrecorded').sum())}")
            print(f"  data check (relative difference): max {np.nanmax(x.check):.3e}; "
                  f"> 1e-8: {int((x.check > 1e-8).sum())}; > 1e-12: {int((x.check > 1e-12).sum())}")
            print("  by (family, threads): " + str(x.groupby(["family", "threads"]).size().to_dict()))


if __name__ == "__main__":
    main()

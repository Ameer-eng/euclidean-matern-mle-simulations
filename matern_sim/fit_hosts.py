"""Map every stored fit to the node, CPU family and thread count that produced it.

  python -m matern_sim.fit_hosts --out results     # writes results/kappa/fit_hosts.csv

The latent field is drawn from an eigendecomposition of S(theta0).  For the smooth
kernel (nu0 = 3/2) the blocks are badly conditioned, and the signs of some
eigenvectors depend on the rounding of the OpenBLAS kernels, which differ by CPU
family (AVX2 vs AVX-512 code paths) and, for large blocks, by thread count.  A
replicate's data are therefore regenerated exactly only on the CPU family and with
the thread count of its fit (checked by kappa_task through the log-likelihood).
The family is the node's Slurm feature set (intel / intel_avx512 / amd /
amd_avx512, from sinfo); the thread count is the task's cpus (from its log).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess

import pandas as pd

LINE = re.compile(r"^\[[0-9:]+\] rep (\d+) (\S+)\s+(\S+)\s+nfev=")


def node_features():
    out = subprocess.run(["sinfo", "-N", "-h", "-o", "%N %f"], capture_output=True,
                         text=True, check=True).stdout
    feats = {}
    for line in out.splitlines():
        node, f = line.split()
        fs = set(f.split(","))
        feats[node] = ("amd" if "amd" in fs else "intel") + ("_avx512" if "avx512" in fs else "")
    return feats


def scan_logs(logdir="logs"):
    rows = []
    for f in glob.glob(os.path.join(logdir, "fit_*.out")):
        host = args = cpus = None
        with open(f, errors="replace") as fh:
            for line in fh:
                if line.startswith("host="):
                    host = line.split()[0][5:].split(".")[0]
                    m = re.search(r"cpus=(\d+)", line)
                    cpus = int(m.group(1)) if m else None
                elif line.startswith("args:"):
                    a = line.split()
                    args = dict(d=int(a[a.index("--d") + 1]), nu0=float(a[a.index("--nu0") + 1]),
                                G=int(a[a.index("--G") + 1]))
                else:
                    m = LINE.match(line)
                    if m and host and args:
                        rows.append(dict(**args, rep=int(m.group(1)), law=m.group(2),
                                         variant=m.group(3), host=host, cpus=cpus,
                                         mtime=os.path.getmtime(f), log=os.path.basename(f)))
    return pd.DataFrame(rows)


def fit_host_table():
    """DataFrame (d, nu0, G, rep, law, variant, host, cpus, family), one row per fit."""
    df = scan_logs()
    if df.empty:
        return df
    feats = node_features()
    df["family"] = df["host"].map(feats).fillna("unknown")
    # a fit logged twice (requeued task): the file on disk is the first write
    return df.sort_values("mtime").drop_duplicates(
        ["d", "nu0", "G", "rep", "law", "variant"]).drop(columns=["mtime"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)
    df = fit_host_table()
    path = os.path.join(a.out, "kappa", "fit_hosts.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"{len(df)} fits mapped -> {path}")
    print(df.groupby(["family", "cpus"]).size().to_string())


if __name__ == "__main__":
    main()

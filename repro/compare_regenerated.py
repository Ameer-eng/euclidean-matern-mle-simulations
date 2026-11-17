"""Compare regenerated summaries/report/figures with the frozen copies.

  python repro/compare_regenerated.py OUTDIR [FROZEN_SUMMARY_DIR] [FROZEN_FIG_DIR]

Defaults: results/summary and paper/figures.  CSV files and report.txt are compared
byte for byte (and, if they differ, numerically).  PDFs are compared after removing
the /CreationDate and /ModDate entries, the only fields that depend on the run time,
and the final startxref byte offset, which shifts with the length of the date string.
Also lists the top-level input directories recorded in OUTDIR/opened_files.tsv.
"""
import collections
import hashlib
import os
import re
import sys

import numpy as np
import pandas as pd

out = sys.argv[1]
fsum = sys.argv[2] if len(sys.argv) > 2 else "results/summary"
ffig = sys.argv[3] if len(sys.argv) > 3 else "paper/figures"
ok = True


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


print(f"regenerated: {out}\nfrozen summaries: {fsum}\nfrozen figures: {ffig}\n")
for f in ["replicates.csv", "summary.csv", "joint.csv", "nugget.csv", "theory.csv", "report.txt"]:
    a, b = os.path.join(fsum, f), os.path.join(out, "summary", f)
    if not os.path.exists(a):
        # replicates.csv is distributed in the data archive, not in the repository
        print(f"{f:16s} SKIPPED (no frozen copy in {fsum})")
        continue
    if md5(a) == md5(b):
        print(f"{f:16s} byte-identical  md5 {md5(a)}")
        continue
    ok = False
    if f.endswith(".csv"):
        x, y = pd.read_csv(a), pd.read_csv(b)
        if x.shape != y.shape or list(x.columns) != list(y.columns):
            print(f"{f:16s} DIFFERENT shape/columns {x.shape} vs {y.shape}")
            continue
        num = x.select_dtypes("number").columns
        dmax = np.nanmax(np.abs(x[num].to_numpy(float) - y[num].to_numpy(float)))
        print(f"{f:16s} DIFFERENT bytes; max |numeric difference| {dmax:.3g}")
    else:
        print(f"{f:16s} DIFFERENT")
# the date strings, and the trailer byte offset that their length shifts
pat = re.compile(rb"/(CreationDate|ModDate) *\(D:[^)]*\)|startxref\s+\d+")
figs = sorted(p for p in os.listdir(ffig) if p.endswith(".pdf"))
nid = 0
for f in figs:
    a, b = os.path.join(ffig, f), os.path.join(out, "figures", f)
    if not os.path.exists(b):
        print(f"{f:32s} MISSING")
        ok = False
        continue
    same = pat.sub(b"", open(a, "rb").read()) == pat.sub(b"", open(b, "rb").read())
    nid += same
    if not same:
        ok = False
        print(f"{f:32s} DIFFERENT (after removing dates)")
print(f"figures: {nid} of {len(figs)} identical apart from creation/modification dates (and the startxref offset)")
log = os.path.join(out, "opened_files.tsv")
if os.path.exists(log):
    rows = [l.rstrip("\n").split("\t") for l in open(log) if l.strip()]
    top = collections.Counter("/".join(p.split("/")[:2]) for _, p in rows)
    pre = [p for _, p in rows if "prerefit" in p]
    print(f"\ninputs opened: {len(rows)} (by directory: {dict(top)})")
    print(f"pre-refit files opened: {len(pre)}")
    ok &= not pre
print("\nRESULT:", "all regenerated outputs match the frozen ones" if ok else "DIFFERENCES FOUND")
sys.exit(0 if ok else 1)

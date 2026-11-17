#!/bin/bash
# Regenerate the simulation summaries, report and figures from the SAVED per-fit
# outputs (no data generation, no fitting) and compare them with the frozen copies.
#
#   repro/regenerate.sh [OUTDIR]          default OUTDIR = repro/regen
#
# Inputs (read-only, through symlinks in OUTDIR): results/{fits,kappa,hess0,theory}.
# These are the post-refit files; the originals of the 143 refitted fits are in
# results/{fits,kappa}_prerefit and are never read (repro/guard.py enforces this).
# Outputs: OUTDIR/summary/{replicates,summary,joint,nugget,theory}.csv, report.txt,
# OUTDIR/figures/*.pdf (incl. matern_mle_se_d{3,4,5}.pdf), OUTDIR/fig_e1_check/,
# OUTDIR/opened_files.tsv, OUTDIR/comparison.txt.
# Wall time on 4 cores: ~8 min aggregate, ~1 min report, a few minutes figures.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJ=$PWD
source slurm/env.sh                       # python 3.12.4 + .venv (requirements.lock.txt)
OUT=${1:-repro/regen}
case "$OUT" in results|results/*) echo "refusing to write into results/"; exit 2;; esac
mkdir -p "$OUT/summary" "$OUT/figures"
for d in fits kappa hess0 theory; do ln -sfn "$PROJ/results/$d" "$OUT/$d"; done
: > "$OUT/opened_files.tsv"
G="python $PROJ/repro/guard.py $OUT/opened_files.tsv -m"
$G matern_sim.aggregate --out "$OUT"
$G matern_sim.report --out "$OUT" > "$OUT/summary/report.txt"
# matplotlib is used only for the figures; it is not in .venv (see repro/REPRODUCE.md)
export PYTHONPATH="$PROJ/repro/pylib${PYTHONPATH:+:$PYTHONPATH}"
export SOURCE_DATE_EPOCH=1794268800       # fixed PDF creation date (2026-11-10 UTC)
$G matern_sim.plots --out "$OUT" --fig "$OUT/figures"
# Figure E1: combined and per-dimension figures must plot identical arrays, and the
# combined figure must differ from its pre-relabel version only in the row-3 label
$G audit.appendixE_fig_e1_by_dim --results "$OUT" --out "$OUT/fig_e1_check"
python repro/compare_regenerated.py "$OUT" | tee "$OUT/comparison.txt"

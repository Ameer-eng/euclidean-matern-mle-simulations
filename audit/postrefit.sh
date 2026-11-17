#!/bin/bash
# AUDIT (2026-11-10): rebuild every summary after the KKT-restart refits
# (matern_sim.refit_task).  Run on the login/OOD node from the project root:
#   bash audit/postrefit.sh            (logs in logs/postrefit_*.log)
# Pre-refit copies: results/summary_prerefit, paper/figures_prerefit,
# audit/out/*_prerefit.*
set -o pipefail
source /etc/profile 2>/dev/null; source slurm/env.sh
export OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
PYLIB=${PYLIB:-repro/pylib}    # matplotlib for the figures (see repro/REPRODUCE.md)
L=logs/postrefit
step() { echo "[$(date +%H:%M:%S)] $1"; }
step aggregate;  python -m matern_sim.aggregate --out results > ${L}_aggregate.log 2>&1 || exit 1
step report;     python -m matern_sim.report --out results > results/summary/report.txt 2> ${L}_report.log || exit 1
step plots;      PYTHONPATH=$PYLIB python -m matern_sim.plots --out results --fig paper/figures > ${L}_plots.log 2>&1 || exit 1
step audit_fits; python audit/audit_fits.py audit/out/fits_audit.pkl > ${L}_audit_fits.log 2>&1 || exit 1
step constraint; python audit/constraint_by_cell.py > ${L}_constraint.log 2>&1 || exit 1
step compare;   python -m audit.refit_compare > audit/out/refit_compare.txt 2> ${L}_compare.log || exit 1
step claims;     python -m audit.refit_claims > audit/out/refit_claims.txt 2> ${L}_claims.log || exit 1
step hess_chart; python -m audit.hess_reported_chart > ${L}_hess_chart.log 2>&1 && \
                 python -m audit.hess_reported_summary > ${L}_hess_summary.log 2>&1 || exit 1
step ALLDONE

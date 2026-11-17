#!/bin/bash
# Self-chaining sweep that finishes the campaign without intervention, in two phases:
#   fit    queue every missing fit of the main campaign, then run again once those
#          jobs have ended (at most MAX_ROUNDS sweeps);
#   kappa  then compute the residual fourth-cumulant estimate kappa_tilde (eq. A.54,
#          matern_sim.kappa_task) for every fit lacking it, each on the CPU family
#          and thread count of its fit (at most 2 sweeps; the last one also takes
#          fits whose node is unknown);
# and finally submit the aggregation.  Start it once, after the jobs in flight:
#   sbatch --dependency=afterany:<id>[:<id>...] slurm/sweep.sbatch 1 3          # fit phase
#   sbatch slurm/sweep.sbatch 1 2 kappa                                           # kappa only
# Progress is appended to logs/sweep.log.  SWEEP_DRY=1 prints the sbatch commands
# instead of running them (for testing outside Slurm).
#SBATCH --job-name=sweep
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
ROUND="${1:-1}"
MAX_ROUNDS="${2:-3}"
PHASE="${3:-fit}"
DRY="${SWEEP_DRY:-0}"
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$PROJECT_DIR/slurm/env.sh"              # also cd's to PROJECT_DIR
LOG=logs/sweep.log
[[ "$DRY" == 1 ]] && LOG=logs/sweep_dry.log
say() { echo "[$(date '+%Y-%m-%d %H:%M')] sweep $PHASE round $ROUND: $*" | tee -a "$LOG"; }

# Submitting from inside a job: drop this job's SLURM_* variables so its settings
# (CPUs, memory, job id) cannot leak into the jobs submitted below.
for v in $(compgen -e | grep -E '^SLURM_' || true); do unset "$v"; done
SB=(sbatch)
[[ "$DRY" == 1 ]] && SB=(echo sbatch)

submit_plan_and_rechain() {   # $1 = plan; next sweep: round $2, phase $3
    local plan="$1" ids
    if [[ "$DRY" == 1 ]]; then
        slurm/submit.sh "$plan" --dry-run | tee -a "$LOG"
        ids="DRY"
    else
        ids=$(slurm/submit.sh "$plan" | tee -a "$LOG" | awk '{print $1}' | paste -sd: -)
    fi
    say "submitted $(wc -l < "$plan") array job(s): ${ids}; next sweep runs after they end"
    "${SB[@]}" --dependency="afterany:${ids}" slurm/sweep.sbatch "$2" "$MAX_ROUNDS" "$3" \
        | tee -a "$LOG"
}

if [[ "$PHASE" == fit ]]; then
    say "checking for missing fits"
    python -m matern_sim.make_manifest fit --name "sweep${ROUND}" --skip-done \
        --target-hours 3 --safety 3 | tee -a "$LOG"
    PLAN="manifests/fit_sweep${ROUND}.plan"
    if [[ -s "$PLAN" ]] && (( ROUND <= MAX_ROUNDS )); then
        submit_plan_and_rechain "$PLAN" $((ROUND + 1)) fit
        exit 0
    fi
    if [[ -s "$PLAN" ]]; then
        say "fits still missing after $((ROUND - 1)) sweeps (ranges in $PLAN, logs in logs/)"
    else
        say "ALL FITS DONE"
    fi
    PHASE=kappa; ROUND=1; MAX_ROUNDS=2
fi

# ---- kappa phase
ANY=()
(( ROUND >= MAX_ROUNDS )) && ANY=(--any-family)
say "checking for fits without kappa_tilde"
if (( ROUND <= MAX_ROUNDS )); then
    python -m matern_sim.make_manifest kappa --name "ksweep${ROUND}" --skip-done \
        --target-hours 2 --safety 3 "${ANY[@]}" | tee -a "$LOG"
    PLAN="manifests/kappa_ksweep${ROUND}.plan"
    if [[ -s "$PLAN" ]]; then
        submit_plan_and_rechain "$PLAN" $((ROUND + 1)) kappa
        exit 0
    fi
fi
NMIS=0
[[ -d results/kappa_mismatch ]] && NMIS=$(find results/kappa_mismatch -name 'rep*.json' | wc -l)
say "kappa phase over ($NMIS fits with a data mismatch in results/kappa_mismatch);" \
    "submitting the aggregation (results/summary/*.csv)"
"${SB[@]}" slurm/aggregate.sbatch | tee -a "$LOG"

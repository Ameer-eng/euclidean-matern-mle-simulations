#!/bin/bash
# Submit every array job in a plan written by `python -m matern_sim.make_manifest`.
#
#   slurm/submit.sh manifests/bench.plan
#   slurm/submit.sh manifests/fit_main.plan --throttle 50
#   slurm/submit.sh manifests/fit_main.plan --dry-run
#   slurm/submit.sh manifests/fit_main.plan --after 123456      # start after job 123456
#
# Plan lines: manifest job_name cpus mem_GB time_min n_tasks [module [sbatch options]]
# (tab-separated; module defaults to matern_sim.run_task, see slurm/run_lines.sbatch).
# Optional: SBATCH_ACCOUNT=<account> / SBATCH_PARTITION=<partition> in the environment.
# Job IDs are appended to logs/submitted_<plan>.txt (used by slurm/aggregate.sbatch).
set -euo pipefail
cd "$(dirname "$0")/.."
PLAN="$1"; shift
THROTTLE=100; DRY=0; AFTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --throttle) THROTTLE="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --after) AFTER="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 1;;
  esac
done
mkdir -p logs
RECORD="logs/submitted_$(basename "$PLAN" .plan).txt"
while IFS=$'\t' read -r MANIFEST NAME CPUS MEM TMIN NTASK MODULE EXTRA; do
  [[ -z "$MANIFEST" ]] && continue
  ARGS=(--job-name "$NAME" --array "1-${NTASK}%${THROTTLE}" --cpus-per-task "$CPUS"
        --mem "${MEM}G" --time "$TMIN")
  [[ -n "${MODULE:-}" ]] && ARGS+=(--export "ALL,TASK_MODULE=${MODULE}")
  # shellcheck disable=SC2206
  [[ -n "${EXTRA:-}" ]] && ARGS+=(${EXTRA})
  [[ -n "$AFTER" ]] && ARGS+=(--dependency "afterany:${AFTER}")
  if [[ $DRY -eq 1 ]]; then
    echo sbatch "${ARGS[@]}" slurm/run_lines.sbatch "$MANIFEST"
  else
    JID=$(sbatch --parsable "${ARGS[@]}" slurm/run_lines.sbatch "$MANIFEST")
    echo "$JID $NAME $MANIFEST ($NTASK tasks, ${CPUS} cpus, ${MEM}G, ${TMIN} min)"
    echo "$JID" >> "$RECORD"
  fi
done < "$PLAN"

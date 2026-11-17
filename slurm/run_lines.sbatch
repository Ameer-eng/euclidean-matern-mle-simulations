#!/bin/bash
# Generic array job: task i runs `python -m $TASK_MODULE <line i of MANIFEST>`
# (TASK_MODULE defaults to matern_sim.run_task; slurm/submit.sh sets it from the plan).
# CPUs, memory, time, array range and job name are set on the sbatch command line
# by slurm/submit.sh (they differ by setting); defaults below are only fallbacks.
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --requeue
set -euo pipefail
MANIFEST="$1"
export PROJECT_DIR="${SLURM_SUBMIT_DIR}"
source "$PROJECT_DIR/slurm/env.sh"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
if [[ -z "$LINE" ]]; then
  echo "no line ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
  exit 1
fi
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK}"
echo "args: ${LINE}"
# shellcheck disable=SC2086
python -m "${TASK_MODULE:-matern_sim.run_task}" ${LINE}

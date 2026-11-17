#!/bin/bash
# Regenerate summaries, report and figures from the saved inputs and compare them
# with the frozen copies (repro/regenerate.sh).
#   OUT=repro/regen_<DATE> sbatch slurm/regenerate.sbatch
#SBATCH --job-name=regenerate
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
repro/regenerate.sh "${OUT:-repro/regen}"

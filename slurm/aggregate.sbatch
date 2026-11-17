#!/bin/bash
# Summarise all fits and theory files into results/summary/*.csv.
#   sbatch slurm/aggregate.sbatch
#   sbatch --dependency=afterany:<jobid>[:<jobid>...] slurm/aggregate.sbatch
#SBATCH --job-name=aggregate
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR}"
source "$PROJECT_DIR/slurm/env.sh"
python -m matern_sim.aggregate --out "${OUT:-results}"

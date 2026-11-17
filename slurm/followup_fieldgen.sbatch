#!/bin/bash
# FOLLOW-UP Task 3: numerical error of the field-generation factor on the largest grids.
#   sbatch --constraint="amd&avx512" slurm/followup_fieldgen.sbatch
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=fieldgen
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR}"
source "$PROJECT_DIR/slurm/env.sh"
echo "host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK}"
python -m audit.field_generation_check --out audit/out/followup \
  --cases 5:0.5:10 5:1.5:10 4:0.5:18 4:1.5:18 3:0.5:46 3:1.5:46

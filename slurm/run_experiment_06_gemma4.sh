#!/bin/bash
# ============================================================================
# GAO — Standalone runner for experiment 06_gemma4
# ============================================================================
# Submit with:
#   sbatch slurm/run_experiment_06_gemma4.sh
#
# Equivalent to:
#   sbatch --array=6 slurm/run_experiments.sh
# ============================================================================

#SBATCH --job-name=gao-exp06
#SBATCH --account=project_2013898
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --output=slurm/logs/exp06_%j.out
#SBATCH --error=slurm/logs/exp06_%j.err

set -euo pipefail
# Under SLURM, ${BASH_SOURCE[0]} resolves to the staged copy in
# /var/spool/slurmd/job<id>/slurm_script, so we locate the helper
# relative to $SLURM_SUBMIT_DIR (the dir from which sbatch was run).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_run_experiment_common.sh
source "${SCRIPT_DIR}/_run_experiment_common.sh"

run_gao_experiment \
    "06_gemma4" \
    "configs/experiments/06_gemma4.yaml" \
    gemma4:31b gemma4:26b gemma4:e4b gemma4:e2b

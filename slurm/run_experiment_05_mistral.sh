#!/bin/bash
# ============================================================================
# GAO — Standalone runner for experiment 05_mistral
# ============================================================================
# Homogeneous baseline: mistral-medium-3.5:128b (~70+ GB Q4). gpumedium
# (4× A100). Heterogeneous pool: Ministral-3 3B / 8B / 14B.
#
# Submit with:
#   sbatch slurm/run_experiment_05_mistral.sh
# ============================================================================

#SBATCH --job-name=gao-exp05
#SBATCH --account=project_2013898
#SBATCH --partition=gpumedium
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:4
#SBATCH --output=slurm/logs/exp05_%j.out
#SBATCH --error=slurm/logs/exp05_%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_run_experiment_common.sh
source "${SCRIPT_DIR}/_run_experiment_common.sh"

run_gao_experiment \
    "05_mistral" \
    "configs/experiments/05_mistral.yaml" \
    mistral-medium-3.5:128b ministral-3:14b ministral-3:8b ministral-3:3b

#!/bin/bash
# ============================================================================
# GAO — Standalone runner for experiment 03_qwen3.5_homo_27b
# ============================================================================
# Homogeneous baseline: qwen3.5:27b. gpumedium (4× A100 per Mahti policy).
#
# Submit with:
#   sbatch slurm/run_experiment_03_qwen3.5_homo_27b.sh
# ============================================================================

#SBATCH --job-name=gao-exp03
#SBATCH --account=project_2013898
#SBATCH --partition=gpumedium
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:4
#SBATCH --output=slurm/logs/exp03_%j.out
#SBATCH --error=slurm/logs/exp03_%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_run_experiment_common.sh
source "${SCRIPT_DIR}/_run_experiment_common.sh"

run_gao_experiment \
    "03_qwen3.5_homo_27b" \
    "configs/experiments/03_qwen3.5_homo_27b.yaml" \
    qwen3.5:27b qwen3.5:9b qwen3.5:4b qwen3.5:2b

#!/bin/bash
# ============================================================================
# GAO — Standalone runner for experiment 01_qwen3.5_homo_122b
# ============================================================================
# Homogeneous baseline: qwen3.5:122b (~70 GB Q4). Uses gpumedium (4× A100).
#
# Submit with:
#   sbatch slurm/run_experiment_01_qwen3.5_homo_122b.sh
#
# Equivalent to:
#   sbatch --array=1 slurm/run_experiments.sh
# ============================================================================

#SBATCH --job-name=gao-exp01
#SBATCH --account=project_2013898
#SBATCH --partition=gpumedium
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:4
#SBATCH --output=slurm/logs/exp01_%j.out
#SBATCH --error=slurm/logs/exp01_%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_run_experiment_common.sh
source "${SCRIPT_DIR}/_run_experiment_common.sh"

run_gao_experiment \
    "01_qwen3.5_homo_122b" \
    "configs/experiments/01_qwen3.5_homo_122b.yaml" \
    qwen3.5:122b qwen3.5:9b qwen3.5:4b qwen3.5:2b

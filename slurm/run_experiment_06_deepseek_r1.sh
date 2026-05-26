#!/bin/bash
# ============================================================================
# GAO — Standalone runner for experiment 06_deepseek_r1
# ============================================================================
# DeepSeek R1 family: 70B homogeneous, 1.5B / 7B / 8B heterogeneous pool.
#
# Submit with:
#   sbatch slurm/run_experiment_06_deepseek_r1.sh
# ============================================================================

#SBATCH --job-name=gao-exp06
#SBATCH --account=project_2013898
#SBATCH --partition=gpumedium
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:4
#SBATCH --output=slurm/logs/exp06_%j.out
#SBATCH --error=slurm/logs/exp06_%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_run_experiment_common.sh
source "${SCRIPT_DIR}/_run_experiment_common.sh"

run_gao_experiment \
    "06_deepseek_r1" \
    "configs/experiments/06_deepseek_r1.yaml" \
    deepseek-r1:70b deepseek-r1:8b deepseek-r1:7b deepseek-r1:1.5b

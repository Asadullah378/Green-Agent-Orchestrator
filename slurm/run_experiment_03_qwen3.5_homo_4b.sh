#!/bin/bash
# ============================================================================
# GAO — Standalone runner for experiment 03_qwen3.5_homo_4b
# ============================================================================
# Submit with:
#   sbatch slurm/run_experiment_03_qwen3.5_homo_4b.sh
#
# Equivalent to:
#   sbatch --array=3 slurm/run_experiments.sh
# ============================================================================

#SBATCH --job-name=gao-exp03
#SBATCH --account=project_2013898
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --output=slurm/logs/exp03_%j.out
#SBATCH --error=slurm/logs/exp03_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_run_experiment_common.sh
source "${SCRIPT_DIR}/_run_experiment_common.sh"

run_gao_experiment \
    "03_qwen3.5_homo_4b" \
    "configs/experiments/03_qwen3.5_homo_4b.yaml" \
    qwen3.5:9b qwen3.5:4b qwen3.5:2b

#!/bin/bash
# ============================================================================
# GAO — Standalone runner for experiment 02_qwen3.5_homo_9b
# ============================================================================
# Submit with:
#   sbatch slurm/run_experiment_02_qwen3.5_homo_9b.sh
#
# Equivalent to:
#   sbatch --array=2 slurm/run_experiments.sh
# ============================================================================

#SBATCH --job-name=gao-exp02
#SBATCH --account=project_2013898
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --output=slurm/logs/exp02_%j.out
#SBATCH --error=slurm/logs/exp02_%j.err

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
    "02_qwen3.5_homo_9b" \
    "configs/experiments/02_qwen3.5_homo_9b.yaml" \
    qwen3.5:9b qwen3.5:4b qwen3.5:2b

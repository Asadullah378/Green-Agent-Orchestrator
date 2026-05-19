#!/bin/bash
# ============================================================================
# GAO — Standalone runner for experiment 05_mistral
# ============================================================================
# Mistral 2026 lineup: Mistral Small 3 (24B) as the homogeneous baseline,
# Ministral-3 (3B / 8B / 14B) as the heterogeneous worker pool. All four
# checkpoints fit on a single A100-40GB at Q4_K_M.
#
# Submit with:
#   sbatch slurm/run_experiment_05_mistral.sh
#
# Equivalent to:
#   sbatch --array=5 slurm/run_experiments.sh
# ============================================================================

#SBATCH --job-name=gao-exp05
#SBATCH --account=project_2013898
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --output=slurm/logs/exp05_%j.out
#SBATCH --error=slurm/logs/exp05_%j.err

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
    "05_mistral" \
    "configs/experiments/05_mistral.yaml" \
    mistral-small:24b ministral-3:14b ministral-3:8b ministral-3:3b

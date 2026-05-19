#!/bin/bash
# ============================================================================
# GAO — Standalone runner for experiment 05_mistral
# ============================================================================
# This experiment uses `mistral-large:latest` as the homogeneous baseline.
# That model is ~73 GB on disk, so it cannot fit on a single A100-40GB.
# We therefore request 2 GPUs and bump CPU memory accordingly. The
# heterogeneous worker pool uses the Ministral-3 family (3B/8B/14B),
# which is small enough that the bottleneck is the homogeneous model.
#
# Submit with:
#   sbatch slurm/run_experiment_05_mistral.sh
#
# NOTE: This experiment is intentionally NOT a 1-GPU array task — running
# it via `sbatch --array=5 slurm/run_experiments.sh` (1 GPU) will most
# likely OOM at inference time.
# ============================================================================

#SBATCH --job-name=gao-exp05
#SBATCH --account=project_2013898
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=160G
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
    mistral-large:latest ministral-3:14b ministral-3:8b ministral-3:3b

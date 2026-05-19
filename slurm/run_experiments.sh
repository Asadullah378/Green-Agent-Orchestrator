#!/bin/bash
# ============================================================================
# Green Agent Orchestrator — SLURM job array for the five Ollama experiments
# ============================================================================
# Submits one SLURM job per experiment (01, 02, 03, 05, 06). Each array task
# starts its own Ollama server inside Apptainer, pulls only the models the
# experiment needs, runs 7 repetitions per (task, flow), and writes all
# output under results/<experiment_name>/.
#
# Experiment 04 (Qwen 3.5 via llama.cpp) uses a different inference stack
# and is submitted separately:
#   sbatch slurm/run_experiment_04_llamacpp.sh
#
# Submit the whole batch:
#   sbatch slurm/run_experiments.sh
#
# Run a single experiment via the array (e.g. just 06):
#   sbatch --array=6 slurm/run_experiments.sh
#
# Or use the per-experiment standalone wrappers — useful when you only
# want to re-run one experiment without touching the others:
#   sbatch slurm/run_experiment_01_qwen3.5_default.sh
#   sbatch slurm/run_experiment_02_qwen3.5_homo_9b.sh
#   sbatch slurm/run_experiment_03_qwen3.5_homo_4b.sh
#   sbatch slurm/run_experiment_05_mistral.sh        # 2x A100 (large model)
#   sbatch slurm/run_experiment_06_gemma4.sh
#
# ============================================================================

#SBATCH --job-name=gao-experiments
#SBATCH --account=project_2013898
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --array=1,2,3,5,6
#SBATCH --output=slurm/logs/exp%a_%A.out
#SBATCH --error=slurm/logs/exp%a_%A.err

set -euo pipefail

# Under SLURM, ${BASH_SOURCE[0]} points at the staged copy in
# /var/spool/slurmd/job<id>/slurm_script (not the file in the repo),
# so we resolve the helper via $SLURM_SUBMIT_DIR (the directory from
# which `sbatch` was invoked — i.e. the repo root). Fall back to a
# script-relative lookup for local testing outside SLURM.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_run_experiment_common.sh
source "${SCRIPT_DIR}/_run_experiment_common.sh"

# ---------------------------------------------------------------------------
# Per-array-task dispatch table
# ---------------------------------------------------------------------------
case "${SLURM_ARRAY_TASK_ID:-1}" in
  1)
    run_gao_experiment \
        "01_qwen3.5_default" \
        "configs/experiments/01_qwen3.5_default.yaml" \
        qwen3.5:27b-q4_K_M qwen3.5:9b qwen3.5:4b qwen3.5:2b
    ;;
  2)
    run_gao_experiment \
        "02_qwen3.5_homo_9b" \
        "configs/experiments/02_qwen3.5_homo_9b.yaml" \
        qwen3.5:9b qwen3.5:4b qwen3.5:2b
    ;;
  3)
    run_gao_experiment \
        "03_qwen3.5_homo_4b" \
        "configs/experiments/03_qwen3.5_homo_4b.yaml" \
        qwen3.5:9b qwen3.5:4b qwen3.5:2b
    ;;
  5)
    run_gao_experiment \
        "05_mistral" \
        "configs/experiments/05_mistral.yaml" \
        mistral-small:24b ministral-3:14b ministral-3:8b ministral-3:3b
    ;;
  6)
    run_gao_experiment \
        "06_gemma4" \
        "configs/experiments/06_gemma4.yaml" \
        gemma4:31b gemma4:26b gemma4:e4b gemma4:e2b
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-} (expected 1,2,3,5,6)" >&2
    exit 1
    ;;
esac

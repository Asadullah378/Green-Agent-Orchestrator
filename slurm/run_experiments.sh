#!/bin/bash
# ============================================================================
# Green Agent Orchestrator — SLURM job array for the five Ollama experiments
# ============================================================================
# Submits one SLURM job per experiment (01, 02, 03, 05, 06). Each task in the
# array starts its own Ollama server inside Apptainer, pulls only the models
# that experiment needs, runs the experiment with 7 repetitions per
# (task, flow), and writes all output under results/<experiment_name>/.
#
# Experiment 04 (Qwen 3.5 via llama.cpp) is NOT part of this array because
# it needs a different inference stack. Submit it separately with:
#   sbatch slurm/run_experiment_04_llamacpp.sh
#
# Submit the array with:
#   sbatch slurm/run_experiments.sh
#
# Run a single experiment (e.g. just exp 06) with:
#   sbatch --array=6 slurm/run_experiments.sh
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

# ---------------------------------------------------------------------------
# Per-experiment dispatch table
# ---------------------------------------------------------------------------
case "${SLURM_ARRAY_TASK_ID:-1}" in
  1)
    EXP_NAME="01_qwen3.5_default"
    CONFIG="configs/experiments/01_qwen3.5_default.yaml"
    MODELS=("qwen3.5:27b-q4_K_M" "qwen3.5:9b" "qwen3.5:4b" "qwen3.5:2b")
    ;;
  2)
    EXP_NAME="02_qwen3.5_homo_9b"
    CONFIG="configs/experiments/02_qwen3.5_homo_9b.yaml"
    MODELS=("qwen3.5:9b" "qwen3.5:4b" "qwen3.5:2b")
    ;;
  3)
    EXP_NAME="03_qwen3.5_homo_4b"
    CONFIG="configs/experiments/03_qwen3.5_homo_4b.yaml"
    MODELS=("qwen3.5:9b" "qwen3.5:4b" "qwen3.5:2b")
    ;;
  5)
    EXP_NAME="05_mistral"
    CONFIG="configs/experiments/05_mistral.yaml"
    MODELS=("mistral-large:latest" "ministral:14b" "ministral:8b" "ministral:3b")
    ;;
  6)
    EXP_NAME="06_gemma4"
    CONFIG="configs/experiments/06_gemma4.yaml"
    MODELS=("gemma4:31b" "gemma4:26b" "gemma4:e4b" "gemma4:e2b")
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-} (expected 1,2,3,5,6)" >&2
    exit 1
    ;;
esac

echo "================================================================"
echo "  GAO experiment ${EXP_NAME}"
echo "  array task : ${SLURM_ARRAY_TASK_ID}"
echo "  job id     : ${SLURM_JOB_ID}"
echo "  node       : ${SLURMD_NODENAME:-unknown}"
echo "  config     : ${CONFIG}"
echo "  models     : ${MODELS[*]}"
echo "  started    : $(date)"
echo "================================================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Apptainer is available natively on Mahti; the legacy `apptainer` module
# was removed. We still try to load the modules in case they reappear, but
# we don't fail the job if they are missing — we only fail if the
# `apptainer` binary itself is not on PATH.
module load apptainer 2>/dev/null || true
module load python-data 2>/dev/null || true

if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not found on PATH. Check Mahti environment." >&2
    exit 1
fi

PROJECT_DIR="/scratch/project_2013898/ollama_env"
OLLAMA_SIF="${PROJECT_DIR}/ollama.sif"
REPO_DIR="${PROJECT_DIR}/Green-Agent-Orchestrator"

export OLLAMA_MODELS="${PROJECT_DIR}/models"
# A scratch-backed directory used as the in-container HOME so ollama can
# write its SSH key into $HOME/.ollama. Mahti's Apptainer policy blocks
# APPTAINERENV_HOME, so we use the supported `--home src:dst` flag below.
CONTAINER_HOME="${PROJECT_DIR}/container_home"
mkdir -p "${OLLAMA_MODELS}" "${CONTAINER_HOME}"

cd "${REPO_DIR}"
mkdir -p slurm/logs

# ---------------------------------------------------------------------------
# Start the Ollama server inside Apptainer
#
# `--home CONTAINER_HOME:/root` mounts a writable scratch directory as
# /root inside the container AND sets HOME=/root, which Mahti's policy
# allows. Apptainer otherwise inherits HOME=/users/<user> from the host,
# and that path is not writable inside the container.
# Then we bind our model store on top of /root/.ollama so ollama finds
# its blobs and is able to write the SSH key it generates on first start.
# ---------------------------------------------------------------------------
echo "Starting Ollama server…"
apptainer run --nv \
    --home "${CONTAINER_HOME}:/root" \
    --bind "${OLLAMA_MODELS}:/root/.ollama" \
    "${OLLAMA_SIF}" serve &
OLLAMA_PID=$!

cleanup() {
    echo "Cleaning up Ollama server (pid=${OLLAMA_PID})…"
    kill "${OLLAMA_PID}" 2>/dev/null || true
    wait "${OLLAMA_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the server to come up
echo "Waiting for Ollama API…"
for _ in {1..60}; do
    if curl -fs http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama API is up."
        break
    fi
    sleep 2
done

# ---------------------------------------------------------------------------
# Pull only the models this experiment needs
# (cached pulls are no-ops, so re-running is cheap)
# ---------------------------------------------------------------------------
echo "Pulling models for ${EXP_NAME}…"
for m in "${MODELS[@]}"; do
    echo "  → ollama pull ${m}"
    apptainer exec --nv \
        --home "${CONTAINER_HOME}:/root" \
        --bind "${OLLAMA_MODELS}:/root/.ollama" \
        "${OLLAMA_SIF}" ollama pull "${m}"
done

# ---------------------------------------------------------------------------
# Run the experiment
# ---------------------------------------------------------------------------
source .venv/bin/activate

echo "Running experiment with --config ${CONFIG}…"
python -m src.run_experiment --config "${CONFIG}"

echo "================================================================"
echo "  Experiment ${EXP_NAME} finished at $(date)"
echo "  Results in: ${REPO_DIR}/results/${EXP_NAME}/"
echo "================================================================"

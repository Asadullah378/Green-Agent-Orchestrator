#!/bin/bash
# ============================================================================
# Green Agent Orchestrator — SLURM job for experiment 04 (Qwen 3.5 via llama.cpp)
# ============================================================================
# Experiment 04 swaps the inference backend from Ollama to llama.cpp
# (accessed through llama-swap, an OpenAI-compatible proxy that hot-swaps
# the underlying GGUF model on demand). It therefore needs its own SLURM
# script: there is no Ollama server in this job, and the Apptainer image
# used here must contain `llama-server` and `llama-swap` instead.
#
# One-time setup on Mahti (do this before submitting):
#
#   1. Build / pull an Apptainer image that contains llama.cpp + llama-swap.
#      A typical recipe (Dockerfile-equivalent) installs:
#          apt: build-essential cmake curl git
#          git clone https://github.com/ggerganov/llama.cpp && cmake build
#          go install github.com/mostlygeek/llama-swap@latest
#      Save it as /scratch/project_2013898/llamacpp_env/llamacpp.sif
#
#   2. Download Qwen 3.5 GGUF checkpoints to
#      /scratch/project_2013898/llamacpp_env/models/qwen3.5/:
#          qwen3.5-2b-instruct-q4_k_m.gguf
#          qwen3.5-4b-instruct-q4_k_m.gguf
#          qwen3.5-9b-instruct-q4_k_m.gguf
#          qwen3.5-27b-instruct-q4_k_m.gguf
#      (For Qwen GGUFs see https://huggingface.co/Qwen)
#
#   3. Update the absolute paths in
#      configs/experiments/04_qwen3.5_llamacpp.swap.yaml to point at the
#      Mahti scratch paths above.
#
# Submit with:
#   sbatch slurm/run_experiment_04_llamacpp.sh
# ============================================================================

#SBATCH --job-name=gao-exp04-llamacpp
#SBATCH --account=project_2013898
#SBATCH --partition=gpumedium
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --output=slurm/logs/exp04_%j.out
#SBATCH --error=slurm/logs/exp04_%j.err

set -euo pipefail

EXP_NAME="04_qwen3.5_llamacpp"
CONFIG="configs/experiments/04_qwen3.5_llamacpp.yaml"
SWAP_CONFIG="configs/experiments/04_qwen3.5_llamacpp.swap.yaml"

echo "================================================================"
echo "  GAO experiment ${EXP_NAME} (llama.cpp backend)"
echo "  job id  : ${SLURM_JOB_ID}"
echo "  node    : ${SLURMD_NODENAME:-unknown}"
echo "  config  : ${CONFIG}"
echo "  started : $(date)"
echo "================================================================"

module load apptainer
module load python-data

PROJECT_DIR="/scratch/project_2013898/llamacpp_env"
LLAMACPP_SIF="${PROJECT_DIR}/llamacpp.sif"
REPO_DIR="/scratch/project_2013898/ollama_env/Green-Agent-Orchestrator"
GGUF_DIR="${PROJECT_DIR}/models/qwen3.5"

cd "${REPO_DIR}"
mkdir -p slurm/logs

# ---------------------------------------------------------------------------
# Start llama-swap inside Apptainer. llama-swap will spawn llama-server
# instances on demand using the commands declared in ${SWAP_CONFIG}.
# ---------------------------------------------------------------------------
echo "Starting llama-swap proxy on :8080…"
apptainer run --nv \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    "${LLAMACPP_SIF}" \
    llama-swap \
        --config "${REPO_DIR}/${SWAP_CONFIG}" \
        --listen :8080 &
SWAP_PID=$!

cleanup() {
    echo "Cleaning up llama-swap proxy (pid=${SWAP_PID})…"
    kill "${SWAP_PID}" 2>/dev/null || true
    wait "${SWAP_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the proxy to be reachable
echo "Waiting for llama-swap API on http://localhost:8080…"
for _ in {1..60}; do
    if curl -fs http://localhost:8080/v1/models > /dev/null 2>&1; then
        echo "llama-swap API is up."
        break
    fi
    sleep 2
done

# Sanity check: verify all four model aliases are visible
echo "Models reported by llama-swap:"
curl -fs http://localhost:8080/v1/models | python -m json.tool || true

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

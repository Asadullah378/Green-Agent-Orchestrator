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
# One-time setup on Mahti (do this before submitting). Everything lives
# under the same /scratch/project_2013898/ollama_env/ tree as the rest of
# the project — there is no separate llamacpp_env directory.
#
#   1. Build / pull an Apptainer image that contains llama.cpp + llama-swap.
#      A typical recipe (Dockerfile-equivalent) installs:
#          apt: build-essential cmake curl git
#          git clone https://github.com/ggerganov/llama.cpp && cmake build
#          go install github.com/mostlygeek/llama-swap@latest
#      Save it as /scratch/project_2013898/ollama_env/llamacpp.sif
#
#   2. Download Qwen 3.5 GGUF checkpoints to
#      /scratch/project_2013898/ollama_env/gguf/qwen3.5/:
#          qwen3.5-2b-instruct-q4_k_m.gguf
#          qwen3.5-4b-instruct-q4_k_m.gguf
#          qwen3.5-9b-instruct-q4_k_m.gguf
#          qwen3.5-27b-instruct-q4_k_m.gguf
#      The easiest way is the bundled helper, which pulls from
#      unsloth/Qwen3.5-<SIZE>-GGUF and renames to the lowercase convention:
#          bash slurm/download_qwen3.5_ggufs.sh
#      Run it on a LOGIN node — compute nodes have no outbound internet.
#      GGUFs are kept in a separate `gguf/` subfolder so they don't
#      collide with the Ollama blobs already under `models/`.
#
#   3. No manual edit of configs/experiments/04_qwen3.5_llamacpp.swap.yaml
#      is required. The repo copy keeps macOS dev paths so it stays usable
#      on a laptop. This script renders a Mahti-flavoured copy on the fly
#      under ${PROJECT_DIR}/run_artifacts/exp04_<jobid>/ and passes that
#      one to llama-swap.
#
# Submit with:
#   sbatch slurm/run_experiment_04_llamacpp.sh
# ============================================================================

#SBATCH --job-name=gao-exp04-llamacpp
#SBATCH --account=project_2013898
#SBATCH --partition=gpusmall
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
LLAMACPP_SIF="${PROJECT_DIR}/llamacpp.sif"
REPO_DIR="${PROJECT_DIR}/Green-Agent-Orchestrator"
GGUF_DIR="${PROJECT_DIR}/gguf/qwen3.5"
# Writable scratch directory used as in-container HOME (Mahti policy blocks
# overriding HOME via env, so we use `--home src:dst` instead).
CONTAINER_HOME="${PROJECT_DIR}/container_home"
mkdir -p "${CONTAINER_HOME}"

cd "${REPO_DIR}"
mkdir -p slurm/logs

REQUIRED_GGUFS=(
    qwen3.5-2b-instruct-q4_k_m.gguf
    qwen3.5-4b-instruct-q4_k_m.gguf
    qwen3.5-9b-instruct-q4_k_m.gguf
    qwen3.5-27b-instruct-q4_k_m.gguf
)

# Sanity-check that the GGUF files and image are actually present, since
# llama-swap's failure mode for a missing model is a confusing
# "process exited with status 1" several minutes into the run.
missing=0
for f in "${REQUIRED_GGUFS[@]}"; do
    if [[ ! -f "${GGUF_DIR}/${f}" ]]; then
        echo "ERROR: missing GGUF file ${GGUF_DIR}/${f}" >&2
        missing=1
    fi
done
if [[ "${missing}" -ne 0 ]]; then
    cat >&2 <<EOF

One or more Qwen 3.5 GGUF checkpoints are missing in ${GGUF_DIR}.
Compute nodes have no outbound internet, so the files have to be
downloaded once from a Mahti LOGIN node:

  ssh mahti-login11    # or any login node
  cd ${REPO_DIR}
  bash slurm/download_qwen3.5_ggufs.sh

That helper pulls the four Q4_K_M GGUFs from unsloth/Qwen3.5-<SIZE>-GGUF,
places them in ${GGUF_DIR}, and renames them to the
qwen3.5-<size>-instruct-q4_k_m.gguf convention this experiment uses.

Once it finishes, re-submit:
  sbatch slurm/run_experiment_04_llamacpp.sh
EOF
    exit 1
fi
if [[ ! -f "${LLAMACPP_SIF}" ]]; then
    echo "ERROR: missing Apptainer image ${LLAMACPP_SIF}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Render a Mahti-flavoured copy of the swap config.
#
# The repo-tracked configs/experiments/04_qwen3.5_llamacpp.swap.yaml is
# intentionally kept with macOS dev paths so it is usable for local
# development on a laptop. On Mahti we rewrite every `--model` path to
# point at the scratch GGUF directory, and stage the result in a
# per-job artifacts folder so concurrent submissions don't clobber each
# other.
# ---------------------------------------------------------------------------
ARTIFACT_DIR="${PROJECT_DIR}/run_artifacts/exp04_${SLURM_JOB_ID:-local}"
mkdir -p "${ARTIFACT_DIR}"
RENDERED_SWAP_CONFIG="${ARTIFACT_DIR}/04_qwen3.5_llamacpp.swap.yaml"

# Replace any "/Users/<name>/models/qwen3.5/" or "~/models/qwen3.5/" prefix
# with the Mahti GGUF directory. Both spellings are supported so the repo
# copy can be edited freely without breaking the HPC run.
sed -E \
    -e "s#/Users/[^/]+/models/qwen3\.5/#${GGUF_DIR}/#g" \
    -e "s#~/models/qwen3\.5/#${GGUF_DIR}/#g" \
    "${REPO_DIR}/${SWAP_CONFIG}" > "${RENDERED_SWAP_CONFIG}"

echo "Rendered llama-swap config (paths rewritten for Mahti):"
echo "  ${RENDERED_SWAP_CONFIG}"
grep -E "^\s*--model" "${RENDERED_SWAP_CONFIG}" || true

# ---------------------------------------------------------------------------
# Start llama-swap inside Apptainer. llama-swap will spawn llama-server
# instances on demand using the commands declared in the rendered config.
# ---------------------------------------------------------------------------
echo "Starting llama-swap proxy on :8080…"
apptainer run --nv \
    --home "${CONTAINER_HOME}:/root" \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    "${LLAMACPP_SIF}" \
    llama-swap \
        --config "${RENDERED_SWAP_CONFIG}" \
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

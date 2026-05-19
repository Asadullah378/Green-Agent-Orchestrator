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
# One-time setup on Mahti (do this on a LOGIN node before submitting —
# compute nodes have no outbound internet). Everything lives under
# /scratch/project_2013898/ollama_env/:
#
#   1. Prepare the llama.cpp Apptainer image and the llama-swap binary:
#          bash slurm/setup_llamacpp_env.sh
#      This pulls a CUDA-enabled llama.cpp image (provides `llama-server`)
#      into ${PROJECT_DIR}/llamacpp.sif and downloads a statically-linked
#      llama-swap Go binary into ${PROJECT_DIR}/bin/llama-swap. llama-swap
#      is run from the host (it is visible inside the container via the
#      ${PROJECT_DIR} bind mount), so it does not need to live in the SIF.
#
#   2. Download the four Qwen 3.5 GGUF checkpoints into
#      ${PROJECT_DIR}/gguf/qwen3.5/:
#          bash slurm/download_qwen3.5_ggufs.sh
#      The helper pulls from unsloth/Qwen3.5-<SIZE>-GGUF and renames the
#      files to the qwen3.5-<size>-instruct-q4_k_m.gguf convention.
#      Kept in a separate `gguf/` subfolder so they don't collide with
#      the Ollama blobs already under `models/`.
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
LLAMA_SWAP_BIN="${PROJECT_DIR}/bin/llama-swap"
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
if [[ ! -f "${LLAMACPP_SIF}" || ! -x "${LLAMA_SWAP_BIN}" ]]; then
    cat >&2 <<EOF

The llama.cpp + llama-swap environment is not set up yet.
Missing:
$([[ ! -f "${LLAMACPP_SIF}"  ]] && echo "  - ${LLAMACPP_SIF}")
$([[ ! -x "${LLAMA_SWAP_BIN}" ]] && echo "  - ${LLAMA_SWAP_BIN}")

Compute nodes have no outbound internet, so both pieces have to be
prepared once from a Mahti LOGIN node:

  ssh mahti-login11    # or any login node
  cd ${REPO_DIR}
  bash slurm/setup_llamacpp_env.sh

That helper pulls the CUDA-enabled llama.cpp Apptainer image and the
llama-swap Go binary into ${PROJECT_DIR}.

Once it finishes, re-submit:
  sbatch slurm/run_experiment_04_llamacpp.sh
EOF
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
# Common Apptainer flags used both for the diagnostic and for llama-swap
# itself. Centralised so a fix in one place applies to both.
#
# We use `apptainer exec` (NOT `apptainer run`) for both: `run` invokes
# the image's runscript, and for this Docker-converted image that runscript
# wraps `ENTRYPOINT=["/app/llama-server"]`, so any positional args we
# pass would be handed to llama-server, not executed as a separate
# binary. `exec` bypasses the runscript and runs our command directly.
#
# IMPORTANT: the ggml-org llama.cpp:server-cuda image installs
# llama-server at /app/llama-server with its shared libraries
# (libllama-common.so.0 etc) sitting next to it in /app. The
# ENTRYPOINT setup arranges PATH and the dynamic-linker search; once
# we bypass it, we have to set both explicitly so:
#   * llama-swap's `cmd: llama-server` entries (in the rendered swap
#     config) resolve to /app/llama-server via PATH lookup;
#   * the dynamic linker can find libllama-common.so.0 etc.
# /usr/local/lib is included as a fallback in case a future image
# version re-locates the .so files.
# ---------------------------------------------------------------------------
APPTAINER_COMMON_FLAGS=(
    --nv
    --home "${CONTAINER_HOME}:/root"
    --bind "${PROJECT_DIR}:${PROJECT_DIR}"
    --env "PATH=/app:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    --env "LD_LIBRARY_PATH=/app:/usr/local/lib:/usr/lib/x86_64-linux-gnu"
)

# ---------------------------------------------------------------------------
# Diagnostic: verify llama-server can actually load a GGUF and serve a
# request on this GPU BEFORE handing off to llama-swap. llama-swap
# captures subprocess exits but does not forward llama-server's own
# stderr, so when an instance crashes during init all we see in the
# SLURM log is a useless "ExitError >> signal: aborted, exit code: -1".
# Running llama-server directly here surfaces the real error (bad
# flag, missing CUDA runtime, GGUF incompatibility, OOM, etc.) into
# the SLURM .err log.
#
# We use the smallest GGUF (2B) so this only adds ~30 s overhead on a
# healthy run. Set GAO_SKIP_LLAMACPP_DIAGNOSTIC=1 to skip.
# ---------------------------------------------------------------------------
if [[ "${GAO_SKIP_LLAMACPP_DIAGNOSTIC:-0}" != "1" ]]; then
    SMALL_GGUF="${GGUF_DIR}/qwen3.5-2b-instruct-q4_k_m.gguf"
    DIAG_LOG="${ARTIFACT_DIR}/llama-server-diagnostic.log"

    # Capture the host's NVIDIA driver version. This is the key piece of
    # information for diagnosing PTX JIT failures: the llama.cpp image
    # has to be built against a CUDA toolkit that the host driver can
    # talk to without forward-compat shims. See slurm/setup_llamacpp_env.sh
    # for the version-pinning rationale.
    echo "Host NVIDIA driver / CUDA snapshot:"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=driver_version,name,compute_cap \
                   --format=csv 2>&1 | head -5
    else
        apptainer exec --nv "${LLAMACPP_SIF}" \
            nvidia-smi --query-gpu=driver_version,name,compute_cap \
            --format=csv 2>&1 | head -5 || true
    fi
    echo
    echo "llama-server build info (ARCHS line reveals the CUDA toolkit major):"
    apptainer exec "${APPTAINER_COMMON_FLAGS[@]}" \
        "${LLAMACPP_SIF}" llama-server --version 2>&1 | head -3 || true
    echo

    echo "Diagnostic: starting llama-server directly with the 2B model on :19999…"
    apptainer exec "${APPTAINER_COMMON_FLAGS[@]}" \
        "${LLAMACPP_SIF}" \
        llama-server \
            --model "${SMALL_GGUF}" \
            --port 19999 \
            --host 127.0.0.1 \
            --ctx-size 1024 \
            --n-gpu-layers 99 \
            --jinja \
            --metrics \
            > "${DIAG_LOG}" 2>&1 &
    DIAG_PID=$!

    diag_ok=0
    for _ in {1..60}; do
        if curl -fs http://127.0.0.1:19999/health > /dev/null 2>&1; then
            diag_ok=1
            echo "  ✓ llama-server diagnostic OK — model loaded and /health responded."
            break
        fi
        if ! kill -0 "${DIAG_PID}" 2>/dev/null; then
            cat >&2 <<EOF

ERROR: llama-server crashed during the diagnostic. Full output:
-----------------------------------------------------------------
$(cat "${DIAG_LOG}")
-----------------------------------------------------------------

Common causes:
  - Bad CLI flag (e.g. --jinja or --metrics not supported by this
    llama-server build). Try removing them from the cmd: blocks in
    configs/experiments/04_qwen3.5_llamacpp.swap.yaml.
  - CUDA runtime / driver mismatch. Confirm the container's CUDA libs
    are compatible with the host driver (see 'nvidia-smi' on a compute
    node and the image's CUDA major version).
  - GGUF format incompatibility (e.g. the model architecture is not
    yet supported by this llama.cpp build).
EOF
            exit 1
        fi
        sleep 2
    done
    kill "${DIAG_PID}" 2>/dev/null || true
    wait "${DIAG_PID}" 2>/dev/null || true

    if [[ "${diag_ok}" -ne 1 ]]; then
        cat >&2 <<EOF

ERROR: llama-server diagnostic did not become healthy after 120 s.
Full output:
-----------------------------------------------------------------
$(cat "${DIAG_LOG}")
-----------------------------------------------------------------
EOF
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Start llama-swap inside Apptainer. llama-swap itself is a small
# statically-linked Go binary that lives on the HOST under
# ${LLAMA_SWAP_BIN}; it is visible inside the container at the same
# path via the ${PROJECT_DIR} bind mount, so we just invoke it by its
# absolute host path. The `llama-server` subprocesses it spawns are
# the ones baked into ${LLAMACPP_SIF} (with CUDA support).
# ---------------------------------------------------------------------------
echo "Starting llama-swap proxy on :8080…"
apptainer exec "${APPTAINER_COMMON_FLAGS[@]}" \
    "${LLAMACPP_SIF}" \
    "${LLAMA_SWAP_BIN}" \
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
api_up=0
for _ in {1..60}; do
    if curl -fs http://localhost:8080/v1/models > /dev/null 2>&1; then
        api_up=1
        echo "llama-swap API is up."
        break
    fi
    # Bail out early if the background llama-swap process has already
    # died — otherwise we sit through the full 120 s timeout for nothing.
    if ! kill -0 "${SWAP_PID}" 2>/dev/null; then
        echo "ERROR: llama-swap (pid=${SWAP_PID}) exited before the API came up." >&2
        echo "       Inspect slurm/logs/exp04_${SLURM_JOB_ID:-local}.err for its stderr." >&2
        exit 1
    fi
    sleep 2
done
if [[ "${api_up}" -ne 1 ]]; then
    echo "ERROR: llama-swap API never came up after 120 s on http://localhost:8080." >&2
    exit 1
fi

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

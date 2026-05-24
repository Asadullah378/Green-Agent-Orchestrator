#!/bin/bash
# ============================================================================
# One-time helper: download Qwen 3.5 GGUF checkpoints for experiment 04
# ============================================================================
# Compute nodes on Mahti have no outbound internet, so the GGUFs required
# by `run_experiment_04_llamacpp.sh` have to land on disk before you
# submit the SLURM job. Run this script ONCE on a Mahti login node.
#
# Usage:
#   bash slurm/download_qwen3.5_ggufs.sh
#
# Override the destination or HuggingFace source by exporting:
#   GGUF_DIR=/somewhere/else bash slurm/download_qwen3.5_ggufs.sh
#   HF_REPO_PREFIX=bartowski/Qwen_Qwen3.5- HF_REPO_SUFFIX=-GGUF \
#       bash slurm/download_qwen3.5_ggufs.sh
#   GAO_GGUF_SIZES="2b 4b 9b 122b" bash slurm/download_qwen3.5_ggufs.sh
#
# Existing files are skipped, so re-running is cheap.
#
# Note on the implementation:
#   We deliberately avoid the `huggingface-cli` binary. On Mahti, the
#   Tykky-containerized Python (`module load python-data`) is mounted at
#   an ephemeral path (e.g. /PUHTI_TYKKY_<id>/miniforge/envs/env1/...),
#   so a `pip install --user huggingface_hub` bakes that ephemeral path
#   into the shebang of ~/.local/bin/huggingface-cli, and the binary
#   stops working as soon as the Tykky session rolls. We instead drive
#   the `huggingface_hub` library directly via the active interpreter.
#
# Note on naming:
#   For Qwen 3.5 the bare `Qwen/Qwen3.5-NB` is already the post-trained
#   instruction model — there is no separate `-Instruct` HF repo any
#   more. We still keep the `-instruct-` infix in the local filename so
#   the rest of the project (swap config, comments, dev laptop setup)
#   continues to work unchanged.
# ============================================================================

set -euo pipefail

GGUF_DIR="${GGUF_DIR:-/scratch/project_2013898/ollama_env/gguf/qwen3.5}"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-unsloth/Qwen3.5-}"
HF_REPO_SUFFIX="${HF_REPO_SUFFIX:--GGUF}"
QUANT="${QUANT:-Q4_K_M}"
# Sizes needed for experiment 04 (hetero pool + 122B homo baseline).
GAO_GGUF_SIZES="${GAO_GGUF_SIZES:-2b 4b 9b 122b}"

echo "Target directory : ${GGUF_DIR}"
echo "HF repo template : ${HF_REPO_PREFIX}<SIZE>${HF_REPO_SUFFIX}"
echo "Quantisation     : ${QUANT}"
echo "Sizes to fetch   : ${GAO_GGUF_SIZES}"
echo

mkdir -p "${GGUF_DIR}"

# Make sure we have a usable Python, and that huggingface_hub is importable.
module load python-data 2>/dev/null || true

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: no 'python' on PATH. Try 'module load python-data' first." >&2
    exit 1
fi

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "huggingface_hub not importable — installing into your user site-packages…"
    python -m pip install --user --quiet huggingface_hub
    python -c "import huggingface_hub" || {
        echo "ERROR: huggingface_hub still not importable after pip install." >&2
        exit 1
    }
fi
echo "Using python : $(command -v python)"
python -c "import huggingface_hub, sys; print(f'huggingface_hub : {huggingface_hub.__version__}')"
echo

for lower in ${GAO_GGUF_SIZES}; do
    case "${lower}" in
        2b)   upper="2B"
              src_file="Qwen3.5-${upper}-${QUANT}.gguf"
              src_repo="${HF_REPO_PREFIX}${upper}${HF_REPO_SUFFIX}"
              ;;
        4b)   upper="4B"
              src_file="Qwen3.5-${upper}-${QUANT}.gguf"
              src_repo="${HF_REPO_PREFIX}${upper}${HF_REPO_SUFFIX}"
              ;;
        9b)   upper="9B"
              src_file="Qwen3.5-${upper}-${QUANT}.gguf"
              src_repo="${HF_REPO_PREFIX}${upper}${HF_REPO_SUFFIX}"
              ;;
        27b)  upper="27B"
              src_file="Qwen3.5-${upper}-${QUANT}.gguf"
              src_repo="${HF_REPO_PREFIX}${upper}${HF_REPO_SUFFIX}"
              ;;
        35b)  upper="35B"
              src_file="Qwen3.5-${upper}-${QUANT}.gguf"
              src_repo="${HF_REPO_PREFIX}${upper}${HF_REPO_SUFFIX}"
              ;;
        122b) upper="122B-A10B"
              # Unsloth splits the 122B Q4_K_M model into 3 pieces
              # We will handle downloading and concatenating them below
              src_file="split"
              src_repo="${HF_REPO_PREFIX}${upper}${HF_REPO_SUFFIX}"
              ;;
        *)
            echo "ERROR: unknown size '${lower}' in GAO_GGUF_SIZES" >&2
            exit 1
            ;;
    esac
    target="${GGUF_DIR}/qwen3.5-${lower}-instruct-q4_k_m.gguf"

    if [[ -f "${target}" ]]; then
        echo "✓ ${target} already exists — skipping"
        continue
    fi

    if [[ "${src_file}" == "split" ]]; then
        echo "↓ ${src_repo}  →  (split files) → ${target}"
        # The 122B model is split into 3 parts in the Q4_K_M folder
        python - <<PY
from huggingface_hub import hf_hub_download
import os

parts = [
    "Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf",
    "Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00002-of-00003.gguf",
    "Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00003-of-00003.gguf"
]

downloaded_paths = []
for p in parts:
    print(f"  Downloading {p}...")
    path = hf_hub_download(
        repo_id="${src_repo}",
        filename=p,
        local_dir="${GGUF_DIR}",
        local_dir_use_symlinks=False,
    )
    downloaded_paths.append(path)

print("  All parts downloaded. Proceeding to concatenate with llama-gguf-split...")
PY
        # The parts are downloaded to e.g. GGUF_DIR/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf
        # We need to run llama-gguf-split --merge to combine them into the target file.
        # We can use the llama.cpp Apptainer image to do this!
        
        PART1="${GGUF_DIR}/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"
        if [[ -f "${PART1}" ]]; then
            echo "  Merging split GGUF files..."
            apptainer exec \
                --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
                --env "LD_LIBRARY_PATH=/app:/usr/local/lib" \
                "${PROJECT_DIR}/llamacpp.sif" \
                /app/llama-gguf-split --merge "${PART1}" "${target}"
                
            echo "  Cleaning up split parts..."
            rm -rf "${GGUF_DIR}/Q4_K_M"
        else
             echo "ERROR: Failed to find the downloaded split file at ${PART1}" >&2
             exit 1
        fi
    else
        echo "↓ ${src_repo}  →  ${src_file}"
        python - <<PY
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="${src_repo}",
    filename="${src_file}",
    local_dir="${GGUF_DIR}",
    local_dir_use_symlinks=False,
)
print(f"  downloaded → {path}")
PY

        src_path="${GGUF_DIR}/${src_file}"
        if [[ -f "${src_path}" && "${src_path}" != "${target}" ]]; then
            mv "${src_path}" "${target}"
        fi
    fi
    echo "  → ${target}"
done

echo
echo "GGUFs in ${GGUF_DIR}:"
ls -lh "${GGUF_DIR}"/qwen3.5-*-instruct-q4_k_m.gguf 2>/dev/null || true
echo
echo "You can now submit the experiment:"
echo "  sbatch slurm/run_experiment_04_llamacpp.sh"

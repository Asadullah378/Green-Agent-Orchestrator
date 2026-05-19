#!/bin/bash
# ============================================================================
# One-time helper: download Qwen 3.5 GGUF checkpoints for experiment 04
# ============================================================================
# Compute nodes on Mahti have no outbound internet, so the four GGUFs
# required by `run_experiment_04_llamacpp.sh` have to land on disk before
# you submit the SLURM job. Run this script ONCE on a Mahti login node.
#
# Usage:
#   bash slurm/download_qwen3.5_ggufs.sh
#
# Override the destination or HuggingFace source by exporting:
#   GGUF_DIR=/somewhere/else bash slurm/download_qwen3.5_ggufs.sh
#   HF_REPO_PREFIX=bartowski/Qwen_Qwen3.5- HF_REPO_SUFFIX=-GGUF \
#       bash slurm/download_qwen3.5_ggufs.sh
#
# Existing files are skipped, so re-running is cheap.
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

echo "Target directory : ${GGUF_DIR}"
echo "HF repo template : ${HF_REPO_PREFIX}<SIZE>${HF_REPO_SUFFIX}"
echo "Quantisation     : ${QUANT}"
echo

mkdir -p "${GGUF_DIR}"
cd "${GGUF_DIR}"

# Make sure huggingface-cli is available
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not on PATH — installing huggingface_hub into your user site-packages…"
    module load python-data 2>/dev/null || true
    pip install --user --quiet huggingface_hub
    # `pip install --user` puts binaries under ~/.local/bin which may not
    # be on PATH on Mahti login nodes.
    export PATH="${HOME}/.local/bin:${PATH}"
fi

# Mapping: local (lowercase) size tag → upstream (PascalCase) size tag
declare -A SIZES=( [2b]=2B [4b]=4B [9b]=9B [27b]=27B )

for lower in 2b 4b 9b 27b; do
    upper="${SIZES[$lower]}"
    target="qwen3.5-${lower}-instruct-q4_k_m.gguf"
    src_file="Qwen3.5-${upper}-${QUANT}.gguf"
    src_repo="${HF_REPO_PREFIX}${upper}${HF_REPO_SUFFIX}"

    if [[ -f "${target}" ]]; then
        echo "✓ ${target} already exists — skipping"
        continue
    fi

    echo "↓ ${src_repo}  →  ${src_file}"
    huggingface-cli download "${src_repo}" "${src_file}" \
        --local-dir . --local-dir-use-symlinks False

    # huggingface-cli download preserves the source filename; rename to
    # the lowercase convention the SLURM script and swap config expect.
    if [[ -f "${src_file}" && "${src_file}" != "${target}" ]]; then
        mv "${src_file}" "${target}"
    fi
    echo "  → ${target}"
done

echo
echo "All four GGUFs are present in ${GGUF_DIR}:"
ls -lh "${GGUF_DIR}"/qwen3.5-*-instruct-q4_k_m.gguf
echo
echo "You can now submit the experiment:"
echo "  sbatch slurm/run_experiment_04_llamacpp.sh"

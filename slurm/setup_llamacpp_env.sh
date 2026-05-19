#!/bin/bash
# ============================================================================
# One-time helper: prepare the llama.cpp + llama-swap environment for exp 04
# ============================================================================
# Run this ONCE on a Mahti login node before submitting
# `slurm/run_experiment_04_llamacpp.sh`. Compute nodes have no outbound
# internet, so both the Apptainer image and the llama-swap binary have to
# land on scratch first.
#
# What this script does:
#   1. Pulls a CUDA-enabled llama.cpp Apptainer image (provides
#      `llama-server` inside the container) into
#      ${PROJECT_DIR}/llamacpp.sif
#   2. Downloads a statically-linked llama-swap Go binary into
#      ${PROJECT_DIR}/bin/llama-swap
#
# llama-swap is run from the HOST (not from inside the container) because
# it is a tiny statically-linked Go binary. The SLURM script already
# binds ${PROJECT_DIR} into the container, so llama-swap can spawn
# `llama-server` instances that live inside llamacpp.sif.
#
# Usage:
#   bash slurm/setup_llamacpp_env.sh
#
# Override any of these via the environment:
#   PROJECT_DIR=/elsewhere
#   LLAMACPP_SIF=/path/to/llamacpp.sif
#   LLAMA_SWAP_BIN=/path/to/llama-swap
#   LLAMACPP_IMAGE=docker://ghcr.io/ggerganov/llama.cpp:server-cuda
#   LLAMA_SWAP_VERSION=v167          # any GitHub tag from
#                                     # https://github.com/mostlygeek/llama-swap/releases
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/project_2013898/ollama_env}"
LLAMACPP_SIF="${LLAMACPP_SIF:-${PROJECT_DIR}/llamacpp.sif}"
LLAMA_SWAP_BIN="${LLAMA_SWAP_BIN:-${PROJECT_DIR}/bin/llama-swap}"

LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-docker://ghcr.io/ggml-org/llama.cpp:server-cuda}"
LLAMA_SWAP_VERSION="${LLAMA_SWAP_VERSION:-v167}"

mkdir -p "${PROJECT_DIR}" "$(dirname "${LLAMA_SWAP_BIN}")"

# Apptainer is now part of the base Mahti image but we still try the
# module in case it ever reappears.
module load apptainer 2>/dev/null || true
if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not on PATH. Run this on a Mahti login node." >&2
    exit 1
fi

# Mahti $HOME and /tmp are tiny — route apptainer's scratch usage at
# /scratch to avoid 'no space left on device' during the layer extract.
export APPTAINER_TMPDIR="${PROJECT_DIR}/.apptainer_tmp"
export APPTAINER_CACHEDIR="${PROJECT_DIR}/.apptainer_cache"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

# ----------------------------------------------------------------------------
# Step 1 — pull the llama.cpp CUDA image
# ----------------------------------------------------------------------------
if [[ -f "${LLAMACPP_SIF}" ]]; then
    echo "✓ ${LLAMACPP_SIF} already exists — skipping pull."
    ls -lh "${LLAMACPP_SIF}"
else
    echo "Pulling ${LLAMACPP_IMAGE}"
    echo "      → ${LLAMACPP_SIF}"
    echo "      (this may take a few minutes; ~3–5 GB after extraction)"
    apptainer pull "${LLAMACPP_SIF}" "${LLAMACPP_IMAGE}"
    echo "✓ pulled $(du -h "${LLAMACPP_SIF}" | cut -f1)"
fi
echo

# Quick smoke test: does `llama-server --version` work inside the image?
echo "Smoke-test llama-server inside the image:"
apptainer exec --nv "${LLAMACPP_SIF}" llama-server --version 2>&1 | head -5 || {
    echo "WARNING: llama-server failed to report its version. The image" >&2
    echo "may be missing CUDA libs or have a different binary name." >&2
}
echo

# ----------------------------------------------------------------------------
# Step 2 — download the llama-swap binary
# ----------------------------------------------------------------------------
if [[ -x "${LLAMA_SWAP_BIN}" ]]; then
    echo "✓ ${LLAMA_SWAP_BIN} already exists — skipping download."
    ls -lh "${LLAMA_SWAP_BIN}"
else
    asset_url="https://github.com/mostlygeek/llama-swap/releases/download/${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION#v}_linux_amd64.tar.gz"
    echo "Downloading llama-swap ${LLAMA_SWAP_VERSION}"
    echo "      from ${asset_url}"
    echo "      to   ${LLAMA_SWAP_BIN}"

    tmpdir="$(mktemp -d)"
    trap 'rm -rf "${tmpdir}"' EXIT

    if ! curl -fsSL "${asset_url}" -o "${tmpdir}/llama-swap.tar.gz"; then
        cat >&2 <<EOF
ERROR: failed to download ${asset_url}

The release-asset naming for llama-swap occasionally changes. Visit
    https://github.com/mostlygeek/llama-swap/releases
to find the correct linux_amd64 tarball and re-run with:

    LLAMA_SWAP_VERSION=<tag> bash slurm/setup_llamacpp_env.sh
EOF
        exit 1
    fi

    tar -xzf "${tmpdir}/llama-swap.tar.gz" -C "${tmpdir}"
    # The tarball contains a single statically-linked binary, usually
    # named just `llama-swap`. Pick the first executable file we find
    # to be tolerant of naming changes.
    extracted="$(find "${tmpdir}" -maxdepth 2 -type f -name 'llama-swap*' | head -1)"
    if [[ -z "${extracted}" ]]; then
        echo "ERROR: could not locate llama-swap binary inside the tarball." >&2
        exit 1
    fi
    install -m 0755 "${extracted}" "${LLAMA_SWAP_BIN}"
    trap - EXIT
    rm -rf "${tmpdir}"
    echo "✓ installed $(du -h "${LLAMA_SWAP_BIN}" | cut -f1)"
fi
echo

echo "Smoke-test llama-swap:"
"${LLAMA_SWAP_BIN}" --version 2>&1 | head -5 || \
    echo "(no --version flag on this build; that's fine)"
echo

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------
echo "All set:"
ls -lh "${LLAMACPP_SIF}" "${LLAMA_SWAP_BIN}"
echo
echo "Next steps:"
echo "  bash slurm/download_qwen3.5_ggufs.sh   # if you haven't already"
echo "  sbatch slurm/run_experiment_04_llamacpp.sh"

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
#   LLAMACPP_IMAGE_TAG=server-cuda-b8329   # any ghcr.io/ggml-org/llama.cpp tag.
#                                          # MUST be a CUDA ≤ 12.6 build to
#                                          # work with Mahti's host driver —
#                                          # see the long comment below.
#   LLAMACPP_IMAGE=docker://ghcr.io/ggml-org/llama.cpp:server-cuda-b8329
#                                          # full override; supersedes the tag
#   LLAMA_SWAP_VERSION=v216                # any GitHub tag from
#                                          # https://github.com/mostlygeek/llama-swap/releases
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/project_2013898/ollama_env}"
LLAMACPP_SIF="${LLAMACPP_SIF:-${PROJECT_DIR}/llamacpp.sif}"
LLAMA_SWAP_BIN="${LLAMA_SWAP_BIN:-${PROJECT_DIR}/bin/llama-swap}"

# -----------------------------------------------------------------------------
# CUDA-version pinning rationale
# -----------------------------------------------------------------------------
# ghcr.io/ggml-org/llama.cpp:server-cuda is a rolling tag. As of April 2026
# it is built against CUDA 12.8.1 (after a brief 12.9.1 detour, see
# https://github.com/ggml-org/llama.cpp/pull/21438). Mahti's installed
# CUDA module top out at 12.6.1, and the host NVIDIA driver does NOT ship
# the cuda-compat package for forward-compat to 12.8+. Result: any kernel
# that misses a precompiled SASS for sm_80 falls back to PTX JIT and
# aborts with "a PTX JIT compilation failed" the first time it runs.
#
# We therefore pin to a specific older build that:
#   * is recent enough to include Qwen 3.5 support (PR #19435/#19468,
#     merged Feb 9, 2026), and
#   * still uses CUDA 12.4 (the docker default until ~late March 2026).
#
# b5514 ships with CUDA 12.4 and works on Mahti.
# Bump cautiously: every build after roughly b8400 may be on CUDA 12.8/12.9
# and crash the same way again. Verify with:
#     apptainer exec llamacpp.sif llama-server --version  (look at ARCHS)
# A safe build emits ARCHS containing 800 (sm_80, the A100) WITHOUT 1200
# (sm_120, Blackwell — that requires CUDA 12.8+).
# -----------------------------------------------------------------------------
LLAMACPP_IMAGE_TAG="${LLAMACPP_IMAGE_TAG:-server-cuda-b5514}"
LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-docker://ghcr.io/ggml-org/llama.cpp:${LLAMACPP_IMAGE_TAG}}"

# Pin to a known-good llama-swap release. Bump as needed — see
# https://github.com/mostlygeek/llama-swap/releases for the current tag.
LLAMA_SWAP_VERSION="${LLAMA_SWAP_VERSION:-v216}"
# Path inside the ggml-org llama.cpp Docker image where `llama-server` lives.
# Verified by the smoke test below and used by run_experiment_04_llamacpp.sh
# (which sets PATH=${LLAMA_SERVER_DIR}:... inside the container).
LLAMA_SERVER_DIR_DEFAULT="/app"

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
# If an old SIF is already present, sanity-check that it isn't the
# Blackwell/CUDA-12.8+ flavour. Otherwise the diagnostic in
# slurm/run_experiment_04_llamacpp.sh will abort with a PTX JIT failure.
_image_uses_blackwell_arch() {
    local sif="$1"
    apptainer exec "${sif}" /app/llama-server --version 2>/dev/null \
        | grep -qE 'ARCHS\s*=\s*[0-9,]*1200\b'
}

if [[ -f "${LLAMACPP_SIF}" ]]; then
    if _image_uses_blackwell_arch "${LLAMACPP_SIF}"; then
        cat >&2 <<EOF
WARNING: ${LLAMACPP_SIF} looks like a CUDA 12.8+/Blackwell build
         (its ARCHS list contains sm_120). Mahti's NVIDIA driver does
         not understand CUDA 12.8 PTX, so llama-server will abort
         during model warmup with:
           CUDA error: a PTX JIT compilation failed

Removing the stale image and re-pulling ${LLAMACPP_IMAGE_TAG}…
EOF
        rm -f "${LLAMACPP_SIF}"
    else
        echo "✓ ${LLAMACPP_SIF} already exists — skipping pull."
        ls -lh "${LLAMACPP_SIF}"
    fi
fi

if [[ ! -f "${LLAMACPP_SIF}" ]]; then
    echo "Pulling ${LLAMACPP_IMAGE}"
    echo "      → ${LLAMACPP_SIF}"
    echo "      (this may take a few minutes; ~3–5 GB after extraction)"
    apptainer pull "${LLAMACPP_SIF}" "${LLAMACPP_IMAGE}"
    echo "✓ pulled $(du -h "${LLAMACPP_SIF}" | cut -f1)"
fi
echo

# Quick smoke test: find `llama-server` inside the image. The ggml-org
# server-cuda image installs it at /app/llama-server and uses it as the
# ENTRYPOINT, so `apptainer exec` (which bypasses ENTRYPOINT) doesn't see
# it on $PATH. We try the known location first, then fall back to a
# filesystem search so we have a hope of discovering future re-locations.
echo "Locating llama-server inside the image…"
LLAMA_SERVER_PATH=""
for candidate in "${LLAMA_SERVER_DIR_DEFAULT}/llama-server" \
                  /usr/local/bin/llama-server \
                  /usr/bin/llama-server; do
    if apptainer exec "${LLAMACPP_SIF}" test -x "${candidate}" 2>/dev/null; then
        LLAMA_SERVER_PATH="${candidate}"
        break
    fi
done
if [[ -z "${LLAMA_SERVER_PATH}" ]]; then
    LLAMA_SERVER_PATH="$(apptainer exec "${LLAMACPP_SIF}" \
        find / -maxdepth 5 -type f -name llama-server 2>/dev/null | head -1)"
fi

if [[ -z "${LLAMA_SERVER_PATH}" ]]; then
    echo "ERROR: could not locate a llama-server binary inside ${LLAMACPP_SIF}." >&2
    echo "       The image may have changed its layout — please inspect with:" >&2
    echo "         apptainer exec ${LLAMACPP_SIF} ls -l /app /usr/local/bin /usr/bin" >&2
    exit 1
fi
echo "✓ llama-server found at ${LLAMA_SERVER_PATH} (inside container)"

# Also locate the directory holding llama.cpp's shared libraries
# (libllama-common.so.0 etc). The image relies on Docker ENTRYPOINT being
# launched from /app for the dynamic linker's RUNPATH to work; under
# `apptainer exec` we have to set LD_LIBRARY_PATH explicitly. We probe
# the common locations and then run the smoke test with the right path.
echo "Locating llama.cpp shared libraries…"
LLAMA_LIB_DIR=""
for candidate in /app /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu; do
    if apptainer exec "${LLAMACPP_SIF}" \
            bash -c "ls ${candidate}/libllama-common.so* 2>/dev/null | head -1" \
            > /dev/null 2>&1; then
        LLAMA_LIB_DIR="${candidate}"
        break
    fi
done
if [[ -z "${LLAMA_LIB_DIR}" ]]; then
    LLAMA_LIB_DIR="$(apptainer exec "${LLAMACPP_SIF}" \
        find / -maxdepth 5 -name 'libllama-common.so*' 2>/dev/null \
        | head -1 | xargs -r dirname)"
fi
if [[ -z "${LLAMA_LIB_DIR}" ]]; then
    echo "WARNING: could not locate libllama-common.so inside the image." >&2
    echo "         Defaulting LD_LIBRARY_PATH to /app for the smoke test." >&2
    LLAMA_LIB_DIR="/app"
fi
echo "✓ llama.cpp libs found in ${LLAMA_LIB_DIR} (inside container)"

apptainer exec \
    --env "LD_LIBRARY_PATH=${LLAMA_LIB_DIR}" \
    "${LLAMACPP_SIF}" "${LLAMA_SERVER_PATH}" --version 2>&1 | head -3 || {
    echo "WARNING: llama-server --version failed even with LD_LIBRARY_PATH=${LLAMA_LIB_DIR}." >&2
    echo "         The SLURM job may still work (it adds --nv); investigate with:" >&2
    echo "           apptainer exec --nv --env LD_LIBRARY_PATH=${LLAMA_LIB_DIR} ${LLAMACPP_SIF} ${LLAMA_SERVER_PATH} --version" >&2
}
echo

# ----------------------------------------------------------------------------
# Step 2 — download the llama-swap binary
# ----------------------------------------------------------------------------
_is_elf() {
    # First four bytes of any Linux ELF executable are 0x7F 'E' 'L' 'F'.
    [[ "$(head -c 4 "$1" 2>/dev/null | od -An -c 2>/dev/null | tr -d ' ')" == "177ELF" ]]
}

if [[ -x "${LLAMA_SWAP_BIN}" ]] && _is_elf "${LLAMA_SWAP_BIN}"; then
    echo "✓ ${LLAMA_SWAP_BIN} already exists — skipping download."
    ls -lh "${LLAMA_SWAP_BIN}"
else
    # Remove any stale/corrupt file from a previous failed run so the
    # `[[ -x ... ]]` guard above doesn't keep skipping it forever.
    if [[ -e "${LLAMA_SWAP_BIN}" ]]; then
        echo "Removing stale ${LLAMA_SWAP_BIN} (not a valid ELF)…"
        rm -f "${LLAMA_SWAP_BIN}"
    fi

    asset_name="llama-swap_${LLAMA_SWAP_VERSION#v}_linux_amd64.tar.gz"
    asset_url="https://github.com/mostlygeek/llama-swap/releases/download/${LLAMA_SWAP_VERSION}/${asset_name}"
    echo "Downloading llama-swap ${LLAMA_SWAP_VERSION}"
    echo "      from ${asset_url}"
    echo "      to   ${LLAMA_SWAP_BIN}"

    tmpdir="$(mktemp -d)"
    trap 'rm -rf "${tmpdir}"' EXIT

    if ! curl -fsSL --retry 3 "${asset_url}" -o "${tmpdir}/${asset_name}"; then
        cat >&2 <<EOF
ERROR: failed to download ${asset_url}

The release-asset naming for llama-swap occasionally changes, or the
version you asked for ('${LLAMA_SWAP_VERSION}') may not exist. Visit
    https://github.com/mostlygeek/llama-swap/releases
to find a valid Linux x86_64 tarball and re-run with:

    LLAMA_SWAP_VERSION=<tag> bash slurm/setup_llamacpp_env.sh
EOF
        exit 1
    fi

    echo "  fetched $(du -h "${tmpdir}/${asset_name}" | cut -f1)"
    tar -xzf "${tmpdir}/${asset_name}" -C "${tmpdir}"

    # Find the actual `llama-swap` binary in the extracted tree. Prefer
    # an exact filename match (the tarball ships docs/checksums too);
    # fall back to any executable named llama-swap*.
    extracted="$(find "${tmpdir}" -type f -name 'llama-swap' | head -1)"
    if [[ -z "${extracted}" ]]; then
        extracted="$(find "${tmpdir}" -type f -name 'llama-swap*' \
            -not -name '*.txt' -not -name '*.md' -not -name '*.tar.gz' \
            | head -1)"
    fi
    if [[ -z "${extracted}" ]]; then
        echo "ERROR: could not locate llama-swap binary inside the tarball." >&2
        echo "Tarball contents:" >&2
        find "${tmpdir}" -mindepth 1 -maxdepth 3 -ls >&2
        exit 1
    fi
    install -m 0755 "${extracted}" "${LLAMA_SWAP_BIN}"
    trap - EXIT
    rm -rf "${tmpdir}"
    echo "✓ installed $(du -h "${LLAMA_SWAP_BIN}" | cut -f1) at ${LLAMA_SWAP_BIN}"
fi

# Hard-validate: the binary must be a Linux x86_64 ELF, otherwise the
# SLURM job will fail much later with a useless "Exec format error".
if ! _is_elf "${LLAMA_SWAP_BIN}"; then
    echo "ERROR: ${LLAMA_SWAP_BIN} is not an ELF binary." >&2
    echo "       The downloaded asset was probably an error page or the" >&2
    echo "       wrong architecture. Inspect with: file ${LLAMA_SWAP_BIN}" >&2
    rm -f "${LLAMA_SWAP_BIN}"
    exit 1
fi
echo

echo "Smoke-test llama-swap:"
if "${LLAMA_SWAP_BIN}" --version 2>&1 | head -3; then
    :
else
    echo "(no --version flag on this build; that's fine — the ELF check passed)"
fi
echo

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------
echo "All set:"
ls -lh "${LLAMACPP_SIF}" "${LLAMA_SWAP_BIN}"
echo
echo "Next steps:"
echo "  bash slurm/download_qwen3.5_ggufs.sh   # 2B/4B/9B/122B GGUFs (122B is ~70 GB)"
echo "  sbatch slurm/run_experiment_04_llamacpp.sh   # gpumedium, 4× A100"

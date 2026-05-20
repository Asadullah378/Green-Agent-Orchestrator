# ============================================================================
# Shared helpers for GAO SLURM scripts on CSC Mahti.
#
# This file is sourced by:
#   - slurm/run_experiments.sh                  (job array, all five exps)
#   - slurm/run_experiment_01_qwen3.5_default.sh
#   - slurm/run_experiment_02_qwen3.5_homo_9b.sh
#   - slurm/run_experiment_03_qwen3.5_homo_4b.sh
#   - slurm/run_experiment_05_mistral.sh
#   - slurm/run_experiment_06_gemma4.sh
#
# It encapsulates the Apptainer + Ollama lifecycle (start server, pull
# models, run experiment, clean up) so the individual scripts only need
# to declare their SBATCH directives and the (EXP_NAME, CONFIG, MODELS)
# triple.
#
# Usage from a caller script:
#
#   #!/bin/bash
#   #SBATCH ...
#   source "$(dirname "$0")/_run_experiment_common.sh"
#   run_gao_experiment \
#       "01_qwen3.5_default" \
#       "configs/experiments/01_qwen3.5_default.yaml" \
#       qwen3.5:122b qwen3.5:9b qwen3.5:4b qwen3.5:2b
# ============================================================================

# Apptainer + project layout on Mahti
PROJECT_DIR="${PROJECT_DIR:-/scratch/project_2013898/ollama_env}"
OLLAMA_SIF="${OLLAMA_SIF:-${PROJECT_DIR}/ollama.sif}"
REPO_DIR="${REPO_DIR:-${PROJECT_DIR}/Green-Agent-Orchestrator}"

# Host directory where Ollama stores blobs/manifests (writable scratch).
# Inside the container we re-point OLLAMA_MODELS at /root/.ollama (the
# bind-mount target) because Apptainer otherwise leaks the host path
# verbatim into the container, which is not a valid in-container path.
export OLLAMA_MODELS="${PROJECT_DIR}/models"

# Mahti's Apptainer policy blocks APPTAINERENV_HOME, so we use the
# supported `--home src:dst` flag and back HOME with a writable scratch
# directory.
CONTAINER_HOME="${CONTAINER_HOME:-${PROJECT_DIR}/container_home}"

_gao_load_modules() {
    # Apptainer is now part of the base Mahti image; the legacy module is
    # gone. Try to load anyway in case it reappears, but only fail if the
    # binary itself is missing.
    module load apptainer 2>/dev/null || true
    module load python-data 2>/dev/null || true

    if ! command -v apptainer >/dev/null 2>&1; then
        echo "ERROR: apptainer not found on PATH. Check Mahti environment." >&2
        return 1
    fi
}

_gao_print_header() {
    local exp_name="$1"
    local config="$2"
    shift 2
    local models=("$@")

    echo "================================================================"
    echo "  GAO experiment ${exp_name}"
    echo "  array task : ${SLURM_ARRAY_TASK_ID:-n/a}"
    echo "  job id     : ${SLURM_JOB_ID:-local}"
    echo "  node       : ${SLURMD_NODENAME:-unknown}"
    echo "  config     : ${config}"
    echo "  models     : ${models[*]}"
    echo "  started    : $(date)"
    echo "================================================================"
}

_gao_start_ollama() {
    echo "Starting Ollama server…"
    apptainer run --nv \
        --home "${CONTAINER_HOME}:/root" \
        --env OLLAMA_MODELS=/root/.ollama \
        --bind "${OLLAMA_MODELS}:/root/.ollama" \
        "${OLLAMA_SIF}" serve &
    OLLAMA_PID=$!

    echo "Waiting for Ollama API…"
    local up=0
    for _ in {1..60}; do
        if curl -fs http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "Ollama API is up."
            up=1
            break
        fi
        sleep 2
    done
    if [[ "${up}" -ne 1 ]]; then
        echo "ERROR: Ollama API never came up after 120s." >&2
        return 1
    fi
}

_gao_stop_ollama() {
    if [[ -n "${OLLAMA_PID:-}" ]]; then
        echo "Cleaning up Ollama server (pid=${OLLAMA_PID})…"
        kill "${OLLAMA_PID}" 2>/dev/null || true
        wait "${OLLAMA_PID}" 2>/dev/null || true
    fi
}

_gao_pull_models() {
    local exp_name="$1"
    shift
    local models=("$@")

    echo "Pulling models for ${exp_name}…"
    for m in "${models[@]}"; do
        echo "  → ollama pull ${m}"
        apptainer exec --nv \
            --home "${CONTAINER_HOME}:/root" \
            --env OLLAMA_MODELS=/root/.ollama \
            --bind "${OLLAMA_MODELS}:/root/.ollama" \
            "${OLLAMA_SIF}" ollama pull "${m}"
    done
}

# Public entry point. Args:
#   $1       : experiment short name (used only for logging)
#   $2       : path (relative to REPO_DIR) of the YAML config
#   $3..$N   : one ollama tag per model to pre-pull
run_gao_experiment() {
    local exp_name="$1"
    local config="$2"
    shift 2
    local models=("$@")

    _gao_print_header "${exp_name}" "${config}" "${models[@]}"

    _gao_load_modules

    mkdir -p "${OLLAMA_MODELS}" "${CONTAINER_HOME}"
    cd "${REPO_DIR}"
    mkdir -p slurm/logs

    _gao_start_ollama
    trap _gao_stop_ollama EXIT

    _gao_pull_models "${exp_name}" "${models[@]}"

    source .venv/bin/activate

    echo "Running experiment with --config ${config}…"
    python -m src.run_experiment --config "${config}"

    echo "================================================================"
    echo "  Experiment ${exp_name} finished at $(date)"
    echo "  Results in: ${REPO_DIR}/results/${exp_name}/"
    echo "================================================================"
}

#!/bin/bash
# ============================================================================
# One-time helper: Clean up unused models and caches on Mahti
# ============================================================================
# Run this on a Mahti login node to free up space.
# The 1 TB quota was exceeded because large models (122B, 128B) were pulled 
# while the old ones (24B, 31B, etc. from earlier configs) were still saved.
# ============================================================================

set -euo pipefail

PROJECT_DIR="/scratch/project_2013898/ollama_env"
export OLLAMA_MODELS="${PROJECT_DIR}/models"
CONTAINER_HOME="${PROJECT_DIR}/container_home"

echo "=========================================================="
echo "1. Cleaning up Apptainer caches"
echo "=========================================================="
rm -rf "${PROJECT_DIR}/.apptainer_cache/"* 2>/dev/null || true
rm -rf "${PROJECT_DIR}/.apptainer_tmp/"* 2>/dev/null || true
echo "✓ Apptainer caches cleared."
echo

echo "=========================================================="
echo "2. Deleting old GGUF files no longer used in Exp 04"
echo "=========================================================="
OLD_GGUF="${PROJECT_DIR}/gguf/qwen3.5/qwen3.5-27b-instruct-q4_k_m.gguf"
if [[ -f "$OLD_GGUF" ]]; then
    rm -f "$OLD_GGUF"
    echo "✓ Deleted $OLD_GGUF"
else
    echo "✓ No old GGUFs found."
fi
echo

echo "=========================================================="
echo "3. Removing old Ollama models"
echo "=========================================================="
echo "Starting Ollama server temporarily..."
apptainer run \
    --home "${CONTAINER_HOME}:/root" \
    --env OLLAMA_MODELS=/root/.ollama \
    --bind "${OLLAMA_MODELS}:/root/.ollama" \
    "${PROJECT_DIR}/ollama.sif" serve > /dev/null 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to boot
sleep 5

MODELS_TO_REMOVE=(
    "mistral-small:24b"
    "mistral-large:latest"
    "gemma4:31b"
    "gemma4:26b"
    "gemma4:e4b"
    "gemma4:e2b"
    "qwen3.5:27b-q4_K_M"
)

for model in "${MODELS_TO_REMOVE[@]}"; do
    echo "  → ollama rm $model"
    apptainer exec \
        --home "${CONTAINER_HOME}:/root" \
        --env OLLAMA_MODELS=/root/.ollama \
        --bind "${OLLAMA_MODELS}:/root/.ollama" \
        "${PROJECT_DIR}/ollama.sif" ollama rm "$model" 2>/dev/null || true
done

echo "Stopping Ollama server..."
kill $OLLAMA_PID 2>/dev/null || true
wait $OLLAMA_PID 2>/dev/null || true
echo "✓ Old Ollama models removed."
echo

echo "=========================================================="
echo "Current disk usage in ${PROJECT_DIR}:"
echo "=========================================================="
du -sh "${PROJECT_DIR}/"* | sort -h

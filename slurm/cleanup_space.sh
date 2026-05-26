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
echo "2. Deleting GGUF files (Exp 04)"
echo "=========================================================="
# Delete all GGUF models to free up space (~70GB for 122B alone)
rm -f "${PROJECT_DIR}/gguf/qwen3.5/"*.gguf 2>/dev/null || true
echo "✓ Cleared GGUF directory."
echo

echo "=========================================================="
echo "3. Removing Ollama models"
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

# Comprehensive list of all models used in the experiments (old and new).
# Running this script will completely reset your Ollama model storage.
MODELS_TO_REMOVE=(
    # --- Old / obsolete models ---
    "mistral-small:24b"
    "mistral-large:latest"
    "gemma4:31b"
    "gemma4:26b"
    "gemma4:e4b"
    "gemma4:e2b"
    "qwen3.5:27b-q4_K_M"
    "gemma3:27b"
    "gemma3:4b"
    "gemma3:1b"
    "gemma3:270m"

    # --- Qwen 3.5 family (Exp 01, 02, 03) ---
    "qwen3.5:122b"
    "qwen3.5:35b"
    "qwen3.5:27b"
    "qwen3.5:9b"
    "qwen3.5:4b"
    "qwen3.5:2b"

    # --- Mistral family (Exp 05) ---
    "mistral-medium-3.5:128b"
    "ministral-3:14b"
    "ministral-3:8b"
    "ministral-3:3b"

    # --- DeepSeek R1 family (Exp 06) ---
    "deepseek-r1:70b"
    "deepseek-r1:8b"
    "deepseek-r1:7b"
    "deepseek-r1:1.5b"
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
du -sh "${PROJECT_DIR}"/* | sort -h
echo
echo "=========================================================="
echo "Overall project quota usage:"
echo "=========================================================="
csc-workspaces


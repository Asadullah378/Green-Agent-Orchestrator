# SLURM scripts for Mahti (CSC)

This folder contains everything needed to run the six paper experiments
on the [Mahti](https://docs.csc.fi/computing/systems-mahti/) GPU cluster
at CSC, using project `project_2013898`.

## What runs where

| Experiment | Backend | SLURM script | Submit command |
|---|---|---|---|
| 01–03, 05–06 | Ollama (inside Apptainer) | `run_experiments.sh` | `sbatch slurm/run_experiments.sh` |
| 04 | llama.cpp + llama-swap (inside Apptainer) | `run_experiment_04_llamacpp.sh` | `sbatch slurm/run_experiment_04_llamacpp.sh` |

Experiments 1, 2, 3, 5, 6 are submitted as a single SLURM **job array**
(`#SBATCH --array=1,2,3,5,6`), so each experiment gets its own job,
runs in parallel when GPUs are available, and is allotted its own
12-hour wall clock without blocking the others. Experiment 04 needs a
different inference stack and is therefore submitted as a separate job.

## Output layout

Every job writes its results to a dedicated subdirectory under `results/`:

```
results/01_qwen3.5_default/
results/02_qwen3.5_homo_9b/
results/03_qwen3.5_homo_4b/
results/04_qwen3.5_llamacpp/
results/05_mistral/
results/06_gemma4/
```

Each subdirectory contains the raw JSON, the per-task CSV, all generated
figures (`figures/`), all LaTeX tables (`tables/`), and aggregate CSV
exports (`csv/`).

SLURM stdout/stderr land under `slurm/logs/` (auto-created).

## Submitting

```bash
# All five Ollama experiments (job array)
sbatch slurm/run_experiments.sh

# Only experiment 06 (re-run a single one):
sbatch --array=6 slurm/run_experiments.sh

# Experiment 04 (llama.cpp), after the one-time GGUF + image setup:
sbatch slurm/run_experiment_04_llamacpp.sh
```

## One-time setup (already in place, listed for reference)

Everything lives under `/scratch/project_2013898/ollama_env/`:

```
/scratch/project_2013898/ollama_env/
├── Green-Agent-Orchestrator/       # this repository (with .venv inside)
├── models/                         # Ollama model store (bind-mounted into ollama.sif)
├── ollama.sif                      # Apptainer image for Ollama (experiments 1-3, 5, 6)
├── llamacpp.sif                    # Apptainer image for llama.cpp + llama-swap (experiment 4)
├── gguf/qwen3.5/                   # GGUF checkpoints for experiment 4
└── run_ollama_server.sh            # legacy single-experiment launcher
```

Required steps if you are starting from a fresh project directory:

1. Build / pull the Ollama Apptainer image at
   `/scratch/project_2013898/ollama_env/ollama.sif`.
2. Clone the repository at
   `/scratch/project_2013898/ollama_env/Green-Agent-Orchestrator` and
   create the Python venv inside it (`python -m venv .venv && pip install
   -r requirements.txt`).
3. For experiment 04, additionally build a llama.cpp + llama-swap
   Apptainer image at
   `/scratch/project_2013898/ollama_env/llamacpp.sif` and download the
   Qwen 3.5 GGUF checkpoints into
   `/scratch/project_2013898/ollama_env/gguf/qwen3.5/`. Then edit
   `configs/experiments/04_qwen3.5_llamacpp.swap.yaml` so every
   `--model` path points at the Mahti GGUF location instead of the
   macOS dev path. (The repo copy uses macOS paths because it is also
   used locally.)

## Resource sizing notes

Both scripts request a **single A100** on the `gpusmall` partition, which
allows 1–2 GPUs and up to 36 h of wall time. `gpumedium` cannot be used
because Mahti requires it to be claimed at full 4-GPU width.

| Experiment | Largest model loaded | Peak GPU memory | Notes |
|---|---|---|---|
| 01 | qwen3.5:27b q4 | ~17 GB | Comfortably fits on one A100. |
| 02 | qwen3.5:9b | ~7 GB | |
| 03 | qwen3.5:9b (worker) | ~7 GB | Hetero pool still includes 9B. |
| 04 | qwen3.5-27b GGUF q4 | ~17 GB | Same as exp 01, but via llama.cpp. |
| 05 | mistral-large:latest | ~70 GB | Tight on one A100 80 GB. Consider `--gres=gpu:a100:2` (still `gpusmall`). |
| 06 | gemma4:31b | ~20 GB | Fits on one A100. |

If experiment 05 OOMs on one A100, edit the `#SBATCH --gres=` line to
`gpu:a100:2` (the `gpusmall` partition still allows two GPUs).

## Re-running just the analysis

If a job finished but you want to regenerate figures and tables after
changing `analyze_results.py`, you can do so on a login node without a
GPU:

```bash
python -m src.analyze_results results/06_gemma4/results_06_gemma4_<timestamp>.json
```

Figures and tables will be written back into the same directory.

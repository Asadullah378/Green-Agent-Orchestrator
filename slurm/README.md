# SLURM scripts for Mahti (CSC)

This folder contains everything needed to run the six paper experiments
on the [Mahti](https://docs.csc.fi/computing/systems-mahti/) GPU cluster
at CSC, using project `project_2013898`.

## What runs where

| Experiment | Backend | Array slot | Standalone script |
|---|---|---|---|
| 01_qwen3.5_default | Ollama | `--array=1` | `run_experiment_01_qwen3.5_default.sh` |
| 02_qwen3.5_homo_9b | Ollama | `--array=2` | `run_experiment_02_qwen3.5_homo_9b.sh` |
| 03_qwen3.5_homo_4b | Ollama | `--array=3` | `run_experiment_03_qwen3.5_homo_4b.sh` |
| 04_qwen3.5_llamacpp | llama.cpp + llama-swap | — | `run_experiment_04_llamacpp.sh` |
| 05_mistral | Ollama | `--array=5` | `run_experiment_05_mistral.sh` |
| 06_gemma4 | Ollama | `--array=6` | `run_experiment_06_gemma4.sh` |

Experiments 1, 2, 3, 5, 6 can be submitted together as a single SLURM
**job array** (`run_experiments.sh`, `#SBATCH --array=1,2,3,5,6`) so they
run in parallel when GPUs are available, each in its own 12-hour wall
clock. Each experiment also has a **standalone wrapper** so you can
re-run a single one without touching the others.

All scripts share a single helper, `_run_experiment_common.sh`, which
encapsulates the Apptainer + Ollama lifecycle. Editing that one file
changes the behaviour of every launcher in this directory.

## Submitting

```bash
# All five Ollama experiments (job array)
sbatch slurm/run_experiments.sh

# Re-run one experiment via the array
sbatch --array=6 slurm/run_experiments.sh

# Re-run one experiment via its standalone wrapper
sbatch slurm/run_experiment_06_gemma4.sh

# llama.cpp experiment (separate stack)
sbatch slurm/run_experiment_04_llamacpp.sh
```

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
exports (`csv/`). Filenames are timestamped (`results_<exp>_<YYYYMMDD_HHMMSS>.json`)
so re-runs never overwrite earlier ones.

SLURM stdout/stderr land under `slurm/logs/` (auto-created):

- Array runs: `exp<N>_<arrayjobid>.{out,err}`
- Standalone runs: `exp0N_<jobid>.{out,err}`

## One-time setup (already in place, listed for reference)

Everything lives under `/scratch/project_2013898/ollama_env/`:

```
/scratch/project_2013898/ollama_env/
├── Green-Agent-Orchestrator/       # this repository (with .venv inside)
├── models/                         # Ollama model store (bind-mounted into ollama.sif)
├── container_home/                 # writable HOME for the Apptainer container
├── ollama.sif                      # Apptainer image for Ollama (experiments 1-3, 5, 6)
├── llamacpp.sif                    # Apptainer image for llama.cpp `llama-server` (experiment 4)
├── bin/llama-swap                  # statically-linked llama-swap Go binary (experiment 4)
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
3. For experiment 04, two one-time setup steps are needed, both run on
   a **Mahti login node** (compute nodes have no outbound internet):

   ```bash
   cd /scratch/project_2013898/ollama_env/Green-Agent-Orchestrator

   # a) Pull the CUDA-enabled llama.cpp Apptainer image and the
   #    llama-swap Go binary into ${PROJECT_DIR}.
   bash slurm/setup_llamacpp_env.sh

   # b) Download the four Q4_K_M Qwen 3.5 GGUFs into
   #    ${PROJECT_DIR}/gguf/qwen3.5/.
   bash slurm/download_qwen3.5_ggufs.sh
   ```

   Both helpers are idempotent — existing files are skipped, so
   re-running is cheap. `setup_llamacpp_env.sh` accepts env overrides
   for the image (`LLAMACPP_IMAGE`) and llama-swap version
   (`LLAMA_SWAP_VERSION`) if you ever need to bump them. No manual
   edit of `configs/experiments/04_qwen3.5_llamacpp.swap.yaml` is
   required — the SLURM script auto-rewrites the macOS dev paths to
   the Mahti scratch paths on the fly.

## Resource sizing notes

All scripts target the `gpusmall` partition (1–2 A100-40GB, up to 36 h
of wall time). `gpumedium` cannot be used because Mahti requires it to
be claimed at full 4-GPU width.

| Experiment | Largest model loaded | Approx GPU memory | Notes |
|---|---|---|---|
| 01 | qwen3.5:27b q4 | ~17 GB | Comfortably fits on one A100. |
| 02 | qwen3.5:9b | ~7 GB | |
| 03 | qwen3.5:9b (worker) | ~7 GB | Hetero pool still includes 9B. |
| 04 | qwen3.5-27b GGUF q4 | ~17 GB | Same as exp 01, but via llama.cpp. |
| 05 | mistral-small:24b | ~14 GB | Fits on one A100. |
| 06 | gemma4:31b | ~20 GB | Fits on one A100. |

## Ollama model names

A couple of Ollama Hub names that have caught us out — use these exact
tags when adding new experiments:

| Family | Wrong name | Correct Ollama tag |
|---|---|---|
| Ministral (3B / 8B / 14B) | `ministral:3b` | `ministral-3:3b` |
| Mistral Small 3 (24B) | — | `mistral-small:24b` |
| Gemma 4 edge models | `gemma4:e2b` | (correct, kept as-is) |

## Re-running just the analysis

If a job finished but you want to regenerate figures and tables after
changing `analyze_results.py`, you can do so on a login node without a
GPU:

```bash
python -m src.analyze_results results/06_gemma4/results_06_gemma4_<timestamp>.json
```

Figures and tables are written back into the same directory.

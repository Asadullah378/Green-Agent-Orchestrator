# Green Agent Orchestrator (GAO)

**Energy-Efficient Orchestration in Heterogeneous Agentic Workflows via Small Language Models**

This repository contains the implementation and experiment harness for a research project comparing two approaches to agentic AI workflows:

1. **Homogeneous flow** — a single 27B-parameter model handles all steps (planning, tool calling, synthesis).
2. **Heterogeneous flow (GAO)** — a small 4B orchestrator decomposes the task into subtasks and routes each to the smallest capable model from a pool of 2B, 4B, and 9B models.

The goal is to measure whether heterogeneous orchestration with small language models can reduce energy consumption without sacrificing task accuracy.

## Key Results

| Metric | Homogeneous (27B) | Heterogeneous (GAO) | Improvement |
|---|---|---|---|
| Energy (mWh/task) | 0.539 | 0.135 | **75.0% less** |
| Duration (s/task) | 125.8 | 33.4 | **73.4% faster** |
| Accuracy | 0.96 | 1.00 | **+4%** |

> Evaluated on 15 benchmark tasks across 3 difficulty tiers, with 3 runs each (90 total runs). All models run locally via Ollama on Apple Silicon.

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| [Ollama](https://ollama.com) | latest |
| OS | macOS (Apple Silicon) or Linux |
| RAM | 16 GB minimum, 32 GB+ recommended |

### Pull the required models

```bash
ollama pull qwen3.5:27b-q4_K_M
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
ollama pull qwen3.5:2b
```

Make sure Ollama is running before starting experiments:

```bash
ollama serve
```

The experiment runner can also use `llama.cpp` as the inference backend
instead of Ollama. See the **Configuration → llama.cpp backend** section
below for the one-time setup.

## Installation

```bash
git clone https://github.com/<your-username>/green-agent-orchestrator.git
cd green-agent-orchestrator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the full experiment

Runs both flows across all 15 tasks with 3 repetitions (90 total runs). The execution alternates between homogeneous and heterogeneous flows per task to control for thermal and caching effects.

```bash
python -m src.run_experiment
```

This loads `configs/default.yaml` (the Qwen3.5 paper baseline). To run with a different model family, pass `--config`:

```bash
python -m src.run_experiment --config configs/llama3.yaml
python -m src.run_experiment --config configs/mistral.yaml
python -m src.run_experiment --config configs/my_custom.yaml
```

You can also set `GAO_CONFIG` in the environment to make every command use a non-default config:

```bash
export GAO_CONFIG=configs/llama3.yaml
python -m src.run_experiment
```

### Run with verbose agent logs

```bash
python -m src.run_experiment -v
```

### Selective runs

```bash
# Only the homogeneous baseline
python -m src.run_experiment --flow homogeneous

# Only the heterogeneous flow
python -m src.run_experiment --flow heterogeneous

# Specific tasks with a single run
python -m src.run_experiment --tasks E1 M1 H1 --runs 1

# Only easy-difficulty tasks
python -m src.run_experiment --difficulty easy

# Skip automatic analysis after the experiment
python -m src.run_experiment --no-analyze
```

### Analyse results

After a run, result files are saved under `results/`. You can generate figures and LaTeX tables from any results JSON:

```bash
python -m src.analyze_results results/<your-results-file>.json
```

This produces publication-quality figures in `results/figures/` and LaTeX tables in `results/tables/`.

### Merge multiple result files

If you ran experiments in separate batches (e.g., by difficulty), merge them into a single dataset:

```bash
python -m src.merge_results results/results_A.json results/results_B.json --analyze
```

## Project Structure

```
├── configs/
│   └── experiments/              # six paper experiments (see README "Configuration")
│       ├── 01_qwen3.5_default.yaml
│       ├── 02_qwen3.5_homo_9b.yaml
│       ├── 03_qwen3.5_homo_4b.yaml
│       ├── 04_qwen3.5_llamacpp.yaml
│       ├── 04_qwen3.5_llamacpp.swap.yaml   # llama-swap proxy config for exp 04
│       ├── 05_mistral.yaml
│       └── 06_gemma4.yaml
├── src/
│   ├── config.py                 # YAML config loader (RunConfig + legacy constants)
│   ├── models.py                 # ChatOllama model factory (cached instances)
│   ├── tools.py                  # 5 deterministic agent tools (no external APIs)
│   ├── tracking.py               # CodeCarbon energy tracking + timing wrapper
│   ├── run_experiment.py         # Main experiment runner (accepts --config PATH)
│   ├── analyze_results.py        # Figure and table generation (model-family-agnostic)
│   ├── merge_results.py          # Utility to combine multiple result files
│   ├── agents/
│   │   ├── homogeneous.py        # Flow 1 — single-model ReAct agent
│   │   └── heterogeneous.py      # Flow 2 — GAO orchestrator + workers + synthesiser
│   └── benchmark/
│       ├── tasks.py              # 15 benchmark tasks (5 easy, 5 medium, 5 hard)
│       └── evaluators.py         # Deterministic accuracy scoring
├── results/                      # Generated results, figures, tables (gitignored)
├── requirements.txt
└── README.md
```

## Architecture

### Homogeneous baseline

A standard LangGraph `create_react_agent` using `qwen3.5:27b-q4_K_M` for all steps.

### Heterogeneous flow (GAO)

A custom LangGraph `StateGraph` with three phases:

1. **Orchestrate** — the 4B model decomposes the user query into 1–4 self-contained subtasks, estimates difficulty, and assigns the smallest capable worker model.
2. **Execute** — each subtask runs as its own ReAct agent with the assigned model (2B, 4B, or 9B) and access to the tool suite.
3. **Synthesise** — a 2B model combines subtask results into the final answer. Skipped for single-subtask plans.

### Tools

All tools are deterministic with hardcoded data to ensure reproducible experiments:

| Tool | Description |
|---|---|
| `calculator` | Safe math expression evaluator (AST-based) |
| `unit_converter` | Converts between common units (length, weight, temperature, currency) |
| `data_lookup` | Retrieves financial and demographic data from a built-in dataset |
| `date_calculator` | Date arithmetic (days between, add/subtract days) |
| `text_processor` | Word count, character count, sentence count |

### Metrics

| Metric | How measured |
|---|---|
| Energy (kWh) | CodeCarbon `OfflineEmissionsTracker` — CPU + GPU + RAM power draw |
| CO₂ emissions (kg) | CodeCarbon estimate using regional grid carbon intensity |
| Accuracy | Deterministic evaluators with expected-value matching (5% numeric tolerance) |
| Duration (s) | `time.perf_counter()` wall-clock time |
| Energy-to-Solution | `energy / accuracy` — penalises incorrect results |

## Configuration

The entire experiment is driven by a single YAML config file in `configs/`.
Nothing in the code is tied to a specific model family; the orchestrator
prompt, worker pool, routing heuristic, and step limits are all built at
runtime from the active config.

All experiment configs live in `configs/experiments/`. Each one writes its
output to a dedicated subdirectory under `results/` so the six experiments
never clobber each other and each can be analysed independently.

| # | Config file | Backend | Homogeneous | Heterogeneous workers |
|---|---|---|---|---|
| 01 | `configs/experiments/01_qwen3.5_default.yaml` | Ollama    | `qwen3.5:27b-q4_K_M` | Qwen 3.5 (2B / 4B / 9B) |
| 02 | `configs/experiments/02_qwen3.5_homo_9b.yaml` | Ollama    | `qwen3.5:9b`         | Qwen 3.5 (2B / 4B / 9B) |
| 03 | `configs/experiments/03_qwen3.5_homo_4b.yaml` | Ollama    | `qwen3.5:4b`         | Qwen 3.5 (2B / 4B / 9B) |
| 04 | `configs/experiments/04_qwen3.5_llamacpp.yaml` | llama.cpp | `qwen3.5-27b`       | Qwen 3.5 (2B / 4B / 9B) |
| 05 | `configs/experiments/05_mistral.yaml`         | Ollama    | `mistral-large:latest` (Large 3) | Ministral 3 (3B / 8B / 14B) |
| 06 | `configs/experiments/06_gemma4.yaml`          | Ollama    | `gemma4:31b`         | Gemma 4 (E2B / E4B / 26B) |

Each experiment runs 7 repetitions per (task, flow) — 15 tasks × 2 flows ×
7 runs = 210 records per experiment.

Output layout for one experiment:

```
results/
└── 01_qwen3.5_default/
    ├── results_01_qwen3.5_default_<timestamp>.json
    ├── results_01_qwen3.5_default_<timestamp>.csv
    ├── figures/         (PNGs at 300 DPI)
    ├── tables/          (LaTeX \input{} fragments)
    └── csv/             (per-task / per-difficulty / overall statistics)
```

To run any experiment:

```bash
python -m src.run_experiment --config configs/experiments/01_qwen3.5_default.yaml
python -m src.run_experiment --config configs/experiments/06_gemma4.yaml
# ...
```

When run with no `--config` flag, the runner falls back to experiment 01
(the paper baseline).

### llama.cpp backend (experiment 04)

Experiment 04 runs the same Qwen 3.5 setup as experiment 01, but the
inference backend is `llama.cpp` (accessed through
[`llama-swap`](https://github.com/mostlygeek/llama-swap), a small
OpenAI-compatible proxy that hot-swaps the underlying `.gguf` model on
demand). This isolates the effect of the runtime/backend from the
model-size effect.

One-time setup (Apple Silicon / macOS):

```bash
# 1. Install llama.cpp and llama-swap
brew install llama.cpp
brew install llama-swap

# 2. Download Qwen 3.5 GGUF checkpoints to ~/models/qwen3.5/
#    (paths used in configs/experiments/04_qwen3.5_llamacpp.swap.yaml)

# 3. Start the proxy in one terminal
llama-swap --config configs/experiments/04_qwen3.5_llamacpp.swap.yaml --listen :8080

# 4. In another terminal, run experiment 04
python -m src.run_experiment --config configs/experiments/04_qwen3.5_llamacpp.yaml
```

The proxy auto-loads each model the first time it is requested, keeps it
warm for a configurable TTL, and unloads it when memory is needed for
another model. The experiment harness is unaware of any of this; from
its perspective, every model lives behind one OpenAI-compatible URL.

### Creating a new config

Copy `configs/default.yaml` and edit the sections you need. The most
common knobs are:

```yaml
experiment:
  name: "my_experiment"       # appended to result filenames
  num_runs: 3                 # repetitions per (task, flow)

llm:
  provider: "ollama"          # "ollama" | "llamacpp"
  temperature: 0.0
  reasoning: false            # disable thinking mode for Qwen-family models
  ollama:
    base_url: "http://localhost:11434"
  llamacpp:
    base_url: "http://localhost:8080/v1"
    api_key: "no-key"

energy:
  country_iso_code: "FIN"     # CodeCarbon ISO-3 country code

homogeneous:
  model: "qwen3.5:27b-q4_K_M"
  size_b: 27
  max_agent_steps: 40

heterogeneous:
  orchestrator:
    model: "qwen3.5:4b"
    size_b: 4
  synthesizer:
    model: "qwen3.5:2b"
    size_b: 2
  workers:                    # add/remove tiers freely
    - tier: "small"
      model: "qwen3.5:2b"
      size_b: 2
      max_steps: 10
      description: "Simple subtasks — 1 to 2 total tool invocations."
    - tier: "medium"
      model: "qwen3.5:4b"
      size_b: 4
      max_steps: 15
      description: "Moderate subtasks — 3 to 5 tool invocations or multi-step math."
    - tier: "large"
      model: "qwen3.5:9b"
      size_b: 9
      max_steps: 25
      description: "Complex subtasks — 6+ tool invocations or advanced reasoning."
  difficulty_to_tier:
    easy:   "small"
    medium: "medium"
    hard:   "large"
  routing_safety:
    enabled: true              # auto-upgrade misrouted complex subtasks
    from_tier: "small"
    to_tier:   "medium"
```

Every result JSON file embeds a copy of the config it was produced with
under `metadata.config`, so figures and tables produced by
`src/analyze_results.py` automatically adapt their labels and color
palette to the model family used.

## Citation

If you use this code in your research, please cite:

```
Warraich, A. N. (2026). Energy-Efficient Orchestration in Heterogeneous
Agentic Workflows via Small Language Models. University of Helsinki.
```

## License

This project is released for academic and research purposes.

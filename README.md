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
├── src/
│   ├── config.py                 # Model pool, Ollama settings, experiment parameters
│   ├── models.py                 # ChatOllama model factory (cached instances)
│   ├── tools.py                  # 5 deterministic agent tools (no external APIs)
│   ├── tracking.py               # CodeCarbon energy tracking + timing wrapper
│   ├── run_experiment.py         # Main experiment runner
│   ├── analyze_results.py        # Figure and table generation
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

Edit `src/config.py` to customise:

- **Model pool** — add or swap models (must be available in Ollama)
- **Orchestrator and worker assignments** — change `ORCHESTRATOR_MODEL`, `DIFFICULTY_MODEL_MAP`
- **Repetitions** — `NUM_RUNS` (default: 3)
- **LLM settings** — `LLM_TEMPERATURE` (default: 0.0), `LLM_REQUEST_TIMEOUT`
- **CodeCarbon** — `COUNTRY_ISO_CODE` (default: `"FIN"` for Finland)

## Citation

If you use this code in your research, please cite:

```
Warraich, A. N. (2026). Energy-Efficient Orchestration in Heterogeneous
Agentic Workflows via Small Language Models. University of Helsinki.
```

## License

This project is released for academic and research purposes.

"""
Green Agent Orchestrator (GAO) — Configuration

Model pool, Ollama settings, and experiment parameters.
"""

OLLAMA_BASE_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Model pool — every model used across both flows
# ---------------------------------------------------------------------------
MODEL_POOL = {
    "qwen3.5:2b":           {"size_b": 2,  "tier": "XS"},
    "qwen3.5:4b":           {"size_b": 4,  "tier": "S"},
    "qwen3.5:9b":           {"size_b": 9,  "tier": "M"},
    "qwen3.5:27b-q4_K_M":  {"size_b": 27, "tier": "XL"},
}

# ---------------------------------------------------------------------------
# Flow 1 — Homogeneous: one large model for everything
# ---------------------------------------------------------------------------
HOMOGENEOUS_MODEL = "qwen3.5:27b-q4_K_M"

# ---------------------------------------------------------------------------
# Flow 2 — Heterogeneous: small orchestrator + smaller workers
# ---------------------------------------------------------------------------
ORCHESTRATOR_MODEL = "qwen3.5:4b"

HETEROGENEOUS_POOL = {
    "qwen3.5:2b":  {"size_b": 2,  "tier": "XS"},
    "qwen3.5:4b":  {"size_b": 4,  "tier": "S"},
    "qwen3.5:9b":  {"size_b": 9,  "tier": "M"},
}

DIFFICULTY_MODEL_MAP = {
    "easy":   "qwen3.5:2b",
    "medium": "qwen3.5:4b",
    "hard":   "qwen3.5:9b",
}

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
NUM_RUNS = 3                # repetitions per task for variance
LLM_TEMPERATURE = 0.0       # deterministic by default
LLM_REQUEST_TIMEOUT = 120   # seconds per LLM call
MAX_AGENT_STEPS = 40        # safety limit on agent loop iterations

# CodeCarbon
CODECARBON_LOG_LEVEL = "error"
COUNTRY_ISO_CODE = "FIN"    # Finland — adjust to your location

# Output
RESULTS_DIR = "results"

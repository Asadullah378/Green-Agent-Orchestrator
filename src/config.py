"""
Green Agent Orchestrator (GAO) — Configuration

The entire experiment (model pool, prompts, routing heuristic, energy
tracker, number of runs, etc.) is driven by a single YAML file.

Lookup order for the config path:

    1. The value passed to `load_config(path)` (used by the CLI).
    2. The environment variable `GAO_CONFIG`.
    3. The default path `configs/default.yaml`.

This module exposes both a typed `CONFIG` object (preferred for new code)
and the legacy module-level constants used by the rest of the codebase
(`HOMOGENEOUS_MODEL`, `ORCHESTRATOR_MODEL`, `MODEL_POOL`, ...).
The legacy constants are populated lazily from the loaded YAML so existing
imports keep working without changes.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "configs/experiments/01_qwen3.5_default.yaml"
_CONFIG_LOCK = threading.Lock()
_CONFIG_CACHE: "RunConfig | None" = None
_LOADED_PATH: str | None = None


# ── Typed config dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class ExperimentSection:
    name: str = "default"
    num_runs: int = 3
    results_dir: str = "results"


@dataclass(frozen=True)
class OllamaProvider:
    base_url: str = "http://localhost:11434"


@dataclass(frozen=True)
class LlamaCppProvider:
    """llama.cpp provider settings.

    Assumes the user is running an OpenAI-compatible endpoint such as
    `llama-swap` (recommended) or `llama-server` directly. The `base_url`
    must include the API root (typically ending in `/v1`).
    """
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "no-key"


@dataclass(frozen=True)
class LLMSection:
    provider: str = "ollama"           # "ollama" | "llamacpp"
    request_timeout: int = 120
    temperature: float = 0.0
    reasoning: bool = False             # honoured by ollama only
    ollama: OllamaProvider = field(default_factory=OllamaProvider)
    llamacpp: LlamaCppProvider = field(default_factory=LlamaCppProvider)

    @property
    def base_url(self) -> str:
        """Active endpoint for the selected provider (backwards-compat shim)."""
        if self.provider == "llamacpp":
            return self.llamacpp.base_url
        return self.ollama.base_url


@dataclass(frozen=True)
class EnergySection:
    country_iso_code: str = "FIN"
    log_level: str = "error"
    tracking_mode: str = "process"


@dataclass(frozen=True)
class HomogeneousSection:
    model: str = ""
    size_b: int = 0
    max_agent_steps: int = 40


@dataclass(frozen=True)
class WorkerTier:
    tier: str
    model: str
    size_b: int
    max_steps: int = 15
    description: str = ""


@dataclass(frozen=True)
class OrchestratorSection:
    model: str = ""
    size_b: int = 0


@dataclass(frozen=True)
class SynthesizerSection:
    model: str = ""
    size_b: int = 0


@dataclass(frozen=True)
class RoutingSafety:
    enabled: bool = True
    from_tier: str = "small"
    to_tier: str = "medium"
    min_complexity_keywords: int = 3
    max_description_length: int = 160
    complexity_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecompositionSection:
    min_subtasks: int = 1
    max_subtasks: int = 4


@dataclass(frozen=True)
class HeterogeneousSection:
    orchestrator: OrchestratorSection
    synthesizer: SynthesizerSection
    workers: tuple[WorkerTier, ...]
    difficulty_to_tier: dict[str, str]
    routing_safety: RoutingSafety
    decomposition: DecompositionSection

    def tier(self, tier_name: str) -> WorkerTier:
        for w in self.workers:
            if w.tier == tier_name:
                return w
        raise KeyError(f"Unknown worker tier: {tier_name!r}")

    def model_for_tier(self, tier_name: str) -> str:
        return self.tier(tier_name).model

    def tier_for_model(self, model_name: str) -> str | None:
        for w in self.workers:
            if w.model == model_name:
                return w.tier
        return None

    def model_for_difficulty(self, difficulty: str) -> str:
        tier_name = self.difficulty_to_tier.get(difficulty)
        if tier_name is None:
            tier_name = self.workers[len(self.workers) // 2].tier
        return self.model_for_tier(tier_name)


@dataclass(frozen=True)
class RunConfig:
    experiment: ExperimentSection
    llm: LLMSection
    energy: EnergySection
    homogeneous: HomogeneousSection
    heterogeneous: HeterogeneousSection
    raw: dict = field(default_factory=dict, repr=False)
    source_path: str = ""

    def all_model_names(self) -> list[str]:
        names = {self.homogeneous.model,
                 self.heterogeneous.orchestrator.model,
                 self.heterogeneous.synthesizer.model}
        names.update(w.model for w in self.heterogeneous.workers)
        return sorted(n for n in names if n)

    def size_for_model(self, model_name: str) -> int:
        if model_name == self.homogeneous.model:
            return self.homogeneous.size_b
        if model_name == self.heterogeneous.orchestrator.model:
            return self.heterogeneous.orchestrator.size_b
        if model_name == self.heterogeneous.synthesizer.model:
            return self.heterogeneous.synthesizer.size_b
        for w in self.heterogeneous.workers:
            if w.model == model_name:
                return w.size_b
        return 0


# ── Loading and access ─────────────────────────────────────────────────────


def _parse_llm(llm: dict) -> LLMSection:
    """Parse the `llm:` block with backwards compatibility for the older
    flat shape (`llm.base_url` instead of `llm.ollama.base_url`).
    """
    provider = llm.get("provider", "ollama")
    if provider not in {"ollama", "llamacpp"}:
        raise ValueError(
            f"Unknown llm.provider {provider!r}. Use 'ollama' or 'llamacpp'."
        )

    legacy_base = llm.get("base_url")
    ollama_raw = llm.get("ollama") or {}
    llamacpp_raw = llm.get("llamacpp") or {}

    ollama_url = ollama_raw.get("base_url")
    if not ollama_url:
        ollama_url = legacy_base if provider == "ollama" else "http://localhost:11434"

    llamacpp_url = llamacpp_raw.get("base_url")
    if not llamacpp_url:
        llamacpp_url = legacy_base if provider == "llamacpp" else "http://localhost:8080/v1"

    return LLMSection(
        provider=provider,
        request_timeout=int(llm.get("request_timeout", 120)),
        temperature=float(llm.get("temperature", 0.0)),
        reasoning=bool(llm.get("reasoning", False)),
        ollama=OllamaProvider(base_url=ollama_url),
        llamacpp=LlamaCppProvider(
            base_url=llamacpp_url,
            api_key=llamacpp_raw.get("api_key", "no-key"),
        ),
    )


def _parse_workers(items: list[dict]) -> tuple[WorkerTier, ...]:
    workers: list[WorkerTier] = []
    seen: set[str] = set()
    for raw in items:
        tier = raw["tier"]
        if tier in seen:
            raise ValueError(f"Duplicate worker tier in config: {tier!r}")
        seen.add(tier)
        workers.append(
            WorkerTier(
                tier=tier,
                model=raw["model"],
                size_b=int(raw.get("size_b", 0)),
                max_steps=int(raw.get("max_steps", 15)),
                description=raw.get("description", ""),
            )
        )
    if not workers:
        raise ValueError("`heterogeneous.workers` must contain at least one tier.")
    return tuple(workers)


def _parse_config(data: dict, source_path: str) -> RunConfig:
    exp = data.get("experiment", {})
    llm = data.get("llm", {})
    energy = data.get("energy", {})
    homo = data.get("homogeneous", {})
    hetero = data.get("heterogeneous", {})

    if not homo.get("model"):
        raise ValueError("`homogeneous.model` must be set in the config file.")

    orch_raw = hetero.get("orchestrator", {})
    synth_raw = hetero.get("synthesizer", {})
    if not orch_raw.get("model"):
        raise ValueError("`heterogeneous.orchestrator.model` must be set.")
    if not synth_raw.get("model"):
        raise ValueError("`heterogeneous.synthesizer.model` must be set.")

    workers = _parse_workers(hetero.get("workers", []))
    diff_map = dict(hetero.get("difficulty_to_tier", {}))

    valid_tiers = {w.tier for w in workers}
    for difficulty, tier in diff_map.items():
        if tier not in valid_tiers:
            raise ValueError(
                f"difficulty_to_tier[{difficulty!r}] = {tier!r} is not a defined worker tier."
            )

    rs_raw = hetero.get("routing_safety", {})
    routing_safety = RoutingSafety(
        enabled=bool(rs_raw.get("enabled", True)),
        from_tier=rs_raw.get("from_tier", "small"),
        to_tier=rs_raw.get("to_tier", "medium"),
        min_complexity_keywords=int(rs_raw.get("min_complexity_keywords", 3)),
        max_description_length=int(rs_raw.get("max_description_length", 160)),
        complexity_keywords=tuple(rs_raw.get("complexity_keywords", [])),
    )
    if routing_safety.enabled:
        for t in (routing_safety.from_tier, routing_safety.to_tier):
            if t not in valid_tiers:
                raise ValueError(
                    f"routing_safety tier {t!r} not defined in workers list."
                )

    decomp_raw = hetero.get("decomposition", {})
    decomposition = DecompositionSection(
        min_subtasks=int(decomp_raw.get("min_subtasks", 1)),
        max_subtasks=int(decomp_raw.get("max_subtasks", 4)),
    )

    return RunConfig(
        experiment=ExperimentSection(
            name=exp.get("name", "default"),
            num_runs=int(exp.get("num_runs", 3)),
            results_dir=exp.get("results_dir", "results"),
        ),
        llm=_parse_llm(llm),
        energy=EnergySection(
            country_iso_code=energy.get("country_iso_code", "FIN"),
            log_level=energy.get("log_level", "error"),
            tracking_mode=energy.get("tracking_mode", "process"),
        ),
        homogeneous=HomogeneousSection(
            model=homo["model"],
            size_b=int(homo.get("size_b", 0)),
            max_agent_steps=int(homo.get("max_agent_steps", 40)),
        ),
        heterogeneous=HeterogeneousSection(
            orchestrator=OrchestratorSection(
                model=orch_raw["model"],
                size_b=int(orch_raw.get("size_b", 0)),
            ),
            synthesizer=SynthesizerSection(
                model=synth_raw["model"],
                size_b=int(synth_raw.get("size_b", 0)),
            ),
            workers=workers,
            difficulty_to_tier=diff_map,
            routing_safety=routing_safety,
            decomposition=decomposition,
        ),
        raw=data,
        source_path=source_path,
    )


def load_config(path: str | os.PathLike | None = None) -> RunConfig:
    """Load (or reload) the experiment configuration from a YAML file.

    Subsequent calls to `get_config()` will return the loaded instance.
    """
    global _CONFIG_CACHE, _LOADED_PATH
    chosen = (
        str(path)
        if path is not None
        else os.environ.get("GAO_CONFIG", DEFAULT_CONFIG_PATH)
    )
    p = Path(chosen)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r") as f:
        data = yaml.safe_load(f) or {}
    cfg = _parse_config(data, source_path=str(p.resolve()))
    with _CONFIG_LOCK:
        _CONFIG_CACHE = cfg
        _LOADED_PATH = str(p.resolve())
        _refresh_legacy_constants(cfg)
    return cfg


def get_config() -> RunConfig:
    """Return the currently-loaded config, loading the default if needed."""
    if _CONFIG_CACHE is None:
        load_config()
    assert _CONFIG_CACHE is not None
    return _CONFIG_CACHE


def get_loaded_path() -> str | None:
    return _LOADED_PATH


# ── Legacy module-level constants ───────────────────────────────────────────
#
# Existing code imports things like `HOMOGENEOUS_MODEL` directly from
# `src.config`. We populate these from the loaded YAML so all existing call
# sites continue to work; new code should prefer `get_config()`.

HOMOGENEOUS_MODEL: str = ""
ORCHESTRATOR_MODEL: str = ""
SYNTHESIZER_MODEL: str = ""
NUM_RUNS: int = 3
LLM_TEMPERATURE: float = 0.0
LLM_REQUEST_TIMEOUT: int = 120
MAX_AGENT_STEPS: int = 40
CODECARBON_LOG_LEVEL: str = "error"
COUNTRY_ISO_CODE: str = "FIN"
RESULTS_DIR: str = "results"
OLLAMA_BASE_URL: str = "http://localhost:11434"
LLM_REASONING: bool = False

MODEL_POOL: dict[str, dict[str, Any]] = {}
HETEROGENEOUS_POOL: dict[str, dict[str, Any]] = {}
DIFFICULTY_MODEL_MAP: dict[str, str] = {}


def _refresh_legacy_constants(cfg: RunConfig) -> None:
    """Populate the legacy module-level constants from a loaded RunConfig."""
    global HOMOGENEOUS_MODEL, ORCHESTRATOR_MODEL, SYNTHESIZER_MODEL
    global NUM_RUNS, LLM_TEMPERATURE, LLM_REQUEST_TIMEOUT, MAX_AGENT_STEPS
    global CODECARBON_LOG_LEVEL, COUNTRY_ISO_CODE, RESULTS_DIR
    global OLLAMA_BASE_URL, LLM_REASONING
    global MODEL_POOL, HETEROGENEOUS_POOL, DIFFICULTY_MODEL_MAP

    HOMOGENEOUS_MODEL = cfg.homogeneous.model
    ORCHESTRATOR_MODEL = cfg.heterogeneous.orchestrator.model
    SYNTHESIZER_MODEL = cfg.heterogeneous.synthesizer.model
    NUM_RUNS = cfg.experiment.num_runs
    LLM_TEMPERATURE = cfg.llm.temperature
    LLM_REQUEST_TIMEOUT = cfg.llm.request_timeout
    MAX_AGENT_STEPS = cfg.homogeneous.max_agent_steps
    CODECARBON_LOG_LEVEL = cfg.energy.log_level
    COUNTRY_ISO_CODE = cfg.energy.country_iso_code
    RESULTS_DIR = cfg.experiment.results_dir
    OLLAMA_BASE_URL = cfg.llm.base_url
    LLM_REASONING = cfg.llm.reasoning

    pool: dict[str, dict[str, Any]] = {}
    pool[cfg.homogeneous.model] = {
        "size_b": cfg.homogeneous.size_b,
        "tier": "baseline",
    }
    pool[cfg.heterogeneous.orchestrator.model] = {
        "size_b": cfg.heterogeneous.orchestrator.size_b,
        "tier": "orchestrator",
    }
    pool[cfg.heterogeneous.synthesizer.model] = {
        "size_b": cfg.heterogeneous.synthesizer.size_b,
        "tier": "synthesizer",
    }
    hetero_pool: dict[str, dict[str, Any]] = {}
    for w in cfg.heterogeneous.workers:
        pool[w.model] = {"size_b": w.size_b, "tier": w.tier}
        hetero_pool[w.model] = {"size_b": w.size_b, "tier": w.tier}

    MODEL_POOL = pool
    HETEROGENEOUS_POOL = hetero_pool
    DIFFICULTY_MODEL_MAP = {
        diff: cfg.heterogeneous.model_for_tier(tier_name)
        for diff, tier_name in cfg.heterogeneous.difficulty_to_tier.items()
    }


# Trigger initial load at import time so the legacy constants are populated.
load_config()

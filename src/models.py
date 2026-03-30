"""
Green Agent Orchestrator (GAO) — Model factory

Thin wrapper around langchain-ollama that returns ChatOllama instances
from the configured pool.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import ChatOllama

from src.config import (
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    MODEL_POOL,
    OLLAMA_BASE_URL,
)


@lru_cache(maxsize=16)
def get_model(model_name: str, *, temperature: float | None = None) -> ChatOllama:
    """Return a cached ChatOllama instance for *model_name*."""
    if model_name not in MODEL_POOL:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_POOL.keys())}"
        )
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        timeout=LLM_REQUEST_TIMEOUT,
        reasoning=False,
    )


def get_all_model_names() -> list[str]:
    return list(MODEL_POOL.keys())


def model_tier(model_name: str) -> str:
    return MODEL_POOL[model_name]["tier"]


def model_size_b(model_name: str) -> int:
    return MODEL_POOL[model_name]["size_b"]

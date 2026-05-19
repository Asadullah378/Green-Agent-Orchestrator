"""
Green Agent Orchestrator (GAO) — Model factory

Returns a LangChain chat model configured for the active provider
(`ollama` or `llamacpp`). The provider is selected in the active YAML
config; everything else (model name, temperature, timeout) is also
read from the config.

Both providers expose the same `BaseChatModel` interface, so all downstream
code (workers, orchestrator, synthesiser) is provider-agnostic.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import get_config


@lru_cache(maxsize=32)
def _build_ollama(model_name: str, temperature: float, timeout: int,
                  base_url: str, reasoning: bool) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
        reasoning=reasoning,
    )


@lru_cache(maxsize=32)
def _build_llamacpp(model_name: str, temperature: float, timeout: int,
                    base_url: str, api_key: str) -> ChatOpenAI:
    # llama.cpp (via llama-swap or llama-server) is OpenAI-compatible, so we
    # talk to it with ChatOpenAI. The `model` field is forwarded as-is and
    # must match the alias configured on the server side.
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
    )


def get_model(model_name: str, *, temperature: float | None = None) -> BaseChatModel:
    """Return a cached chat-model instance for *model_name* using the
    active provider configured in the YAML.
    """
    cfg = get_config()
    known = set(cfg.all_model_names())
    if model_name not in known:
        raise ValueError(
            f"Unknown model '{model_name}'. Configured models: {sorted(known)}"
        )
    t = cfg.llm.temperature if temperature is None else temperature

    if cfg.llm.provider == "ollama":
        return _build_ollama(
            model_name=model_name,
            temperature=t,
            timeout=cfg.llm.request_timeout,
            base_url=cfg.llm.ollama.base_url,
            reasoning=cfg.llm.reasoning,
        )
    if cfg.llm.provider == "llamacpp":
        return _build_llamacpp(
            model_name=model_name,
            temperature=t,
            timeout=cfg.llm.request_timeout,
            base_url=cfg.llm.llamacpp.base_url,
            api_key=cfg.llm.llamacpp.api_key,
        )
    raise ValueError(f"Unsupported llm.provider: {cfg.llm.provider!r}")


def get_all_model_names() -> list[str]:
    return get_config().all_model_names()


def model_size_b(model_name: str) -> int:
    return get_config().size_for_model(model_name)


def model_tier(model_name: str) -> str:
    """Return the tier label for *model_name* (e.g. small/medium/large/baseline)."""
    cfg = get_config()
    if model_name == cfg.homogeneous.model:
        return "baseline"
    if model_name == cfg.heterogeneous.orchestrator.model:
        return "orchestrator"
    if model_name == cfg.heterogeneous.synthesizer.model:
        return "synthesizer"
    tier = cfg.heterogeneous.tier_for_model(model_name)
    return tier or "unknown"

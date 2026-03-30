"""
Green Agent Orchestrator (GAO) — Deterministic accuracy evaluators

No LLM-as-a-judge: scoring is based on whether expected values appear
in the agent's response and whether the correct tools were invoked.
"""

from __future__ import annotations

import re


def _normalize(text: str) -> str:
    """Lowercase, strip whitespace/commas/currency signs/LaTeX formatting."""
    text = text.lower().strip()
    # Strip LaTeX thin-space (\,) and other formatting before removing commas
    text = re.sub(r"\\,", "", text)
    text = re.sub(r"\\[; ]", "", text)
    text = re.sub(r"\\text\{[^}]*\}", "", text)
    text = text.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    return text


def value_present(response: str, expected: str) -> bool:
    """Check whether *expected* appears in *response* (fuzzy-numeric)."""
    norm_resp = _normalize(response)
    norm_exp = _normalize(expected)

    if norm_exp in norm_resp:
        return True

    # Try numeric comparison with tolerance
    try:
        exp_num = float(norm_exp)
    except ValueError:
        return False

    numbers = re.findall(r"-?\d[\d,]*\.?\d*", norm_resp)
    for num_str in numbers:
        try:
            found = float(num_str.replace(",", ""))
            if abs(found - exp_num) / max(abs(exp_num), 1e-9) < 0.05:
                return True
        except ValueError:
            continue

    return False


def score_task(response: str, expected_values: list[str]) -> float:
    """Return accuracy in [0, 1] as fraction of expected values found."""
    if not expected_values:
        return 1.0
    hits = sum(value_present(response, v) for v in expected_values)
    return round(hits / len(expected_values), 4)


def evaluate_record(record) -> float:
    """Convenience: score a TaskRecord against its benchmark task."""
    from src.benchmark.tasks import get_task_by_id

    task = get_task_by_id(record.task_id)
    if task is None:
        return 0.0
    return score_task(record.response, task["expected_values"])

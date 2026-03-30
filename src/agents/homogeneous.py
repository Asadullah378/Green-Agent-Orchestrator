"""
Green Agent Orchestrator (GAO) — Flow 1: Homogeneous baseline

A standard ReAct agent that uses the same model (gpt-oss:20b) for every
step: planning, tool calling, and synthesis.
"""

from __future__ import annotations

import json
import textwrap

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from src.config import HOMOGENEOUS_MODEL, MAX_AGENT_STEPS
from src.models import get_model
from src.tools import ALL_TOOLS
from src.tracking import TaskRecord, TrackingResult, track_energy

_INDENT = "        "
_BLUE = "\033[94m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _log_messages(messages: list) -> None:
    """Pretty-print an agent message trace."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            print(f"{_INDENT}{_CYAN}[USER]{_RESET} {msg.content[:200]}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = json.dumps(tc["args"], ensure_ascii=False)
                    print(f"{_INDENT}{_YELLOW}[TOOL CALL]{_RESET} {tc['name']}({args})")
            if msg.content:
                wrapped = textwrap.shorten(msg.content, width=300, placeholder="…")
                print(f"{_INDENT}{_BLUE}[LLM]{_RESET} {wrapped}")
        elif isinstance(msg, ToolMessage):
            content = textwrap.shorten(str(msg.content), width=200, placeholder="…")
            print(f"{_INDENT}{_GREEN}[TOOL RESULT]{_RESET} {content}")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use the provided tools to answer the user's question accurately. "
    "Always use tools when a calculation, conversion, data lookup, or "
    "date/text operation is needed — do NOT guess numeric answers. "
    "After obtaining tool results, provide a clear final answer."
)


def _build_agent():
    model = get_model(HOMOGENEOUS_MODEL)
    return create_react_agent(
        model,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )


def _count_messages(messages: list) -> tuple[int, int]:
    """Return (num_llm_calls, num_tool_calls) from a message list."""
    llm_calls = sum(1 for m in messages if isinstance(m, AIMessage))
    tool_calls = sum(1 for m in messages if isinstance(m, ToolMessage))
    return llm_calls, tool_calls


def run_task(task_id: str, query: str, run_idx: int = 0, *, verbose: bool = False) -> TaskRecord:
    """Execute a single benchmark task with the homogeneous agent.

    Returns a fully populated TaskRecord including energy measurements.
    """
    agent = _build_agent()
    record = TaskRecord(
        task_id=task_id,
        flow="homogeneous",
        run_idx=run_idx,
        query=query,
        models_used=[HOMOGENEOUS_MODEL],
    )

    if verbose:
        print(f"\n{_INDENT}{_DIM}── homogeneous agent ({HOMOGENEOUS_MODEL}) ──{_RESET}")

    with track_energy(f"homo_{task_id}_r{run_idx}") as tracking:
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"recursion_limit": MAX_AGENT_STEPS},
            )
            messages = result.get("messages", [])
            record.response = messages[-1].content if messages else ""
            llm_calls, tool_calls = _count_messages(messages)
            record.num_llm_calls = llm_calls
            record.num_tool_calls = tool_calls
            if verbose:
                _log_messages(messages)
        except Exception as exc:
            record.response = f"ERROR: {exc}"
            if verbose:
                print(f"{_INDENT}\033[91m[ERROR]{_RESET} {exc}")

    record.tracking = tracking
    return record

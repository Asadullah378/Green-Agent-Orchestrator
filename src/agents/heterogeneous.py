"""
Green Agent Orchestrator (GAO) — Flow 2: Heterogeneous routed agent

The orchestrator decomposes the user query into a small number of
self-contained subtasks and assigns each to the smallest capable worker
model from the configured pool. Each subtask then runs as a ReAct sub-agent
(with prior subtask results injected as context). A final synthesiser
combines the subtask outputs into a single answer.

Nothing in this module is hardcoded to a specific model family: the
orchestrator prompt, the worker pool, the difficulty→tier mapping, the
post-plan safety heuristic, and the per-tier step limits are all read from
the active `RunConfig` (see `src/config.py`).
"""

from __future__ import annotations

import json
import operator
import re
import textwrap
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, create_model
from typing_extensions import TypedDict

from src.config import RunConfig, get_config
from src.models import get_model, model_size_b
from src.tools import ALL_TOOLS
from src.tracking import TaskRecord, TrackingResult, track_energy


# ── Verbose logging helpers ──────────────────────────────────────────────────

_INDENT = "        "
_BLUE = "\033[94m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RED = "\033[91m"
_RESET = "\033[0m"

_verbose = False


def _log(text: str) -> None:
    if _verbose:
        print(f"{_INDENT}{text}")


_THINKING_RE = re.compile(r"<think>[\s\S]*?</think>")


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks emitted by some reasoning models."""
    return _THINKING_RE.sub("", text).strip()


# ── Pydantic schema for the orchestrator's structured plan ───────────────────


class Subtask(BaseModel):
    id: int = Field(description="Sequential subtask number starting from 1")
    description: str = Field(
        description=(
            "A SELF-CONTAINED description of what to do. "
            "Include all concrete values, numbers, and data the worker needs. "
            "The worker CANNOT see the original user query or other subtasks."
        )
    )
    tools_needed: list[str] = Field(
        default_factory=list,
        description="Tool names: calculator, unit_converter, data_lookup, date_calculator, text_processor",
    )
    difficulty: str = Field(description="One of: easy, medium, hard")
    assigned_model: str = Field(
        description=(
            "Name of the worker model to run this subtask. "
            "Choose the smallest capable model from the configured pool."
        )
    )


class TaskPlan(BaseModel):
    subtasks: list[Subtask] = Field(
        description="1-4 self-contained subtasks. Use 1 subtask when the task is simple. Fewer is always better."
    )
    synthesis_model: str = Field(
        default="",
        description="Model for the final synthesis step.",
    )


# ── LangGraph state ─────────────────────────────────────────────────────────


class GAOState(TypedDict):
    user_query: str
    plan: dict | None
    subtask_results: Annotated[list[dict], operator.add]
    current_idx: int
    final_response: str
    models_used: Annotated[list[str], operator.add]
    total_llm_calls: int
    total_tool_calls: int
    subtask_details: Annotated[list[dict], operator.add]


# ── Prompts (built dynamically from config) ──────────────────────────────────


def _build_orchestrator_prompt(cfg: RunConfig) -> str:
    """Build a model-family-agnostic orchestrator prompt from the worker pool."""
    hetero = cfg.heterogeneous
    lines = [
        "You are a task decomposition engine. "
        f"Break the user query into {hetero.decomposition.min_subtasks}-"
        f"{hetero.decomposition.max_subtasks} subtasks.",
        "",
        "Rules:",
        "- Use as FEW subtasks as possible. 1 subtask is ideal for simple tasks.",
        "- Each subtask description must be SELF-CONTAINED with ALL data/numbers needed.",
        "- Combine data-fetch + computation into ONE subtask when they need the same data.",
        "- Tools: calculator, unit_converter, data_lookup, date_calculator, text_processor.",
        "- Choose `assigned_model` for each subtask by matching expected complexity to one"
        " of the configured worker models:",
    ]
    for w in hetero.workers:
        desc = w.description or f"{w.tier} tier ({w.size_b}B params)"
        lines.append(f"    {w.model} = {desc}")
    lines.append(
        f'- Set `synthesis_model` to "{hetero.synthesizer.model}".'
    )
    return "\n".join(lines)


WORKER_SYSTEM = (
    "You are a focused worker agent. Complete the subtask described below "
    "using the available tools. Be precise and return concrete results.\n\n"
    "Subtask: {description}"
)

WORKER_SYSTEM_WITH_CONTEXT = (
    "You are a focused worker agent. Complete the subtask described below "
    "using the available tools. You have results from prior subtasks for context.\n\n"
    "Prior results:\n{context}\n\n"
    "Subtask: {description}"
)

SYNTHESIS_SYSTEM = (
    "You are a synthesis agent. Given the user's original question and "
    "the results of several subtasks, compose a clear, complete final answer. "
    "Include all relevant numbers and facts. Be concise."
)


# ── Plan parsing & validation ────────────────────────────────────────────────


def _extract_plan_from_text(raw: str) -> TaskPlan | None:
    """Best-effort recovery of a TaskPlan from a free-text model response."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    for candidate in [text, "{" + text.split("{", 1)[-1]]:
        try:
            data = json.loads(candidate)
            return TaskPlan.model_validate(data)
        except Exception:
            continue

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group())
            return TaskPlan.model_validate(data)
        except Exception:
            pass

    return None


_STRUCTURED_METHODS = ["function_calling", "json_schema"]


def _validate_plan(plan: TaskPlan, cfg: RunConfig) -> TaskPlan:
    """Apply config-driven safety checks: valid model names, routing safety."""
    hetero = cfg.heterogeneous
    worker_models = {w.model for w in hetero.workers}
    safety = hetero.routing_safety
    keywords = {k.lower() for k in safety.complexity_keywords}

    for st in plan.subtasks:
        if st.assigned_model not in worker_models:
            fallback_tier = hetero.difficulty_to_tier.get(
                st.difficulty, hetero.workers[len(hetero.workers) // 2].tier
            )
            st.assigned_model = hetero.model_for_tier(fallback_tier)

        if safety.enabled:
            from_model = hetero.model_for_tier(safety.from_tier)
            to_model = hetero.model_for_tier(safety.to_tier)
            if st.assigned_model == from_model:
                desc_lower = st.description.lower()
                hits = sum(1 for k in keywords if k in desc_lower)
                if (
                    hits >= safety.min_complexity_keywords
                    or len(st.description) > safety.max_description_length
                ):
                    old = st.assigned_model
                    st.assigned_model = to_model
                    st.difficulty = "medium"
                    _log(
                        f"{_YELLOW}[AUTO-UPGRADE]{_RESET} #{st.id}: "
                        f"{old}→{st.assigned_model} ({hits} complexity keywords)"
                    )

    if plan.synthesis_model not in worker_models | {hetero.synthesizer.model}:
        plan.synthesis_model = hetero.synthesizer.model

    return plan


# ── Graph node functions ─────────────────────────────────────────────────────


def orchestrate(state: GAOState) -> dict:
    """Decompose the user query into a structured plan."""
    cfg = get_config()
    hetero = cfg.heterogeneous
    orch_model_name = hetero.orchestrator.model

    _log(f"\n{_DIM}── orchestrator ({orch_model_name}) decomposing query ──{_RESET}")
    model = get_model(orch_model_name)
    messages = [
        SystemMessage(content=_build_orchestrator_prompt(cfg)),
        HumanMessage(content=state["user_query"]),
    ]

    plan: TaskPlan | None = None
    for method in _STRUCTURED_METHODS:
        try:
            structured = model.with_structured_output(
                TaskPlan, method=method, include_raw=True,
            )
            result = structured.invoke(messages)
            if result["parsed"] is not None:
                plan = result["parsed"]
                _log(f"{_GREEN}[PLAN OK]{_RESET} method={method}")
                break
            raw_text = ""
            if result.get("raw"):
                raw_text = (
                    result["raw"].content
                    if hasattr(result["raw"], "content")
                    else str(result["raw"])
                )
                plan = _extract_plan_from_text(raw_text)
                if plan:
                    _log(
                        f"{_GREEN}[RECOVERED]{_RESET} Extracted plan "
                        f"from raw output (method={method})"
                    )
                    break
            _log(
                f"{_YELLOW}[PARSE FAIL]{_RESET} method={method}, "
                f"raw={raw_text[:200]!r}"
            )
        except Exception as exc:
            _log(f"{_YELLOW}[METHOD FAIL]{_RESET} method={method}: {exc}")
            continue

    if plan is None:
        _log(f"{_YELLOW}[FALLBACK]{_RESET} Creating single-subtask plan")
        plan = TaskPlan(
            subtasks=[Subtask(
                id=1,
                description=state["user_query"],
                tools_needed=[t.name for t in ALL_TOOLS],
                difficulty="hard",
                assigned_model=hetero.model_for_difficulty("hard"),
            )],
            synthesis_model=hetero.synthesizer.model,
        )

    plan = _validate_plan(plan, cfg)

    if _verbose:
        _log(
            f"{_MAGENTA}[PLAN]{_RESET} {len(plan.subtasks)} subtask(s), "
            f"synthesis model: {plan.synthesis_model}"
        )
        for st in plan.subtasks:
            tools_str = ", ".join(st.tools_needed) if st.tools_needed else "none"
            _log(
                f"  {_BOLD}#{st.id}{_RESET} [{st.difficulty}] "
                f"{_CYAN}{st.assigned_model}{_RESET} — "
                f"{st.description[:120]}  {_DIM}tools: {tools_str}{_RESET}"
            )

    return {
        "plan": plan.model_dump(),
        "current_idx": 0,
        "models_used": [orch_model_name],
        "total_llm_calls": 1,
        "total_tool_calls": 0,
    }


def _step_limit_for_subtask(subtask: dict, cfg: RunConfig) -> int:
    """Resolve the per-subtask recursion limit from the assigned model's tier."""
    model_name = subtask["assigned_model"]
    tier = cfg.heterogeneous.tier_for_model(model_name)
    if tier is not None:
        return cfg.heterogeneous.tier(tier).max_steps
    # Fallback: use the middle tier's step limit.
    workers = cfg.heterogeneous.workers
    return workers[len(workers) // 2].max_steps


def execute_subtask(state: GAOState) -> dict:
    """Execute the current subtask with its assigned model + prior context."""
    cfg = get_config()
    plan = state["plan"]
    idx = state["current_idx"]
    subtask = plan["subtasks"][idx]

    model_name = subtask["assigned_model"]
    model = get_model(model_name)
    description = subtask["description"]

    _log(f"\n{_DIM}── subtask #{subtask['id']} ({model_name}) ──{_RESET}")
    _log(f"{_CYAN}[SUBTASK]{_RESET} {description[:150]}")

    tool_name_set = set(subtask.get("tools_needed", []))
    tools_for_subtask = [t for t in ALL_TOOLS if t.name in tool_name_set] or ALL_TOOLS

    prior_results = state.get("subtask_results", [])
    if prior_results:
        context = "\n".join(
            f"- Subtask {r['id']}: {r['result']}" for r in prior_results
        )
        system_prompt = WORKER_SYSTEM_WITH_CONTEXT.format(
            context=context, description=description
        )
    else:
        system_prompt = WORKER_SYSTEM.format(description=description)

    agent = create_react_agent(model, tools=tools_for_subtask, prompt=system_prompt)

    subtask_detail = {
        "subtask_id": subtask["id"],
        "description": description,
        "assigned_model": model_name,
        "model_size_b": model_size_b(model_name),
        "difficulty": subtask["difficulty"],
    }

    step_limit = _step_limit_for_subtask(subtask, cfg)
    _log(f"{_DIM}(step limit: {step_limit}){_RESET}")

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=description)]},
            config={"recursion_limit": step_limit},
        )
        messages = result.get("messages", [])
        response = _strip_thinking(messages[-1].content) if messages else ""
        llm_calls = sum(1 for m in messages if isinstance(m, AIMessage))
        tool_calls = sum(1 for m in messages if isinstance(m, ToolMessage))

        if _verbose:
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            args = json.dumps(tc["args"], ensure_ascii=False)
                            _log(f"{_YELLOW}[TOOL CALL]{_RESET} {tc['name']}({args})")
                    if msg.content:
                        wrapped = textwrap.shorten(msg.content, width=300, placeholder="…")
                        _log(f"{_BLUE}[LLM]{_RESET} {wrapped}")
                elif isinstance(msg, ToolMessage):
                    content = textwrap.shorten(str(msg.content), width=200, placeholder="…")
                    _log(f"{_GREEN}[TOOL RESULT]{_RESET} {content}")

    except Exception as exc:
        exc_str = str(exc)
        if "recursion" in exc_str.lower() or "limit" in exc_str.lower():
            _log(
                f"{_RED}[STEP LIMIT]{_RESET} Worker hit step limit "
                f"({step_limit}) — returning partial result"
            )
            response = f"Worker exceeded step limit ({step_limit} steps). Partial or no result."
        else:
            response = f"ERROR: {exc}"
            _log(f"{_RED}[ERROR]{_RESET} {exc}")
        llm_calls = 1
        tool_calls = 0

    subtask_detail["response"] = response
    subtask_detail["llm_calls"] = llm_calls
    subtask_detail["tool_calls"] = tool_calls

    return {
        "subtask_results": [{"id": subtask["id"], "description": description, "result": response}],
        "current_idx": idx + 1,
        "models_used": [model_name],
        "total_llm_calls": state["total_llm_calls"] + llm_calls,
        "total_tool_calls": state["total_tool_calls"] + tool_calls,
        "subtask_details": [subtask_detail],
    }


def synthesise(state: GAOState) -> dict:
    """Combine subtask results into a final answer."""
    cfg = get_config()
    plan = state["plan"]
    model_name = plan.get("synthesis_model") or cfg.heterogeneous.synthesizer.model
    model = get_model(model_name)

    _log(f"\n{_DIM}── synthesis ({model_name}) ──{_RESET}")

    results_text = "\n".join(
        f"- Subtask {r['id']}: {r['description']}\n  Result: {r['result']}"
        for r in state["subtask_results"]
    )

    prompt = (
        f"Original question: {state['user_query']}\n\n"
        f"Subtask results:\n{results_text}\n\n"
        "Provide a complete, well-structured final answer."
    )

    try:
        resp = model.invoke([
            SystemMessage(content=SYNTHESIS_SYSTEM),
            HumanMessage(content=prompt),
        ])
        final = _strip_thinking(resp.content)
    except Exception as exc:
        final = ""
        _log(f"{_RED}[ERROR]{_RESET} Synthesis failed: {exc}")

    if not final:
        final = "\n\n".join(r["result"] for r in state["subtask_results"])
        _log(f"{_YELLOW}[SYNTHESIS FALLBACK]{_RESET} Using concatenated subtask results")

    if _verbose:
        wrapped = textwrap.shorten(final, width=400, placeholder="…")
        _log(f"{_BLUE}[SYNTHESIS]{_RESET} {wrapped}")

    return {
        "final_response": final,
        "models_used": [model_name],
        "total_llm_calls": state["total_llm_calls"] + 1,
    }


def should_continue(state: GAOState) -> str:
    plan = state["plan"]
    if state["current_idx"] < len(plan["subtasks"]):
        return "execute_subtask"
    if len(plan["subtasks"]) == 1:
        return "skip_synthesis"
    return "synthesise"


# ── Build the graph ──────────────────────────────────────────────────────────


def skip_synthesis(state: GAOState) -> dict:
    """Use the single worker's response directly as the final answer."""
    result = state["subtask_results"][0]["result"]
    _log(f"\n{_DIM}── skipping synthesis (single subtask) ──{_RESET}")
    return {"final_response": result}


def _build_graph():
    g = StateGraph(GAOState)
    g.add_node("orchestrate", orchestrate)
    g.add_node("execute_subtask", execute_subtask)
    g.add_node("synthesise", synthesise)
    g.add_node("skip_synthesis", skip_synthesis)

    g.set_entry_point("orchestrate")
    g.add_conditional_edges("orchestrate", should_continue)
    g.add_conditional_edges("execute_subtask", should_continue)
    g.add_edge("synthesise", END)
    g.add_edge("skip_synthesis", END)

    return g.compile()


# ── Public API ───────────────────────────────────────────────────────────────


def run_task(task_id: str, query: str, run_idx: int = 0, *, verbose: bool = False) -> TaskRecord:
    """Execute a single benchmark task with the heterogeneous GAO agent."""
    global _verbose
    _verbose = verbose

    graph = _build_graph()
    record = TaskRecord(
        task_id=task_id,
        flow="heterogeneous",
        run_idx=run_idx,
        query=query,
    )

    initial_state: GAOState = {
        "user_query": query,
        "plan": None,
        "subtask_results": [],
        "current_idx": 0,
        "final_response": "",
        "models_used": [],
        "total_llm_calls": 0,
        "total_tool_calls": 0,
        "subtask_details": [],
    }

    with track_energy(f"hetero_{task_id}_r{run_idx}") as tracking:
        try:
            final_state = graph.invoke(initial_state)
            record.response = final_state.get("final_response", "")
            record.models_used = final_state.get("models_used", [])
            record.num_llm_calls = final_state.get("total_llm_calls", 0)
            record.num_tool_calls = final_state.get("total_tool_calls", 0)
            record.subtask_details = final_state.get("subtask_details", [])
        except Exception as exc:
            record.response = f"ERROR: {exc}"

    record.tracking = tracking
    return record

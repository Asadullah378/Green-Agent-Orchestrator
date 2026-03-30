"""
Green Agent Orchestrator (GAO) — Main experiment runner

Runs all benchmark tasks through both flows, records energy / accuracy /
timing, and writes results to JSON + CSV.  After saving, automatically
runs the analysis pipeline to generate paper-ready figures and tables.

Usage:
    python -m src.run_experiment                          # full run (all tasks, 3 runs)
    python -m src.run_experiment --flow homogeneous        # one flow only
    python -m src.run_experiment --tasks E1 E2             # specific tasks
    python -m src.run_experiment --difficulty easy          # all easy tasks
    python -m src.run_experiment --runs 1                   # single repetition
    python -m src.run_experiment -v                         # verbose agent logs
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import homogeneous, heterogeneous
from src.benchmark.evaluators import evaluate_record
from src.benchmark.tasks import BENCHMARK_TASKS, get_tasks_by_difficulty
from src.config import (
    HOMOGENEOUS_MODEL,
    ORCHESTRATOR_MODEL,
    HETEROGENEOUS_POOL,
    NUM_RUNS,
    RESULTS_DIR,
)
from src.tracking import TaskRecord


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _collect_metadata() -> dict:
    """Capture system and configuration info for reproducibility."""
    import codecarbon

    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "codecarbon_version": codecarbon.__version__,
        "homogeneous_model": HOMOGENEOUS_MODEL,
        "orchestrator_model": ORCHESTRATOR_MODEL,
        "heterogeneous_pool": list(HETEROGENEOUS_POOL.keys()),
    }


def run_single(
    flow: str, task: dict, run_idx: int, *, verbose: bool = False
) -> TaskRecord:
    """Run one task on one flow and return a scored TaskRecord."""
    runner = homogeneous if flow == "homogeneous" else heterogeneous
    print(
        f"  [{flow[:5].upper()}] Task {task['id']} "
        f"(run {run_idx + 1}) — {task['description']}…",
        end="\n" if verbose else " ",
        flush=True,
    )
    t0 = time.perf_counter()
    record = runner.run_task(task["id"], task["query"], run_idx, verbose=verbose)
    record.accuracy_score = evaluate_record(record)
    elapsed = time.perf_counter() - t0
    if verbose:
        print(
            f"  ✓  acc={record.accuracy_score:.2f}  "
            f"energy={record.tracking.energy_kwh:.6f} kWh  "
            f"time={elapsed:.1f}s"
        )
    else:
        print(
            f"✓  acc={record.accuracy_score:.2f}  "
            f"energy={record.tracking.energy_kwh:.6f} kWh  "
            f"time={elapsed:.1f}s"
        )
    return record


def run_experiment(
    flows: list[str],
    task_ids: list[str] | None,
    difficulty: str | None,
    num_runs: int,
    *,
    verbose: bool = False,
) -> list[dict]:
    """Execute the full experiment matrix and return a list of result dicts."""
    tasks = BENCHMARK_TASKS
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]
    elif difficulty:
        tasks = get_tasks_by_difficulty(difficulty)

    if not tasks:
        print("No matching tasks found.")
        return []

    results: list[dict] = []

    # Alternate flows per task so both see the same thermal/cache conditions.
    # Order: for each task, for each run, run every flow back-to-back.
    try:
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"  TASK: {task['id']} — {task['description']}")
            print(f"{'='*60}")
            for run_idx in range(num_runs):
                for flow in flows:
                    record = run_single(flow, task, run_idx, verbose=verbose)
                    results.append(record.to_dict())
    except KeyboardInterrupt:
        print(f"\n\n  ⚠ Interrupted! Saving {len(results)} partial results…")

    return results


def save_results(results: list[dict], tag: str = "") -> tuple[str, str]:
    """Persist results as JSON (with metadata) and CSV."""
    _ensure_results_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"results_{tag}_{ts}" if tag else f"results_{ts}"

    json_path = os.path.join(RESULTS_DIR, f"{stem}.json")
    csv_path = os.path.join(RESULTS_DIR, f"{stem}.csv")

    payload = {
        "metadata": _collect_metadata(),
        "results": results,
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to:\n  {json_path}\n  {csv_path}")
    return json_path, csv_path


def print_summary(results: list[dict]):
    """Print a quick comparison table to stdout."""
    df = pd.DataFrame(results)
    if df.empty:
        return

    summary = (
        df.groupby("flow")
        .agg(
            tasks=("task_id", "nunique"),
            runs=("task_id", "count"),
            avg_accuracy=("accuracy_score", "mean"),
            avg_energy_kwh=("energy_kwh", "mean"),
            total_energy_kwh=("energy_kwh", "sum"),
            avg_duration_s=("duration_seconds", "mean"),
            total_duration_s=("duration_seconds", "sum"),
            avg_llm_calls=("num_llm_calls", "mean"),
        )
        .round(6)
    )
    print(f"\n{'='*60}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(summary.to_string())

    if {"homogeneous", "heterogeneous"}.issubset(set(df["flow"])):
        e_homo = df.loc[df.flow == "homogeneous", "energy_kwh"].sum()
        e_hetero = df.loc[df.flow == "heterogeneous", "energy_kwh"].sum()
        if e_homo > 0:
            saving = (1 - e_hetero / e_homo) * 100
            print(f"\n  Energy saving (heterogeneous vs homogeneous): {saving:+.1f}%")

    from src.benchmark.tasks import get_task_by_id

    df["difficulty"] = df["task_id"].apply(
        lambda tid: (get_task_by_id(tid) or {}).get("difficulty", "?")
    )
    diff_summary = (
        df.groupby(["flow", "difficulty"])
        .agg(
            avg_accuracy=("accuracy_score", "mean"),
            avg_energy_kwh=("energy_kwh", "mean"),
            avg_duration_s=("duration_seconds", "mean"),
        )
        .round(6)
    )
    print(f"\n{'='*60}")
    print("  PER-DIFFICULTY BREAKDOWN")
    print(f"{'='*60}")
    print(diff_summary.to_string())


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="GAO experiment runner")
    parser.add_argument(
        "--flow",
        choices=["homogeneous", "heterogeneous", "both"],
        default="both",
        help="Which flow(s) to run (default: both)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Specific task IDs to run (default: all)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Run only tasks of this difficulty level",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=NUM_RUNS,
        help=f"Repetitions per task (default: {NUM_RUNS})",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional tag appended to result filenames",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Print detailed agent logs (LLM responses, tool calls, plans)",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        default=False,
        help="Skip automatic analysis after experiment",
    )
    args = parser.parse_args()

    flows = (
        ["homogeneous", "heterogeneous"]
        if args.flow == "both"
        else [args.flow]
    )

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Green Agent Orchestrator — Experiment Runner          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Flows      : {flows}")
    print(f"  Tasks      : {args.tasks or args.difficulty or 'ALL'}")
    print(f"  Runs       : {args.runs}")
    print(f"  Verbose    : {args.verbose}")

    results = run_experiment(
        flows, args.tasks, args.difficulty, args.runs, verbose=args.verbose,
    )

    if results:
        json_path, _ = save_results(results, args.tag)
        print_summary(results)

        if not args.no_analyze:
            print(f"\n{'='*60}")
            print("  GENERATING ANALYSIS & FIGURES")
            print(f"{'='*60}")
            from src.analyze_results import run_analysis
            run_analysis(json_path)


if __name__ == "__main__":
    main()

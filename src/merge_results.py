"""
Green Agent Orchestrator (GAO) — Merge multiple result JSON files

When experiments are run in batches (e.g., easy tasks first, then medium,
then hard), this utility merges them into a single file for analysis.

Usage:
    python -m src.merge_results results/results_easy.json results/results_medium.json results/results_hard.json
    python -m src.merge_results results/results_*.json
    python -m src.merge_results results/results_*.json -o results/merged_all.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR


def load_json(path: str) -> tuple[dict | None, list[dict]]:
    """Load a results JSON, returning (metadata, results_list)."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        return data.get("metadata"), data["results"]
    if isinstance(data, list):
        return None, data
    raise ValueError(f"Unrecognized format in {path}")


def merge(paths: list[str]) -> dict:
    """Merge multiple result files, deduplicating by (task_id, flow, run_idx)."""
    all_results: list[dict] = []
    metadata_list: list[dict] = []
    seen: set[tuple] = set()

    for p in paths:
        meta, results = load_json(p)
        if meta:
            metadata_list.append(meta)
        for r in results:
            key = (r.get("task_id"), r.get("flow"), r.get("run_idx"))
            if key not in seen:
                seen.add(key)
                all_results.append(r)
            else:
                print(f"  Skipping duplicate: {key}")

    merged_meta = {
        "merged_at": datetime.now().isoformat(),
        "source_files": paths,
        "total_records": len(all_results),
    }
    if metadata_list:
        merged_meta["source_metadata"] = metadata_list[0]

    return {"metadata": merged_meta, "results": all_results}


def main():
    parser = argparse.ArgumentParser(description="Merge GAO result files")
    parser.add_argument("files", nargs="+", help="JSON result files to merge")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path (default: results/merged_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        default=False,
        help="Run analysis on the merged file",
    )
    args = parser.parse_args()

    print(f"Merging {len(args.files)} file(s)…")
    merged = merge(args.files)
    n = merged["metadata"]["total_records"]
    print(f"  Total records: {n}")

    out_path = args.output
    if not out_path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(RESULTS_DIR, f"merged_{ts}.json")

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"  Saved → {out_path}")

    if args.analyze:
        from src.analyze_results import run_analysis
        run_analysis(out_path)


if __name__ == "__main__":
    main()

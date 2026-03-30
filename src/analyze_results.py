"""
Green Agent Orchestrator (GAO) — Comprehensive results analysis

Generates all figures, LaTeX tables, and statistics needed for the
IEEE-format research paper.  Outputs are saved to:
  results/figures/  — high-resolution PNGs (300 DPI)
  results/tables/   — LaTeX table snippets for \\input{}

Usage:
    python -m src.analyze_results results/results_XXXX.json
    python -m src.analyze_results results/merged_all.json --out-dir results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
})

# Colorblind-friendly palette (orange vs blue)
FLOW_COLORS = {"homogeneous": "#D35400", "heterogeneous": "#2471A3"}
FLOW_HATCHES = {"homogeneous": "", "heterogeneous": ""}
FLOW_LABELS = {"homogeneous": "Homogeneous (27B)", "heterogeneous": "Heterogeneous (GAO)"}
DIFF_ORDER = ["easy", "medium", "hard"]
DIFF_COLORS = {"easy": "#27AE60", "medium": "#F39C12", "hard": "#C0392B"}

_FIG_DIR = ""
_TAB_DIR = ""


def _savefig(fig: plt.Figure, name: str):
    path = os.path.join(_FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Figure → {path}")


def _savetex(content: str, name: str):
    path = os.path.join(_TAB_DIR, name)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Table  → {path}")


def _add_bar_labels(ax, bars, fmt="%.4f", fontsize=6.5):
    """Add value labels above bars with automatic vertical room."""
    for bar in bars:
        val = bar.get_height()
        if np.isnan(val):
            continue
        label = fmt % val
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=fontsize,
            color="#333333",
        )


# ── Loading ──────────────────────────────────────────────────────────────────


def load_results(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        records = data["results"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unrecognized format in {path}")

    df = pd.DataFrame(records)
    df["difficulty"] = df["task_id"].apply(
        lambda t: {"E": "easy", "M": "medium", "H": "hard"}.get(t[0], "?")
    )
    df["ets"] = df["energy_kwh"] / df["accuracy_score"].clip(lower=0.01)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  LATEX TABLES
# ═══════════════════════════════════════════════════════════════════════════════


def _fmt(val, precision=4):
    return f"{val:.{precision}f}"


def _fmt_pm(mean, std, precision=4):
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def table_overall_comparison(df: pd.DataFrame):
    rows = []
    for flow in ["homogeneous", "heterogeneous"]:
        sub = df[df.flow == flow]
        rows.append({
            "flow": flow.capitalize(),
            "n": len(sub),
            "acc": _fmt_pm(sub.accuracy_score.mean(), sub.accuracy_score.std(), 2),
            "energy": _fmt_pm(sub.energy_kwh.mean(), sub.energy_kwh.std(), 6),
            "energy_total": _fmt(sub.energy_kwh.sum(), 6),
            "duration": _fmt_pm(sub.duration_seconds.mean(), sub.duration_seconds.std(), 1),
            "ets": _fmt_pm(sub.ets.mean(), sub.ets.std(), 6),
            "co2": _fmt_pm(sub.emissions_kg_co2.mean(), sub.emissions_kg_co2.std(), 8),
            "llm_calls": _fmt(sub.num_llm_calls.mean(), 1),
        })

    e_homo = df.loc[df.flow == "homogeneous", "energy_kwh"].mean()
    e_hetero = df.loc[df.flow == "heterogeneous", "energy_kwh"].mean()
    saving = (1 - e_hetero / e_homo) * 100 if e_homo > 0 else 0

    tex = r"""\begin{table}[t]
\centering
\caption{Overall comparison of homogeneous and heterogeneous flows.}
\label{tab:overall}
\begin{tabularx}{\columnwidth}{lXX}
\toprule
\textbf{Metric} & \textbf{Homogeneous} & \textbf{Heterogeneous} \\
\midrule
"""
    labels = [
        ("Accuracy", "acc"),
        ("Energy (kWh)", "energy"),
        ("Total Energy (kWh)", "energy_total"),
        ("Duration (s)", "duration"),
        ("EtS (kWh/acc)", "ets"),
        (r"CO$_2$ (kg)", "co2"),
        ("Avg LLM Calls", "llm_calls"),
    ]
    for label, key in labels:
        tex += f"{label} & {rows[0][key]} & {rows[1][key]} \\\\\n"

    tex += r"""\midrule
"""
    tex += f"Energy Saving & \\multicolumn{{2}}{{c}}{{{saving:+.1f}\\%}} \\\\\n"
    tex += r"""\bottomrule
\end{tabularx}
\end{table}
"""
    _savetex(tex, "table_overall.tex")
    print(f"\n  Overall energy saving: {saving:+.1f}%")
    return saving


def table_per_difficulty(df: pd.DataFrame):
    tex = r"""\begin{table}[t]
\centering
\caption{Results by task difficulty.}
\label{tab:difficulty}
\begin{tabularx}{\columnwidth}{llXXXX}
\toprule
\textbf{Difficulty} & \textbf{Flow} & \textbf{Acc.} & \textbf{Energy (kWh)} & \textbf{Duration (s)} & \textbf{EtS} \\
\midrule
"""
    for diff in DIFF_ORDER:
        first = True
        for flow in ["homogeneous", "heterogeneous"]:
            sub = df[(df.difficulty == diff) & (df.flow == flow)]
            if sub.empty:
                continue
            diff_label = diff.capitalize() if first else ""
            first = False
            tex += (
                f"{diff_label} & {flow[:5].capitalize()}. "
                f"& {_fmt(sub.accuracy_score.mean(), 2)} "
                f"& {_fmt(sub.energy_kwh.mean(), 6)} "
                f"& {_fmt(sub.duration_seconds.mean(), 1)} "
                f"& {_fmt(sub.ets.mean(), 6)} \\\\\n"
            )
        tex += r"\midrule" + "\n"

    tex = tex.rstrip("\\midrule\n") + "\n"
    tex += r"""\bottomrule
\end{tabularx}
\end{table}
"""
    _savetex(tex, "table_difficulty.tex")


def table_per_task(df: pd.DataFrame):
    tasks_sorted = sorted(df.task_id.unique(), key=lambda t: (
        {"E": 0, "M": 1, "H": 2}.get(t[0], 3), int(t[1:])
    ))

    tex = r"""\begin{table*}[t]
\centering
\caption{Per-task results (averaged over runs).}
\label{tab:pertask}
\begin{tabularx}{\textwidth}{ll *{6}{X}}
\toprule
& & \multicolumn{3}{c}{\textbf{Homogeneous}} & \multicolumn{3}{c}{\textbf{Heterogeneous}} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8}
\textbf{ID} & \textbf{Difficulty} & \textbf{Acc.} & \textbf{Energy} & \textbf{Dur.(s)} & \textbf{Acc.} & \textbf{Energy} & \textbf{Dur.(s)} \\
\midrule
"""
    for tid in tasks_sorted:
        diff = {"E": "easy", "M": "medium", "H": "hard"}.get(tid[0], "?")
        homo = df[(df.task_id == tid) & (df.flow == "homogeneous")]
        hetero = df[(df.task_id == tid) & (df.flow == "heterogeneous")]
        h_acc = _fmt(homo.accuracy_score.mean(), 2) if not homo.empty else "--"
        h_eng = _fmt(homo.energy_kwh.mean(), 6) if not homo.empty else "--"
        h_dur = _fmt(homo.duration_seconds.mean(), 1) if not homo.empty else "--"
        t_acc = _fmt(hetero.accuracy_score.mean(), 2) if not hetero.empty else "--"
        t_eng = _fmt(hetero.energy_kwh.mean(), 6) if not hetero.empty else "--"
        t_dur = _fmt(hetero.duration_seconds.mean(), 1) if not hetero.empty else "--"
        tex += f"{tid} & {diff} & {h_acc} & {h_eng} & {h_dur} & {t_acc} & {t_eng} & {t_dur} \\\\\n"

    tex += r"""\bottomrule
\end{tabularx}
\end{table*}
"""
    _savetex(tex, "table_pertask.tex")


def table_energy_savings_by_difficulty(df: pd.DataFrame):
    tex = r"""\begin{table}[t]
\centering
\caption{Energy savings of heterogeneous flow by difficulty.}
\label{tab:savings}
\begin{tabularx}{\columnwidth}{lXXX}
\toprule
\textbf{Difficulty} & \textbf{Homo. (kWh)} & \textbf{Hetero. (kWh)} & \textbf{Saving (\%)} \\
\midrule
"""
    savings_by_diff = {}
    for diff in DIFF_ORDER:
        homo = df[(df.difficulty == diff) & (df.flow == "homogeneous")]
        hetero = df[(df.difficulty == diff) & (df.flow == "heterogeneous")]
        if homo.empty or hetero.empty:
            continue
        e_h = homo.energy_kwh.mean()
        e_t = hetero.energy_kwh.mean()
        saving = (1 - e_t / e_h) * 100 if e_h > 0 else 0
        savings_by_diff[diff] = saving
        tex += f"{diff.capitalize()} & {_fmt(e_h, 6)} & {_fmt(e_t, 6)} & {saving:+.1f}\\% \\\\\n"

    e_h_all = df.loc[df.flow == "homogeneous", "energy_kwh"].mean()
    e_t_all = df.loc[df.flow == "heterogeneous", "energy_kwh"].mean()
    total_saving = (1 - e_t_all / e_h_all) * 100 if e_h_all > 0 else 0
    tex += r"\midrule" + "\n"
    tex += f"\\textbf{{Overall}} & {_fmt(e_h_all, 6)} & {_fmt(e_t_all, 6)} & \\textbf{{{total_saving:+.1f}\\%}} \\\\\n"

    tex += r"""\bottomrule
\end{tabularx}
\end{table}
"""
    _savetex(tex, "table_savings.tex")

    for d, s in savings_by_diff.items():
        print(f"  {d}: {s:+.1f}% energy saving")


def table_model_usage(df: pd.DataFrame):
    hetero = df[df.flow == "heterogeneous"]
    if hetero.empty:
        return

    all_models: list[str] = []
    for models_used in hetero["models_used"]:
        if isinstance(models_used, list):
            all_models.extend(models_used)
        elif isinstance(models_used, str):
            try:
                parsed = json.loads(models_used.replace("'", '"'))
                all_models.extend(parsed)
            except (json.JSONDecodeError, TypeError):
                pass

    if not all_models:
        return

    counts = Counter(all_models)
    total = sum(counts.values())

    tex = r"""\begin{table}[t]
\centering
\caption{Model invocation distribution in heterogeneous flow.}
\label{tab:modelusage}
\begin{tabularx}{\columnwidth}{lXXX}
\toprule
\textbf{Model} & \textbf{Size} & \textbf{Invocations} & \textbf{Share (\%)} \\
\midrule
"""
    from src.config import MODEL_POOL
    for model in sorted(counts.keys(), key=lambda m: MODEL_POOL.get(m, {}).get("size_b", 0)):
        size = MODEL_POOL.get(model, {}).get("size_b", "?")
        pct = counts[model] / total * 100
        tex += f"{model} & {size}B & {counts[model]} & {pct:.1f}\\% \\\\\n"

    tex += r"""\bottomrule
\end{tabularx}
\end{table}
"""
    _savetex(tex, "table_model_usage.tex")


def table_experiment_setup(df: pd.DataFrame):
    n_tasks = df.task_id.nunique()
    runs_per = df.groupby(["flow", "task_id"]).size().max() if not df.empty else 0
    from src.config import HOMOGENEOUS_MODEL, ORCHESTRATOR_MODEL, HETEROGENEOUS_POOL, LLM_TEMPERATURE

    tex = r"""\begin{table}[t]
\centering
\caption{Experiment configuration.}
\label{tab:setup}
\begin{tabularx}{\columnwidth}{lX}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
"""
    rows = [
        ("Homogeneous model", HOMOGENEOUS_MODEL.replace("_", r"\_")),
        ("Orchestrator model", ORCHESTRATOR_MODEL),
        ("Worker model pool", ", ".join(HETEROGENEOUS_POOL.keys())),
        ("Benchmark tasks", f"{n_tasks} (5 easy, 5 medium, 5 hard)"),
        ("Runs per task", str(runs_per)),
        ("Temperature", str(LLM_TEMPERATURE)),
        ("Energy tracker", "CodeCarbon (offline, process mode)"),
        ("Inference runtime", "Ollama (local, CPU)"),
    ]
    for k, v in rows:
        tex += f"{k} & {v} \\\\\n"

    tex += r"""\bottomrule
\end{tabularx}
\end{table}
"""
    _savetex(tex, "table_setup.tex")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

_COL1 = (4.0, 3.0)
_COL2 = (7.5, 3.2)


def _flow_bar(df: pd.DataFrame, col: str, ylabel: str, title: str, fname: str,
              fmt: str = "%.4f", figsize=_COL1):
    """Two-bar chart comparing flows with error bars and headroom."""
    means = df.groupby("flow")[col].mean().reindex(["homogeneous", "heterogeneous"])
    stds = df.groupby("flow")[col].std().reindex(["homogeneous", "heterogeneous"])
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(means))
    colors = [FLOW_COLORS[f] for f in means.index]
    bars = ax.bar(x, means.values, yerr=stds.values, color=colors,
                  capsize=5, edgecolor="white", linewidth=0.8, width=0.5,
                  error_kw={"linewidth": 0.8, "capthick": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([FLOW_LABELS.get(f, f) for f in means.index])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _add_bar_labels(ax, bars, fmt=fmt)
    ymax = (means + stds.fillna(0)).max()
    ax.set_ylim(0, ymax * 1.25)
    fig.tight_layout()
    _savefig(fig, fname)


def _difficulty_grouped_bar(df: pd.DataFrame, col: str, ylabel: str, title: str,
                            fname: str, fmt: str = "%.4f", figsize=(4.0, 3.0)):
    """Grouped bar: difficulties on x-axis, flows as groups, with headroom."""
    pivot = df.pivot_table(index="difficulty", columns="flow", values=col, aggfunc="mean")
    pivot = pivot.reindex(DIFF_ORDER).reindex(columns=["homogeneous", "heterogeneous"])

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(DIFF_ORDER))
    w = 0.32
    all_bars = []
    for i, flow in enumerate(["homogeneous", "heterogeneous"]):
        if flow not in pivot.columns:
            continue
        vals = pivot[flow].values
        bars = ax.bar(x + i * w - w / 2, vals, w,
                      label=FLOW_LABELS[flow], color=FLOW_COLORS[flow],
                      edgecolor="white", linewidth=0.8)
        _add_bar_labels(ax, bars, fmt=fmt, fontsize=6)
        all_bars.extend(bars)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DIFF_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none")
    max_val = max(b.get_height() for b in all_bars if not np.isnan(b.get_height()))
    ax.set_ylim(0, max_val * 1.25)
    fig.tight_layout()
    _savefig(fig, fname)


# ── Individual figure functions ───────────────────────────────────────────────

def fig_energy_by_difficulty(df: pd.DataFrame):
    _difficulty_grouped_bar(
        df, "energy_kwh", "Energy (kWh)", "Energy Consumption by Difficulty",
        "fig_energy_difficulty.png", "%.5f",
    )


def fig_accuracy_by_difficulty(df: pd.DataFrame):
    pivot = df.pivot_table(index="difficulty", columns="flow", values="accuracy_score", aggfunc="mean")
    pivot = pivot.reindex(DIFF_ORDER).reindex(columns=["homogeneous", "heterogeneous"])

    fig, ax = plt.subplots(figsize=_COL1)
    x = np.arange(len(DIFF_ORDER))
    w = 0.32
    for i, flow in enumerate(["homogeneous", "heterogeneous"]):
        if flow not in pivot.columns:
            continue
        vals = pivot[flow].values
        bars = ax.bar(x + i * w - w / 2, vals, w,
                      label=FLOW_LABELS[flow], color=FLOW_COLORS[flow],
                      edgecolor="white", linewidth=0.8)
        _add_bar_labels(ax, bars, fmt="%.2f", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DIFF_ORDER])
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Difficulty")
    ax.set_ylim(0, 1.18)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    _savefig(fig, "fig_accuracy_difficulty.png")


def fig_duration_by_difficulty(df: pd.DataFrame):
    _difficulty_grouped_bar(
        df, "duration_seconds", "Duration (s)", "Execution Time by Difficulty",
        "fig_duration_difficulty.png", "%.0f",
    )


def fig_ets_by_difficulty(df: pd.DataFrame):
    _difficulty_grouped_bar(
        df, "ets", "EtS (kWh / accuracy)", "Energy-to-Solution by Difficulty",
        "fig_ets_difficulty.png", "%.5f",
    )


def fig_energy_overall(df: pd.DataFrame):
    _flow_bar(df, "energy_kwh", "Avg Energy (kWh)",
              "Average Energy per Task", "fig_energy_overall.png", "%.6f")


def fig_accuracy_overall(df: pd.DataFrame):
    means = df.groupby("flow")["accuracy_score"].mean().reindex(["homogeneous", "heterogeneous"])
    stds = df.groupby("flow")["accuracy_score"].std().reindex(["homogeneous", "heterogeneous"])
    fig, ax = plt.subplots(figsize=_COL1)
    x = np.arange(len(means))
    colors = [FLOW_COLORS[f] for f in means.index]
    bars = ax.bar(x, means.values, yerr=stds.values, color=colors,
                  capsize=5, edgecolor="white", linewidth=0.8, width=0.5,
                  error_kw={"linewidth": 0.8, "capthick": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([FLOW_LABELS.get(f, f) for f in means.index])
    ax.set_ylabel("Accuracy")
    ax.set_title("Average Accuracy")
    _add_bar_labels(ax, bars, fmt="%.2f")
    ax.set_ylim(0, 1.18)
    fig.tight_layout()
    _savefig(fig, "fig_accuracy_overall.png")


def fig_ets_overall(df: pd.DataFrame):
    _flow_bar(df, "ets", "EtS (kWh / accuracy)",
              "Energy-to-Solution (lower is better)", "fig_ets_overall.png", "%.6f")


def fig_per_task_energy(df: pd.DataFrame):
    """Horizontal grouped bars: energy per task, colored by difficulty region."""
    tasks_sorted = sorted(df.task_id.unique(), key=lambda t: (
        {"E": 0, "M": 1, "H": 2}.get(t[0], 3), int(t[1:])
    ))
    pivot = df.pivot_table(index="task_id", columns="flow", values="energy_kwh", aggfunc="mean")
    pivot = pivot.reindex(tasks_sorted).reindex(columns=["homogeneous", "heterogeneous"])

    fig, ax = plt.subplots(figsize=(4.5, 5.0))
    y = np.arange(len(tasks_sorted))
    h = 0.35
    for i, flow in enumerate(["homogeneous", "heterogeneous"]):
        if flow not in pivot.columns:
            continue
        vals = pivot[flow].values
        ax.barh(y + i * h - h / 2, vals, h,
                label=FLOW_LABELS[flow], color=FLOW_COLORS[flow],
                edgecolor="white", linewidth=0.5)

    # Shade difficulty regions
    for region, ys, color in [("Easy", (-0.5, 4.5), "#27AE6015"),
                               ("Medium", (4.5, 9.5), "#F39C1215"),
                               ("Hard", (9.5, 14.5), "#C0392B15")]:
        ax.axhspan(ys[0], ys[1], color=color, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(tasks_sorted, fontsize=7)
    ax.set_xlabel("Energy (kWh)")
    ax.set_title("Per-Task Energy Comparison")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9, edgecolor="none")
    ax.invert_yaxis()
    fig.tight_layout()
    _savefig(fig, "fig_per_task_energy.png")


def fig_per_task_accuracy(df: pd.DataFrame):
    """Horizontal grouped bars: accuracy per task."""
    tasks_sorted = sorted(df.task_id.unique(), key=lambda t: (
        {"E": 0, "M": 1, "H": 2}.get(t[0], 3), int(t[1:])
    ))
    pivot = df.pivot_table(index="task_id", columns="flow", values="accuracy_score", aggfunc="mean")
    pivot = pivot.reindex(tasks_sorted).reindex(columns=["homogeneous", "heterogeneous"])

    fig, ax = plt.subplots(figsize=(4.5, 5.0))
    y = np.arange(len(tasks_sorted))
    h = 0.35
    for i, flow in enumerate(["homogeneous", "heterogeneous"]):
        if flow not in pivot.columns:
            continue
        vals = pivot[flow].values
        ax.barh(y + i * h - h / 2, vals, h,
                label=FLOW_LABELS[flow], color=FLOW_COLORS[flow],
                edgecolor="white", linewidth=0.5)

    for region, ys, color in [("Easy", (-0.5, 4.5), "#27AE6015"),
                               ("Medium", (4.5, 9.5), "#F39C1215"),
                               ("Hard", (9.5, 14.5), "#C0392B15")]:
        ax.axhspan(ys[0], ys[1], color=color, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(tasks_sorted, fontsize=7)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Task Accuracy Comparison")
    ax.set_xlim(0, 1.12)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9, edgecolor="none")
    ax.invert_yaxis()
    fig.tight_layout()
    _savefig(fig, "fig_per_task_accuracy.png")


def fig_energy_savings_bar(df: pd.DataFrame):
    """Bar chart showing energy saving % by difficulty + overall."""
    savings = {}
    for diff in DIFF_ORDER:
        homo = df[(df.difficulty == diff) & (df.flow == "homogeneous")].energy_kwh.mean()
        hetero = df[(df.difficulty == diff) & (df.flow == "heterogeneous")].energy_kwh.mean()
        if homo > 0:
            savings[diff] = (1 - hetero / homo) * 100

    e_all_h = df.loc[df.flow == "homogeneous", "energy_kwh"].mean()
    e_all_t = df.loc[df.flow == "heterogeneous", "energy_kwh"].mean()
    if e_all_h > 0:
        savings["overall"] = (1 - e_all_t / e_all_h) * 100

    if not savings:
        return

    fig, ax = plt.subplots(figsize=_COL1)
    labels = list(savings.keys())
    vals = list(savings.values())
    x = np.arange(len(labels))
    colors = [DIFF_COLORS.get(l, "#2C3E50") for l in labels]
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.55)

    for bar, v in zip(bars, vals):
        ax.annotate(
            f"{v:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in labels])
    ax.set_ylabel("Energy Saving (%)")
    ax.set_title("Energy Savings by Difficulty")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylim(0, max(vals) * 1.2)
    fig.tight_layout()
    _savefig(fig, "fig_energy_savings.png")


def fig_duration_savings_bar(df: pd.DataFrame):
    """Bar chart showing time saving % by difficulty + overall."""
    savings = {}
    for diff in DIFF_ORDER:
        homo = df[(df.difficulty == diff) & (df.flow == "homogeneous")].duration_seconds.mean()
        hetero = df[(df.difficulty == diff) & (df.flow == "heterogeneous")].duration_seconds.mean()
        if homo > 0:
            savings[diff] = (1 - hetero / homo) * 100

    d_all_h = df.loc[df.flow == "homogeneous", "duration_seconds"].mean()
    d_all_t = df.loc[df.flow == "heterogeneous", "duration_seconds"].mean()
    if d_all_h > 0:
        savings["overall"] = (1 - d_all_t / d_all_h) * 100

    if not savings:
        return

    fig, ax = plt.subplots(figsize=_COL1)
    labels = list(savings.keys())
    vals = list(savings.values())
    x = np.arange(len(labels))
    colors = [DIFF_COLORS.get(l, "#2C3E50") for l in labels]
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.55)

    for bar, v in zip(bars, vals):
        ax.annotate(
            f"{v:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in labels])
    ax.set_ylabel("Time Saving (%)")
    ax.set_title("Execution Time Savings by Difficulty")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylim(0, max(vals) * 1.2)
    fig.tight_layout()
    _savefig(fig, "fig_duration_savings.png")


def fig_tradeoff_scatter(df: pd.DataFrame):
    """Energy vs accuracy scatter with difficulty markers and flow colors."""
    markers = {"easy": "o", "medium": "s", "hard": "D"}
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for flow in ["homogeneous", "heterogeneous"]:
        for diff in DIFF_ORDER:
            sub = df[(df.flow == flow) & (df.difficulty == diff)]
            if sub.empty:
                continue
            ax.scatter(
                sub.energy_kwh, sub.accuracy_score,
                label=f"{FLOW_LABELS[flow][:5]}. ({diff})",
                color=FLOW_COLORS[flow],
                marker=markers[diff],
                alpha=0.7, edgecolors="white", linewidth=0.5, s=35,
            )

    ax.set_xlabel("Energy (kWh)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Energy–Accuracy Trade-off")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=6, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.35), framealpha=0.9, edgecolor="none")
    fig.subplots_adjust(bottom=0.28)
    _savefig(fig, "fig_tradeoff.png")


def fig_energy_breakdown(df: pd.DataFrame):
    """Stacked bar: CPU + GPU + RAM energy by flow and difficulty."""
    components = []
    for c in ["cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh"]:
        if c in df.columns:
            components.append(c)
    if not components:
        return

    comp_labels = {"cpu_energy_kwh": "CPU", "gpu_energy_kwh": "GPU", "ram_energy_kwh": "RAM"}
    comp_colors = {"cpu_energy_kwh": "#2980B9", "gpu_energy_kwh": "#E74C3C", "ram_energy_kwh": "#F39C12"}

    groups = df.groupby("flow")[components].mean()
    groups = groups.reindex(["homogeneous", "heterogeneous"])

    fig, ax = plt.subplots(figsize=_COL1)
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))

    for comp in components:
        vals = groups[comp].values
        ax.bar(x, vals, 0.5, bottom=bottom,
               label=comp_labels[comp], color=comp_colors[comp],
               edgecolor="white", linewidth=0.8)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([FLOW_LABELS.get(f, f) for f in groups.index], fontsize=8)
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Energy Breakdown by Component")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9, edgecolor="none")
    ax.set_ylim(0, bottom.max() * 1.15)
    fig.tight_layout()
    _savefig(fig, "fig_energy_breakdown.png")


def fig_co2_comparison(df: pd.DataFrame):
    """CO2 emissions comparison by difficulty."""
    if "emissions_kg_co2" not in df.columns:
        return
    _difficulty_grouped_bar(
        df, "emissions_kg_co2", r"CO$_2$ Emissions (kg)",
        r"CO$_2$ Emissions by Difficulty", "fig_co2.png", "%.7f",
    )


def fig_llm_calls(df: pd.DataFrame):
    _difficulty_grouped_bar(
        df, "num_llm_calls", "Avg LLM Calls",
        "LLM Invocations by Difficulty", "fig_llm_calls.png", "%.1f",
    )


def fig_model_usage_donut(df: pd.DataFrame):
    """Donut chart: model usage distribution in heterogeneous flow."""
    hetero = df[df.flow == "heterogeneous"]
    if hetero.empty:
        return

    all_models: list[str] = []
    for models_used in hetero["models_used"]:
        if isinstance(models_used, list):
            all_models.extend(models_used)
        elif isinstance(models_used, str):
            try:
                parsed = json.loads(models_used.replace("'", '"'))
                all_models.extend(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
    if not all_models:
        return

    counts = Counter(all_models)
    model_colors = {
        "qwen3.5:2b": "#2ECC71",
        "qwen3.5:4b": "#3498DB",
        "qwen3.5:9b": "#F39C12",
        "qwen3.5:27b-q4_K_M": "#E74C3C",
    }

    sorted_models = sorted(counts.keys(), key=lambda m: counts[m], reverse=True)
    labels = [f"{m}\n({counts[m]})" for m in sorted_models]
    sizes = [counts[m] for m in sorted_models]
    colors = [model_colors.get(m, "#95A5A6") for m in sorted_models]

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        textprops={"fontsize": 7}, pctdistance=0.78,
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 1.5},
        startangle=90,
    )
    for t in autotexts:
        t.set_fontsize(7)
        t.set_fontweight("bold")
    ax.set_title("Model Usage in Heterogeneous Flow", pad=12)
    fig.tight_layout()
    _savefig(fig, "fig_model_usage.png")


def fig_combined_energy_accuracy(df: pd.DataFrame):
    """Two-panel: energy and accuracy by difficulty, with proper spacing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_COL2)

    for ax, col, ylabel, title, fmt, ylim_mult in [
        (ax1, "energy_kwh", "Energy (kWh)", "Energy by Difficulty", "%.5f", 1.25),
        (ax2, "accuracy_score", "Accuracy", "Accuracy by Difficulty", "%.2f", None),
    ]:
        pivot = df.pivot_table(index="difficulty", columns="flow", values=col, aggfunc="mean")
        pivot = pivot.reindex(DIFF_ORDER).reindex(columns=["homogeneous", "heterogeneous"])
        x = np.arange(len(DIFF_ORDER))
        w = 0.32
        all_bars = []
        for i, flow in enumerate(["homogeneous", "heterogeneous"]):
            if flow not in pivot.columns:
                continue
            vals = pivot[flow].values
            bars = ax.bar(x + i * w - w / 2, vals, w,
                          label=FLOW_LABELS[flow], color=FLOW_COLORS[flow],
                          edgecolor="white", linewidth=0.8)
            _add_bar_labels(ax, bars, fmt=fmt, fontsize=5.5)
            all_bars.extend(bars)
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in DIFF_ORDER])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, framealpha=0.9, edgecolor="none")

        if ylim_mult:
            max_val = max(b.get_height() for b in all_bars if not np.isnan(b.get_height()))
            ax.set_ylim(0, max_val * ylim_mult)
        else:
            ax.set_ylim(0, 1.18)

    fig.tight_layout(w_pad=3)
    _savefig(fig, "fig_combined_energy_accuracy.png")


def fig_energy_per_llm_call(df: pd.DataFrame):
    """Energy per LLM call by difficulty and flow."""
    df_copy = df.copy()
    df_copy["energy_per_call"] = df_copy["energy_kwh"] / df_copy["num_llm_calls"].clip(lower=1)
    _difficulty_grouped_bar(
        df_copy, "energy_per_call", "Energy per LLM Call (kWh)",
        "Energy Efficiency per LLM Invocation", "fig_energy_per_call.png", "%.6f",
    )


def fig_combined_savings(df: pd.DataFrame):
    """Side-by-side bars: energy saving % and time saving % by difficulty."""
    e_savings, t_savings = {}, {}
    for diff in DIFF_ORDER:
        homo_e = df[(df.difficulty == diff) & (df.flow == "homogeneous")].energy_kwh.mean()
        het_e = df[(df.difficulty == diff) & (df.flow == "heterogeneous")].energy_kwh.mean()
        homo_t = df[(df.difficulty == diff) & (df.flow == "homogeneous")].duration_seconds.mean()
        het_t = df[(df.difficulty == diff) & (df.flow == "heterogeneous")].duration_seconds.mean()
        if homo_e > 0:
            e_savings[diff] = (1 - het_e / homo_e) * 100
        if homo_t > 0:
            t_savings[diff] = (1 - het_t / homo_t) * 100

    if not e_savings:
        return

    fig, ax = plt.subplots(figsize=_COL1)
    x = np.arange(len(DIFF_ORDER))
    w = 0.32
    e_vals = [e_savings.get(d, 0) for d in DIFF_ORDER]
    t_vals = [t_savings.get(d, 0) for d in DIFF_ORDER]

    bars1 = ax.bar(x - w / 2, e_vals, w, label="Energy Saving",
                   color="#2471A3", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + w / 2, t_vals, w, label="Time Saving",
                   color="#1ABC9C", edgecolor="white", linewidth=0.8)

    _add_bar_labels(ax, bars1, fmt="%+.0f%%", fontsize=6)
    _add_bar_labels(ax, bars2, fmt="%+.0f%%", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DIFF_ORDER])
    ax.set_ylabel("Saving (%)")
    ax.set_title("Energy & Time Savings by Difficulty")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    all_vals = e_vals + t_vals
    ax.set_ylim(0, max(all_vals) * 1.2)
    fig.tight_layout()
    _savefig(fig, "fig_combined_savings.png")


def fig_radar_comparison(df: pd.DataFrame):
    """Radar / spider chart comparing the two flows across key metrics."""
    metrics = {
        "Accuracy": "accuracy_score",
        "Energy Eff.": "energy_kwh",
        "Speed": "duration_seconds",
        "LLM Eff.": "num_llm_calls",
    }
    homo = df[df.flow == "homogeneous"]
    hetero = df[df.flow == "heterogeneous"]
    if homo.empty or hetero.empty:
        return

    homo_vals = []
    hetero_vals = []
    for label, col in metrics.items():
        hv = homo[col].mean()
        tv = hetero[col].mean()
        if label == "Accuracy":
            homo_vals.append(hv)
            hetero_vals.append(tv)
        else:
            # Invert so higher = better (lower energy/time/calls is better)
            max_val = max(hv, tv) * 1.1
            homo_vals.append(1 - hv / max_val if max_val > 0 else 0)
            hetero_vals.append(1 - tv / max_val if max_val > 0 else 0)

    labels = list(metrics.keys())
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    homo_vals += homo_vals[:1]
    hetero_vals += hetero_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw={"projection": "polar"})
    ax.fill(angles, homo_vals, alpha=0.15, color=FLOW_COLORS["homogeneous"])
    ax.plot(angles, homo_vals, "o-", color=FLOW_COLORS["homogeneous"],
            label=FLOW_LABELS["homogeneous"], linewidth=1.5, markersize=4)
    ax.fill(angles, hetero_vals, alpha=0.15, color=FLOW_COLORS["heterogeneous"])
    ax.plot(angles, hetero_vals, "o-", color=FLOW_COLORS["heterogeneous"],
            label=FLOW_LABELS["heterogeneous"], linewidth=1.5, markersize=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticklabels([])
    ax.set_title("Multi-Metric Comparison\n(higher = better)", fontsize=10, pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=7,
              framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    _savefig(fig, "fig_radar.png")


def fig_per_task_duration(df: pd.DataFrame):
    """Horizontal grouped bars: duration per task."""
    tasks_sorted = sorted(df.task_id.unique(), key=lambda t: (
        {"E": 0, "M": 1, "H": 2}.get(t[0], 3), int(t[1:])
    ))
    pivot = df.pivot_table(index="task_id", columns="flow", values="duration_seconds", aggfunc="mean")
    pivot = pivot.reindex(tasks_sorted).reindex(columns=["homogeneous", "heterogeneous"])

    fig, ax = plt.subplots(figsize=(4.5, 5.0))
    y = np.arange(len(tasks_sorted))
    h = 0.35
    for i, flow in enumerate(["homogeneous", "heterogeneous"]):
        if flow not in pivot.columns:
            continue
        vals = pivot[flow].values
        ax.barh(y + i * h - h / 2, vals, h,
                label=FLOW_LABELS[flow], color=FLOW_COLORS[flow],
                edgecolor="white", linewidth=0.5)

    for region, ys, color in [("Easy", (-0.5, 4.5), "#27AE6015"),
                               ("Medium", (4.5, 9.5), "#F39C1215"),
                               ("Hard", (9.5, 14.5), "#C0392B15")]:
        ax.axhspan(ys[0], ys[1], color=color, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(tasks_sorted, fontsize=7)
    ax.set_xlabel("Duration (s)")
    ax.set_title("Per-Task Execution Time Comparison")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9, edgecolor="none")
    ax.invert_yaxis()
    fig.tight_layout()
    _savefig(fig, "fig_per_task_duration.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════


def print_statistics(df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("  COMPREHENSIVE STATISTICS")
    print(f"{'='*70}")

    for flow in ["homogeneous", "heterogeneous"]:
        sub = df[df.flow == flow]
        if sub.empty:
            continue
        print(f"\n  {flow.upper()} (n={len(sub)}):")
        for col, label in [
            ("accuracy_score", "Accuracy"),
            ("energy_kwh", "Energy (kWh)"),
            ("duration_seconds", "Duration (s)"),
            ("ets", "EtS"),
            ("num_llm_calls", "LLM calls"),
            ("emissions_kg_co2", "CO2 (kg)"),
        ]:
            if col not in sub.columns:
                continue
            vals = sub[col]
            sem = sp_stats.sem(vals) if len(vals) > 1 else 0
            if len(vals) > 1 and sem > 0:
                ci = sp_stats.t.interval(0.95, len(vals) - 1, loc=vals.mean(), scale=sem)
            else:
                ci = (vals.mean(), vals.mean())
            print(f"    {label:20s}: mean={vals.mean():.6f}  std={vals.std():.6f}  "
                  f"95%CI=[{ci[0]:.6f}, {ci[1]:.6f}]")

    if {"homogeneous", "heterogeneous"}.issubset(set(df["flow"])):
        print(f"\n  ENERGY SAVINGS BY DIFFICULTY:")
        for diff in DIFF_ORDER:
            homo = df[(df.difficulty == diff) & (df.flow == "homogeneous")].energy_kwh
            hetero = df[(df.difficulty == diff) & (df.flow == "heterogeneous")].energy_kwh
            if homo.empty or hetero.empty:
                continue
            saving = (1 - hetero.mean() / homo.mean()) * 100
            print(f"    {diff:8s}: {saving:+.1f}%")

        print(f"\n  TIME SAVINGS BY DIFFICULTY:")
        for diff in DIFF_ORDER:
            homo = df[(df.difficulty == diff) & (df.flow == "homogeneous")].duration_seconds
            hetero = df[(df.difficulty == diff) & (df.flow == "heterogeneous")].duration_seconds
            if homo.empty or hetero.empty:
                continue
            saving = (1 - hetero.mean() / homo.mean()) * 100
            print(f"    {diff:8s}: {saving:+.1f}%")


def save_csv_exports(df: pd.DataFrame, out_dir: str):
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    pivot = df.pivot_table(
        index="task_id", columns="flow",
        values=["accuracy_score", "energy_kwh", "duration_seconds", "ets"],
        aggfunc=["mean", "std"],
    ).round(6)
    pivot.to_csv(os.path.join(csv_dir, "per_task_stats.csv"))

    diff_pivot = df.pivot_table(
        index="difficulty", columns="flow",
        values=["accuracy_score", "energy_kwh", "duration_seconds", "ets"],
        aggfunc=["mean", "std"],
    ).round(6)
    diff_pivot.to_csv(os.path.join(csv_dir, "per_difficulty_stats.csv"))

    overall = df.groupby("flow").agg(
        accuracy_mean=("accuracy_score", "mean"),
        accuracy_std=("accuracy_score", "std"),
        energy_mean=("energy_kwh", "mean"),
        energy_std=("energy_kwh", "std"),
        duration_mean=("duration_seconds", "mean"),
        duration_std=("duration_seconds", "std"),
        ets_mean=("ets", "mean"),
        ets_std=("ets", "std"),
        co2_mean=("emissions_kg_co2", "mean"),
        llm_calls_mean=("num_llm_calls", "mean"),
    ).round(8)
    overall.to_csv(os.path.join(csv_dir, "overall_stats.csv"))

    print(f"  CSV exports → {csv_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def run_analysis(results_path: str, out_dir: str | None = None):
    """Run the full analysis pipeline. Called by run_experiment or standalone."""
    global _FIG_DIR, _TAB_DIR

    base = out_dir or RESULTS_DIR
    _FIG_DIR = os.path.join(base, "figures")
    _TAB_DIR = os.path.join(base, "tables")
    os.makedirs(_FIG_DIR, exist_ok=True)
    os.makedirs(_TAB_DIR, exist_ok=True)

    df = load_results(results_path)
    n = len(df)
    print(f"\nLoaded {n} records from {results_path}")
    print(f"  Tasks: {sorted(df.task_id.unique())}")
    print(f"  Flows: {sorted(df.flow.unique())}")

    # ── LaTeX tables ──
    print(f"\n{'─'*50}")
    print("  Generating LaTeX tables…")
    print(f"{'─'*50}")
    table_experiment_setup(df)
    table_overall_comparison(df)
    table_per_difficulty(df)
    table_per_task(df)
    table_energy_savings_by_difficulty(df)
    table_model_usage(df)

    # ── Figures ──
    print(f"\n{'─'*50}")
    print("  Generating figures…")
    print(f"{'─'*50}")
    fig_energy_by_difficulty(df)
    fig_accuracy_by_difficulty(df)
    fig_duration_by_difficulty(df)
    fig_ets_by_difficulty(df)
    fig_energy_overall(df)
    fig_accuracy_overall(df)
    fig_ets_overall(df)
    fig_per_task_energy(df)
    fig_per_task_accuracy(df)
    fig_per_task_duration(df)
    fig_energy_savings_bar(df)
    fig_duration_savings_bar(df)
    fig_combined_savings(df)
    fig_tradeoff_scatter(df)
    fig_energy_breakdown(df)
    fig_co2_comparison(df)
    fig_llm_calls(df)
    fig_model_usage_donut(df)
    fig_combined_energy_accuracy(df)
    fig_energy_per_llm_call(df)
    fig_radar_comparison(df)

    # ── Statistics ──
    print_statistics(df)

    # ── CSV exports ──
    save_csv_exports(df, base)

    print(f"\n{'='*70}")
    print(f"  Analysis complete. Outputs in:")
    print(f"    Figures : {_FIG_DIR}/")
    print(f"    Tables  : {_TAB_DIR}/")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Analyse GAO experiment results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument(
        "--out-dir", default=None,
        help=f"Base output directory (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()

    run_analysis(args.results_file, args.out_dir)


if __name__ == "__main__":
    main()

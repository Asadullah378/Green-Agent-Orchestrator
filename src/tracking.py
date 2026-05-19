"""
Green Agent Orchestrator (GAO) — Energy & timing measurement

Wraps CodeCarbon's EmissionsTracker and wall-clock timing into a single
context manager that returns a structured result dict.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

from codecarbon import OfflineEmissionsTracker

from src.config import get_config


@dataclass
class TrackingResult:
    """Holds measurements for one tracked block."""
    energy_kwh: float = 0.0
    emissions_kg_co2: float = 0.0
    duration_seconds: float = 0.0
    cpu_energy_kwh: float = 0.0
    gpu_energy_kwh: float = 0.0
    ram_energy_kwh: float = 0.0
    cpu_power_w: float = 0.0
    gpu_power_w: float = 0.0
    ram_power_w: float = 0.0

    def to_dict(self) -> dict:
        return {
            "energy_kwh": self.energy_kwh,
            "emissions_kg_co2": self.emissions_kg_co2,
            "duration_seconds": self.duration_seconds,
            "cpu_energy_kwh": self.cpu_energy_kwh,
            "gpu_energy_kwh": self.gpu_energy_kwh,
            "ram_energy_kwh": self.ram_energy_kwh,
            "cpu_power_w": self.cpu_power_w,
            "gpu_power_w": self.gpu_power_w,
            "ram_power_w": self.ram_power_w,
        }


@dataclass
class TaskRecord:
    """Full record for one benchmark-task execution."""
    task_id: str = ""
    flow: str = ""  # "homogeneous" or "heterogeneous"
    run_idx: int = 0
    query: str = ""
    response: str = ""
    models_used: list[str] = field(default_factory=list)
    num_llm_calls: int = 0
    num_tool_calls: int = 0
    accuracy_score: float = 0.0
    tracking: TrackingResult = field(default_factory=TrackingResult)
    subtask_details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "flow": self.flow,
            "run_idx": self.run_idx,
            "query": self.query,
            "response": self.response,
            "models_used": self.models_used,
            "num_llm_calls": self.num_llm_calls,
            "num_tool_calls": self.num_tool_calls,
            "accuracy_score": self.accuracy_score,
            "subtask_details": self.subtask_details,
            **self.tracking.to_dict(),
        }


@contextmanager
def track_energy(label: str = "task") -> Generator[TrackingResult, None, None]:
    """Context manager that tracks energy via CodeCarbon and wall-clock time.

    Usage::

        with track_energy("my_task") as result:
            ... do work ...
        print(result.energy_kwh)
    """
    cfg = get_config()
    result = TrackingResult()
    # CodeCarbon's OfflineEmissionsTracker validates `output_dir` eagerly in
    # __init__ and raises OSError if it doesn't exist — even when
    # save_to_file=False. Create it on first use so the tracker can start.
    os.makedirs(cfg.experiment.results_dir, exist_ok=True)

    tracker: OfflineEmissionsTracker | None = None
    started = False
    try:
        tracker = OfflineEmissionsTracker(
            country_iso_code=cfg.energy.country_iso_code,
            log_level=cfg.energy.log_level,
            tracking_mode=cfg.energy.tracking_mode,
            output_dir=cfg.experiment.results_dir,
            project_name=label,
            save_to_file=False,
        )
        tracker.start()
        started = True
    except Exception as exc:  # noqa: BLE001
        # Don't let a misbehaving tracker take down the whole experiment;
        # the run still produces valid accuracy / timing data.
        print(
            f"  [tracking] WARNING: CodeCarbon failed to start for "
            f"'{label}': {exc.__class__.__name__}: {exc}"
        )

    t0 = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - t0
        result.duration_seconds = round(elapsed, 4)

        if started and tracker is not None:
            try:
                emissions = tracker.stop()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [tracking] WARNING: CodeCarbon stop() failed for "
                    f"'{label}': {exc.__class__.__name__}: {exc}"
                )
                emissions = None

            data = getattr(tracker, "final_emissions_data", None)
            if emissions is not None and data is not None:
                result.energy_kwh = data.energy_consumed or 0.0
                result.emissions_kg_co2 = emissions
                result.cpu_energy_kwh = data.cpu_energy or 0.0
                result.gpu_energy_kwh = data.gpu_energy or 0.0
                result.ram_energy_kwh = data.ram_energy or 0.0
                result.cpu_power_w = data.cpu_power or 0.0
                result.gpu_power_w = data.gpu_power or 0.0
                result.ram_power_w = data.ram_power or 0.0

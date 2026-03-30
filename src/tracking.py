"""
Green Agent Orchestrator (GAO) — Energy & timing measurement

Wraps CodeCarbon's EmissionsTracker and wall-clock timing into a single
context manager that returns a structured result dict.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

from codecarbon import OfflineEmissionsTracker

from src.config import CODECARBON_LOG_LEVEL, COUNTRY_ISO_CODE, RESULTS_DIR


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
    result = TrackingResult()
    tracker = OfflineEmissionsTracker(
        country_iso_code=COUNTRY_ISO_CODE,
        log_level=CODECARBON_LOG_LEVEL,
        tracking_mode="process",
        output_dir=RESULTS_DIR,
        project_name=label,
        save_to_file=False,
    )

    tracker.start()
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        emissions = tracker.stop()
        elapsed = time.perf_counter() - t0

        result.duration_seconds = round(elapsed, 4)
        if emissions is not None:
            result.energy_kwh = tracker.final_emissions_data.energy_consumed or 0.0
            result.emissions_kg_co2 = emissions
            result.cpu_energy_kwh = tracker.final_emissions_data.cpu_energy or 0.0
            result.gpu_energy_kwh = tracker.final_emissions_data.gpu_energy or 0.0
            result.ram_energy_kwh = tracker.final_emissions_data.ram_energy or 0.0
            result.cpu_power_w = tracker.final_emissions_data.cpu_power or 0.0
            result.gpu_power_w = tracker.final_emissions_data.gpu_power or 0.0
            result.ram_power_w = tracker.final_emissions_data.ram_power or 0.0

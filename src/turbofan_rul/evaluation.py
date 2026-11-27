"""Score the official cutoff once per engine and rank maintenance review."""

from dataclasses import dataclass

import numpy as np

from .dataset import Trajectory
from .features import WINDOW
from .model import LifeModel


@dataclass(frozen=True)
class Forecast:
    unit: int
    observed_cycles: int
    review_status: str
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None


def assess(model: LifeModel, engines: tuple[Trajectory, ...], horizon: int) -> list[Forecast]:
    forecasts = []
    for engine in engines:
        if engine.cycles < WINDOW:
            forecasts.append(Forecast(engine.unit, engine.cycles, "insufficient_history"))
            continue
        estimate, lower, upper = model.estimate(engine, engine.cycles)
        status = "review" if lower <= horizon else "monitor"
        forecasts.append(Forecast(engine.unit, engine.cycles, status, estimate, lower, upper))
    return sorted(forecasts, key=lambda item: (
        item.lower if item.lower is not None else float("inf"),
        item.estimate if item.estimate is not None else float("inf"),
        item.unit,
    ))


def _errors(actual, predicted):
    errors = predicted - actual
    return dict(
        mae_cycles=float(np.abs(errors).mean()),
        rmse_cycles=float(np.sqrt(np.mean(errors ** 2))),
    )


def score(model: LifeModel, forecasts: list[Forecast], actual_by_unit: dict, horizon: int) -> dict:
    if any(item.estimate is None for item in forecasts):
        raise ValueError("all official test engines must have enough history to score")
    actual = np.asarray([actual_by_unit[item.unit] for item in forecasts])
    point = np.asarray([item.estimate for item in forecasts])
    lower = np.asarray([item.lower for item in forecasts])
    upper = np.asarray([item.upper for item in forecasts])
    age = np.asarray([model.age_baseline(item.observed_cycles) for item in forecasts])
    reviewed = np.asarray([item.review_status == "review" for item in forecasts])
    due = actual <= horizon
    covered = (lower <= actual) & (actual <= upper)
    return dict(
        engines=len(forecasts),
        sensor_model=_errors(actual, point),
        age_baseline=_errors(actual, age),
        interval=dict(
            empirical_coverage=float(covered.mean()),
            mean_width_cycles=float((upper - lower).mean()),
        ),
        review=dict(
            horizon_cycles=horizon,
            flagged=int(reviewed.sum()),
            due_within_horizon=int(due.sum()),
            missed_due=int((due & ~reviewed).sum()),
            flagged_outside_horizon=int((~due & reviewed).sum()),
        ),
    )

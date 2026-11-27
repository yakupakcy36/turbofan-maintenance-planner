"""Score the official cutoff once per engine and rank maintenance review."""

from dataclasses import dataclass, replace

import numpy as np

from .dataset import Trajectory, supports_fd001_conditions
from .features import WINDOW
from .model import LifeModel


REVIEW_CANDIDATES = {"review", "deferred"}
BENCHMARK_CAPACITIES = (10, 25, 50)


@dataclass(frozen=True)
class Forecast:
    unit: int
    observed_cycles: int
    review_status: str
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None


def assess(
    model: LifeModel,
    engines: tuple[Trajectory, ...],
    horizon: int,
    *,
    review_limit: int | None = None,
) -> list[Forecast]:
    forecasts = []
    for engine in engines:
        if not supports_fd001_conditions(engine):
            forecasts.append(Forecast(engine.unit, engine.cycles, "unsupported_conditions"))
            continue
        if engine.cycles < WINDOW:
            forecasts.append(Forecast(engine.unit, engine.cycles, "insufficient_history"))
            continue
        estimate, lower, upper = model.estimate(engine, engine.cycles)
        status = "review" if lower <= horizon else "monitor"
        forecasts.append(Forecast(engine.unit, engine.cycles, status, estimate, lower, upper))
    forecasts.sort(key=lambda item: (
        item.lower if item.lower is not None else float("inf"),
        item.estimate if item.estimate is not None else float("inf"),
        item.unit,
    ))
    return _allocate_review(forecasts, review_limit)


def _allocate_review(forecasts: list[Forecast], limit: int | None) -> list[Forecast]:
    if limit is not None and limit < 0:
        raise ValueError("review limit must be non-negative")
    remaining = limit
    plan = []
    for item in forecasts:
        if item.review_status in REVIEW_CANDIDATES:
            status = "deferred" if remaining == 0 else "review"
            if remaining is not None and remaining > 0:
                remaining -= 1
            if item.review_status != status:
                item = replace(item, review_status=status)
        plan.append(item)
    return plan


def _errors(actual, predicted):
    errors = predicted - actual
    return dict(
        mae_cycles=float(np.abs(errors).mean()),
        rmse_cycles=float(np.sqrt(np.mean(errors ** 2))),
    )


def _review_counts(forecasts: list[Forecast], due: np.ndarray) -> dict:
    scheduled = np.asarray([item.review_status == "review" for item in forecasts])
    deferred = np.asarray([item.review_status == "deferred" for item in forecasts])
    flagged = scheduled | deferred
    return dict(
        flagged=int(flagged.sum()),
        scheduled=int(scheduled.sum()),
        deferred=int(deferred.sum()),
        due_within_horizon=int(due.sum()),
        due_scheduled=int((due & scheduled).sum()),
        due_deferred=int((due & deferred).sum()),
        missed_due=int((due & ~scheduled).sum()),
        flagged_outside_horizon=int((~due & flagged).sum()),
        scheduled_outside_horizon=int((~due & scheduled).sum()),
    )


def score(model: LifeModel, forecasts: list[Forecast], actual_by_unit: dict, horizon: int) -> dict:
    if any(item.estimate is None for item in forecasts):
        raise ValueError("all official test engines must meet FD001 conditions and history requirements")
    actual = np.asarray([actual_by_unit[item.unit] for item in forecasts])
    point = np.asarray([item.estimate for item in forecasts])
    lower = np.asarray([item.lower for item in forecasts])
    upper = np.asarray([item.upper for item in forecasts])
    age = np.asarray([model.age_baseline(item.observed_cycles) for item in forecasts])
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
        review=dict(horizon_cycles=horizon, **_review_counts(forecasts, due)),
        review_capacity_scenarios=[
            dict(limit=limit, **_review_counts(_allocate_review(forecasts, limit), due))
            for limit in BENCHMARK_CAPACITIES
        ],
    )

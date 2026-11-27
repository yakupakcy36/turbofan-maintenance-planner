"""Fit an engine-balanced regression and calibrate on held-out engines."""

from dataclasses import dataclass
import math

import numpy as np
from sklearn.linear_model import Ridge

from .dataset import Trajectory, supports_fd001_conditions
from .features import FEATURE_COUNT, WINDOW, features_at


TARGET_CAP = 125
TRAINING_STRIDE = 5
CALIBRATION_COVERAGE = 0.90
MODEL_SCHEMA = 1


class UnsupportedConditionsError(ValueError):
    """The observed operating settings fall outside the FD001 model scope."""


@dataclass(frozen=True)
class LifeModel:
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    residual_radius: float
    baseline_lifetime: float

    def point(self, engine: Trajectory, cycle: int) -> float:
        if not supports_fd001_conditions(engine, through_cycle=cycle):
            raise UnsupportedConditionsError(f"unit {engine.unit} has settings outside FD001")
        vector = features_at(engine, cycle)
        estimate = ((vector - self.center) / self.scale) @ self.coefficients + self.intercept
        return float(np.clip(estimate, 0, TARGET_CAP))

    def estimate(self, engine: Trajectory, cycle: int) -> tuple[float, float, float]:
        value = self.point(engine, cycle)
        return value, max(0.0, value - self.residual_radius), value + self.residual_radius

    def age_baseline(self, cycle: int) -> float:
        return float(np.clip(self.baseline_lifetime - cycle, 0, TARGET_CAP))

    def document(self) -> dict:
        return dict(
            schema=MODEL_SCHEMA,
            window=WINDOW,
            center=self.center.tolist(),
            scale=self.scale.tolist(),
            coefficients=self.coefficients.tolist(),
            intercept=self.intercept,
            residual_radius=self.residual_radius,
            baseline_lifetime=self.baseline_lifetime,
        )

    @classmethod
    def from_document(cls, value: dict) -> "LifeModel":
        try:
            if value["schema"] != MODEL_SCHEMA or value["window"] != WINDOW:
                raise ValueError("unsupported model schema or feature window")
            vectors = tuple(
                np.asarray(value[name], dtype=float)
                for name in ("center", "scale", "coefficients")
            )
            scalars = tuple(
                float(value[name]) for name in ("intercept", "residual_radius", "baseline_lifetime")
            )
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("invalid model document") from exc
        if any(v.shape != (FEATURE_COUNT,) or not np.isfinite(v).all() for v in vectors):
            raise ValueError("invalid model feature vectors")
        if (vectors[1] < 1e-10).any() or not np.isfinite(scalars).all():
            raise ValueError("invalid model scales or scalar values")
        if any(np.max(np.abs(v)) > 1e12 for v in vectors) or max(map(abs, scalars)) > 1e12:
            raise ValueError("model values outside the supported range")
        if scalars[1] < 0 or scalars[2] <= 0:
            raise ValueError("invalid interval radius or baseline lifetime")
        return cls(*vectors, *scalars)


@dataclass(frozen=True)
class TrainingResult:
    model: LifeModel
    fit_units: tuple[int, ...]
    calibration_cutoffs: dict[int, int]


def split_engines(histories):
    indices = np.random.default_rng(2026).permutation(len(histories))
    calibration_indices = set(indices[:len(histories) // 4])
    fitting = tuple(engine for i, engine in enumerate(histories) if i not in calibration_indices)
    calibration = tuple(engine for i, engine in enumerate(histories) if i in calibration_indices)
    return fitting, calibration


def conformal_radius(residuals) -> float:
    scores = np.sort(np.asarray(residuals, dtype=float))
    rank = math.ceil((len(scores) + 1) * CALIBRATION_COVERAGE)
    if rank > len(scores) or not np.isfinite(scores).all():
        raise ValueError("not enough finite calibration residuals for 90% coverage")
    return float(scores[rank - 1])


def fit(histories: tuple[Trajectory, ...]) -> TrainingResult:
    if len(histories) < 40 or len({engine.unit for engine in histories}) != len(histories):
        raise ValueError("training requires at least 40 distinct run-to-failure engines")
    if min(engine.cycles for engine in histories) <= WINDOW:
        raise ValueError(f"training trajectories need more than {WINDOW} cycles")

    fitting, calibration = split_engines(histories)
    rows, targets, weights = [], [], []
    for engine in fitting:
        cycles = list(range(WINDOW, engine.cycles, TRAINING_STRIDE)) + [engine.cycles]
        rows.extend(features_at(engine, cycle) for cycle in cycles)
        targets.extend(min(TARGET_CAP, engine.cycles - cycle) for cycle in cycles)
        weights.extend([1 / len(cycles)] * len(cycles))

    matrix = np.asarray(rows)
    center = np.average(matrix, axis=0, weights=weights)
    scale = np.sqrt(np.average((matrix - center) ** 2, axis=0, weights=weights))
    scale[scale < 1e-8] = 1.0
    learner = Ridge(alpha=1.0, solver="cholesky")
    learner.fit((matrix - center) / scale, targets, sample_weight=np.asarray(weights))
    baseline = float(np.median([engine.cycles for engine in fitting]))

    rng = np.random.default_rng(513)
    cutoffs = {engine.unit: int(rng.integers(WINDOW, engine.cycles)) for engine in calibration}
    point_model = LifeModel(center, scale, learner.coef_, float(learner.intercept_), 0.0, baseline)
    residuals = [
        abs(point_model.point(engine, cutoffs[engine.unit]) - (engine.cycles - cutoffs[engine.unit]))
        for engine in calibration
    ]
    model = LifeModel(center, scale, learner.coef_, point_model.intercept,
                      conformal_radius(residuals), baseline)
    return TrainingResult(model, tuple(engine.unit for engine in fitting), cutoffs)

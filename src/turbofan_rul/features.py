"""Features defined at the current cycle using only observations so far."""

import numpy as np

from .dataset import SENSOR_COUNT, Trajectory


WINDOW = 20
FEATURE_COUNT = 1 + 2 * SENSOR_COUNT


class InsufficientHistoryError(ValueError):
    """The requested cutoff cannot supply a complete feature window."""


def features_at(engine: Trajectory, cycle: int) -> np.ndarray:
    if not WINDOW <= cycle <= engine.cycles:
        raise InsufficientHistoryError(f"unit {engine.unit} needs at least {WINDOW} observed cycles")
    window = engine.sensors[cycle - WINDOW:cycle]
    recent, earlier = window[WINDOW // 2:], window[:WINDOW // 2]
    trend = recent.mean(axis=0) - earlier.mean(axis=0)
    return np.concatenate(([cycle], window.mean(axis=0), trend))

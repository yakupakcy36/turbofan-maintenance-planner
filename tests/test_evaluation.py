import unittest

import numpy as np

from turbofan_rul.dataset import SENSOR_COUNT, Trajectory
from turbofan_rul.evaluation import assess, score
from turbofan_rul.features import FEATURE_COUNT
from turbofan_rul.model import LifeModel


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        coefficients = np.zeros(FEATURE_COUNT)
        coefficients[0] = -1
        self.model = LifeModel(np.zeros(FEATURE_COUNT), np.ones(FEATURE_COUNT),
                               coefficients, 100, 10, 100)

    def test_rank_respects_the_lower_bound_and_marks_short_history(self):
        engines = tuple(Trajectory(i, np.zeros((cycles, SENSOR_COUNT)))
                        for i, cycles in enumerate((40, 80, 19), start=1))
        plans = assess(self.model, engines, 30)
        self.assertEqual([row.unit for row in plans], [2, 1, 3])
        self.assertEqual([row.review_status for row in plans],
                         ["review", "monitor", "insufficient_history"])

    def test_score_uses_outcomes_linked_by_engine_id(self):
        engines = tuple(Trajectory(i, np.zeros((cycles, SENSOR_COUNT)))
                        for i, cycles in enumerate((70, 80), start=1))
        forecasts = assess(self.model, engines, 20)
        report = score(self.model, forecasts, {1: 31, 2: 19}, 20)
        self.assertEqual(report["sensor_model"]["mae_cycles"], 1)
        self.assertEqual(report["interval"]["empirical_coverage"], 1)
        self.assertEqual(report["review"]["flagged_outside_horizon"], 1)
        self.assertEqual(report["review"]["missed_due"], 0)

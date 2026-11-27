import unittest

import numpy as np

from turbofan_rul.dataset import SENSOR_COUNT, Trajectory
from turbofan_rul.evaluation import assess, score
from turbofan_rul.features import FEATURE_COUNT
from turbofan_rul.model import LifeModel


def trajectory(unit, cycles):
    return Trajectory(unit, np.zeros((cycles, SENSOR_COUNT)),
                      np.tile([0.0, 0.0, 100.0], (cycles, 1)))


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        coefficients = np.zeros(FEATURE_COUNT)
        coefficients[0] = -1
        self.model = LifeModel(np.zeros(FEATURE_COUNT), np.ones(FEATURE_COUNT),
                               coefficients, 100, 10, 100)

    def test_rank_respects_the_lower_bound_and_marks_short_history(self):
        engines = tuple(trajectory(i, cycles)
                        for i, cycles in enumerate((40, 80, 19), start=1))
        plans = assess(self.model, engines, 30)
        self.assertEqual([row.unit for row in plans], [2, 1, 3])
        self.assertEqual([row.review_status for row in plans],
                         ["review", "monitor", "insufficient_history"])

    def test_score_uses_outcomes_linked_by_engine_id(self):
        engines = tuple(trajectory(i, cycles)
                        for i, cycles in enumerate((70, 80), start=1))
        forecasts = assess(self.model, engines, 20)
        report = score(self.model, forecasts, {1: 31, 2: 19}, 20)
        self.assertEqual(report["sensor_model"]["mae_cycles"], 1)
        self.assertEqual(report["interval"]["empirical_coverage"], 1)
        self.assertEqual(report["review"]["flagged_outside_horizon"], 1)
        self.assertEqual(report["review"]["missed_due"], 0)

    def test_review_limit_uses_stable_order_and_keeps_other_statuses(self):
        engines = tuple(trajectory(unit, cycles)
                        for unit, cycles in ((4, 19), (3, 40), (2, 80), (1, 80)))
        limited = assess(self.model, engines, 30, review_limit=1)
        self.assertEqual([row.unit for row in limited], [1, 2, 3, 4])
        self.assertEqual([row.review_status for row in limited],
                         ["review", "deferred", "monitor", "insufficient_history"])
        no_slots = assess(self.model, engines, 30, review_limit=0)
        self.assertEqual([row.review_status for row in no_slots],
                         ["deferred", "deferred", "monitor", "insufficient_history"])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            assess(self.model, engines, 30, review_limit=-1)

    def test_capacity_report_distinguishes_flags_from_reviewed_engines(self):
        engines = tuple(trajectory(i, cycles)
                        for i, cycles in enumerate((90, 80, 40), start=1))
        limited = assess(self.model, engines, 30, review_limit=1)
        report = score(self.model, limited, {1: 31, 2: 29, 3: 28}, 30)
        review = report["review"]
        self.assertEqual(review["flagged"], 2)
        self.assertEqual(review["scheduled"], 1)
        self.assertEqual(review["due_deferred"], 1)
        self.assertEqual(review["missed_due"], 2)
        expanded = report["review_capacity_scenarios"][0]
        self.assertEqual(expanded["limit"], 10)
        self.assertEqual(expanded["due_scheduled"], 1)
        self.assertEqual(expanded["missed_due"], 1)

    def test_unsupported_conditions_never_take_a_review_slot(self):
        supported = trajectory(2, 80)
        outside_fd001 = trajectory(1, 95)
        outside_fd001.operating_settings[0, 0] = 20.0

        ranked = assess(self.model, (outside_fd001, supported), 30, review_limit=1)
        self.assertEqual([(row.unit, row.review_status) for row in ranked],
                         [(2, "review"), (1, "unsupported_conditions")])
        self.assertEqual((ranked[1].estimate, ranked[1].lower, ranked[1].upper),
                         (None, None, None))

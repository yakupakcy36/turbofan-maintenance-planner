import unittest

import numpy as np

from turbofan_rul.dataset import SENSOR_COUNT, Trajectory
from turbofan_rul.features import FEATURE_COUNT, WINDOW, features_at
from turbofan_rul.model import LifeModel, conformal_radius, split_engines


class ModelTest(unittest.TestCase):
    def test_feature_snapshot_is_independent_of_future_rows(self):
        history = np.arange((WINDOW + 3) * SENSOR_COUNT).reshape(-1, SENSOR_COUNT)
        prefix = Trajectory(1, history[:WINDOW])
        extended = Trajectory(1, history)
        np.testing.assert_array_equal(features_at(prefix, WINDOW), features_at(extended, WINDOW))
        with self.assertRaisesRegex(ValueError, "at least 20 observed cycles"):
            features_at(prefix, WINDOW - 1)

    def test_partition_uses_disjoint_engines(self):
        engines = tuple(Trajectory(i, np.empty((0, SENSOR_COUNT))) for i in range(1, 101))
        fitting, calibration = split_engines(engines)
        self.assertEqual((len(fitting), len(calibration)), (75, 25))
        self.assertFalse(set(e.unit for e in fitting) & set(e.unit for e in calibration))
        self.assertEqual(set(e.unit for e in fitting + calibration), set(range(1, 101)))

    def test_radius_uses_finite_sample_rank(self):
        self.assertEqual(conformal_radius(range(1, 26)), 24)
        with self.assertRaisesRegex(ValueError, "not enough"):
            conformal_radius([1, 2])

    def test_model_round_trip_and_rejects_invalid_scale(self):
        model = LifeModel(np.zeros(FEATURE_COUNT), np.ones(FEATURE_COUNT),
                          np.ones(FEATURE_COUNT), 0, 10, 80)
        restored = LifeModel.from_document(model.document())
        engine = Trajectory(1, np.zeros((WINDOW, SENSOR_COUNT)))
        self.assertEqual(restored.estimate(engine, WINDOW), (20, 10, 30))
        invalid = model.document()
        invalid["scale"][0] = 0
        with self.assertRaisesRegex(ValueError, "invalid model scales"):
            LifeModel.from_document(invalid)

import unittest

from verify_benchmark import verify_review_outcomes


class BenchmarkGateTest(unittest.TestCase):
    def test_regressed_queue_fails_even_when_its_report_matches(self):
        labels = {unit: 31 if unit <= 25 else 30 for unit in range(1, 51)}
        fleet = [
            dict(unit=str(unit), actual_rul=str(actual),
                 review_status="review" if unit <= 25 else "deferred")
            for unit, actual in labels.items()
        ]
        plans = [
            dict(limit=limit, scheduled=limit, due_scheduled=max(0, limit - 25),
                 missed_due=min(25, 50 - limit), scheduled_outside_horizon=min(limit, 25))
            for limit in (10, 25, 50)
        ]
        report = {
            "review": {"horizon_cycles": 30, **plans[1]},
            "review_capacity_scenarios": plans,
        }
        with self.assertRaisesRegex(RuntimeError, "capacity 25 captured 0"):
            verify_review_outcomes(report, fleet, labels)

import unittest

from turbofan_rul.dataset import SENSOR_COUNT, parse_trajectories


def row(unit, cycle):
    return " ".join(map(str, [unit, cycle, 0, 0, 100, *range(SENSOR_COUNT)]))


class DatasetTest(unittest.TestCase):
    def test_parses_whitespace_and_keeps_separate_engine_histories(self):
        raw = ("\n".join((row(1, 1), row(1, 2), row(3, 1))) + "  \n").encode()
        histories = parse_trajectories(raw, "sample")
        self.assertEqual([(item.unit, item.cycles) for item in histories], [(1, 2), (3, 1)])
        self.assertEqual(histories[0].sensors[1, -1], 20.0)

    def test_rejects_a_gap_duplicate_or_engine_out_of_order(self):
        for records in ((row(1, 1), row(1, 3)), (row(1, 1), row(1, 1)),
                        (row(2, 1), row(1, 1))):
            with self.subTest(records=records):
                with self.assertRaisesRegex(ValueError, "consecutive cycles"):
                    parse_trajectories("\n".join(records).encode(), "sample")

    def test_rejects_an_incomplete_or_nonfinite_row(self):
        for raw in (b"1 1 0", row(1, 1).replace(" 20", " nan").encode()):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "26 finite columns"):
                    parse_trajectories(raw, "sample")

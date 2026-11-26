import unittest

from scoring import score
from replay import replay


class EvidenceTests(unittest.TestCase):
    def test_likely_step_and_boundary(self):
        self.assertTrue(score(False, [[(22, .5, -2)]])["peak_alarm"])
        self.assertFalse(score(False, [[(22, .5, -2.001)]])["combined"])
        with self.assertRaises(ValueError):
            score(False, [[(float("nan"), .5, -1)]])

    def test_chain_does_not_span_starts_or_gaps(self):
        step = (3, .4, -2)
        self.assertFalse(score(False, [[step, step], [step]])["chain"])
        self.assertFalse(score(False, [[step, (2, .4, -2), step]])["chain"])
        self.assertTrue(score(False, [[step] * 3])["chain"])

    def test_published_tables(self):
        d = replay()
        self.assertEqual((d["repair"]["valid"], d["repair"]["removed"]), (46, 46))
        self.assertEqual((d["detection"]["baseline"], d["detection"]["with_peak"]), (22, 27))
        self.assertEqual(dict(d["by_base"]["qwen3-06b"]), dict(valid=12, baseline=7, with_peak=12))
        self.assertEqual((d["clean"]["complete"], d["clean"]["combined_alarms"]), (6, 0))
        self.assertEqual(d["peak_gain_stress_test"]["still_above_11_without_best_start"], 3)


if __name__ == "__main__":
    unittest.main()

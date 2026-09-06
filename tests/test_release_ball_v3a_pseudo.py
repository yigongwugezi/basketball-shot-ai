import unittest

from scripts.make_release_ball_v3a_pseudo_labels import adjacent_agreement, agreement, temporal_support


class ReleaseBallV3APseudoTests(unittest.TestCase):
    def test_cross_model_agreement_keeps_best_pair(self):
        pair, score = agreement([{"bbox": [0, 0, 10, 10], "confidence": 0.8}], [{"bbox": [1, 1, 11, 11], "confidence": 0.7}])
        self.assertIsNotNone(pair)
        self.assertGreater(score, 0.3)

    def test_temporal_support_needs_an_adjacent_agreement(self):
        rows = [{"width": 100, "height": 100, "agreement": {"v2": {"bbox": [10, 10, 20, 20]}}}, {"width": 100, "height": 100, "agreement": {"v2": {"bbox": [12, 12, 22, 22]}}}, {"width": 100, "height": 100, "agreement": None}]
        self.assertTrue(temporal_support(rows, 0))
        self.assertFalse(temporal_support(rows, 2))
        self.assertTrue(adjacent_agreement(rows, 2))


if __name__ == "__main__":
    unittest.main()

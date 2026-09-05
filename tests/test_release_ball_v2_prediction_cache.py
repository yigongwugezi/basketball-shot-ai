from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest

from scripts.release_ball_v2_prediction_cache import flatten_frame_candidates, write_prediction_cache


class ReleaseBallV2PredictionCacheTests(unittest.TestCase):
    def test_preserves_every_candidate_and_geometry(self):
        rows = flatten_frame_candidates(12, 0.5, [{"bbox": [1, 2, 5, 8], "confidence": 0.7, "class": "ball"}, {"bbox": [10, 20, 14, 26], "confidence": 0.4, "class": "ball"}])
        self.assertEqual(2, len(rows))
        self.assertEqual([2, 2], [row["candidate_count"] for row in rows])
        self.assertEqual((3.0, 5.0), (rows[0]["center_x"], rows[0]["center_y"]))

    def test_writes_explicit_schema(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            write_prediction_cache(path, [(1, 0.0, [{"bbox": [0, 0, 2, 2], "confidence": 0.5, "class": 0}]), (2, 1 / 30, [])])
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(1, len(rows))
        self.assertEqual("1", rows[0]["candidate_count"])
        self.assertEqual("1.0", rows[0]["center_x"])


if __name__ == "__main__":
    unittest.main()

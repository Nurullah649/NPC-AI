from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_competition import _read_rows


class EvaluateCompetitionTests(unittest.TestCase):
    def test_sparse_keyframe_anchor_indices_are_allowed_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_index", "raw_x", "raw_y", "raw_z"])
                writer.writerow([0, 0, 0, 0])
                writer.writerow([37, 1, 2, 3])
            rows = _read_rows(path, require_contiguous=False)
            self.assertEqual([int(row["sample_index"]) for row in rows], [0, 37])
            with self.assertRaisesRegex(ValueError, "contiguous"):
                _read_rows(path)


if __name__ == "__main__":
    unittest.main()


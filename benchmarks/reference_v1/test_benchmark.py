from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from benchmark import _decode_release
from model_adapters import bbox_iou


def main() -> None:
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bbox_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0
    evidence = []
    for frame in range(8):
        distance = 1.5 if frame <= 3 else 3.0
        evidence.append(
            {
                "frame_index": frame,
                "ball_wrist_distance_diameters": distance,
                "ball_center": (float(frame * 3), 0.0),
            }
        )
    predicted, risks = _decode_release(evidence, 3)
    assert predicted == 4
    assert "experimental_geometry_proxy" in risks
    print("reference_v1 smoke tests passed")


if __name__ == "__main__":
    main()

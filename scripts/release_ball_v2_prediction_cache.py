"""Write detector evaluation records without discarding competing ball candidates."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


FIELDS = ("frame_index", "timestamp_s", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "center_x", "center_y", "confidence", "class", "candidate_count")


def flatten_frame_candidates(frame_index: int, timestamp_s: float | None, candidates: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    items = list(candidates)
    rows: list[dict[str, object]] = []
    for candidate in items:
        x1, y1, x2, y2 = (float(value) for value in candidate["bbox"])
        rows.append({"frame_index": frame_index, "timestamp_s": timestamp_s, "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2, "center_x": (x1 + x2) / 2, "center_y": (y1 + y2) / 2, "confidence": float(candidate["confidence"]), "class": candidate["class"], "candidate_count": len(items)})
    return rows


def write_prediction_cache(path: Path, frame_records: Iterable[tuple[int, float | None, Iterable[Mapping[str, object]]]]) -> None:
    rows = [row for frame_index, timestamp_s, candidates in frame_records for row in flatten_frame_candidates(frame_index, timestamp_s, candidates)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)

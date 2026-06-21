"""Build a single-class YOLO dataset for release-neighborhood basketball detection.

Inputs:
- datasets/annotations/release_ball_batch_001/labels.csv
- datasets/annotations/release_ball_batch_003/labels.csv
- tmp/release_ball_annotation_batch_001/<clip_id>/frames/
- tmp/release_ball_annotation_batch_003/<clip_id>/frames/

Output:
- datasets/processed/yolo_release_ball/
"""

from __future__ import annotations

import csv
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "datasets" / "processed" / "yolo_release_ball"

FIELDNAMES = [
    "source_batch",
    "clip_id",
    "frame_index",
    "original_image_file",
    "yolo_image_file",
    "split",
    "ball_visible",
    "is_release_pose_frame",
    "is_ball_release_frame",
    "label_confidence",
    "ball_visibility_quality",
    "occlusion",
    "notes",
]

SPLIT_BY_CLIP = {
    "BILI_003_A": "val",
    "NEW_012": "val",
}

SOURCE_BATCHES = {
    "release_ball_batch_001": {
        "labels_csv": ROOT / "datasets" / "annotations" / "release_ball_batch_001" / "labels.csv",
        "frames_root": ROOT / "tmp" / "release_ball_annotation_batch_001",
    },
    "release_ball_batch_003": {
        "labels_csv": ROOT / "datasets" / "annotations" / "release_ball_batch_003" / "labels.csv",
        "frames_root": ROOT / "tmp" / "release_ball_annotation_batch_003",
    },
}


@dataclass(frozen=True)
class RowRef:
    source_batch: str
    row: dict[str, str]


def reset_output() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val"):
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def load_rows() -> list[RowRef]:
    loaded: list[RowRef] = []
    for source_batch, cfg in SOURCE_BATCHES.items():
        with cfg["labels_csv"].open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                loaded.append(RowRef(source_batch=source_batch, row=dict(row)))
    return loaded


def split_for_clip(clip_id: str) -> str:
    return SPLIT_BY_CLIP.get(clip_id, "train")


def image_source_path(ref: RowRef) -> Path:
    clip_id = ref.row["clip_id"]
    image_file = ref.row["image_file"].replace("/", "\\")
    return SOURCE_BATCHES[ref.source_batch]["frames_root"] / clip_id / image_file


def parse_float(value: str, field: str, clip_id: str, frame_index: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{clip_id} frame {frame_index}: invalid {field}={value!r}") from exc


def validate_and_build_label(
    ref: RowRef,
    width: int,
    height: int,
) -> str:
    row = ref.row
    clip_id = row["clip_id"]
    frame_index = row["frame_index"]
    ball_visible = row["ball_visible"].strip().lower()
    if ball_visible == "no":
        for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2", "ball_center_x", "ball_center_y"):
            if row[field].strip():
                raise ValueError(
                    f"{clip_id} frame {frame_index}: ball_visible=no but {field} is present"
                )
        return ""

    if ball_visible != "yes":
        raise ValueError(f"{clip_id} frame {frame_index}: unexpected ball_visible={ball_visible!r}")

    x1 = parse_float(row["ball_x1"], "ball_x1", clip_id, frame_index)
    y1 = parse_float(row["ball_y1"], "ball_y1", clip_id, frame_index)
    x2 = parse_float(row["ball_x2"], "ball_x2", clip_id, frame_index)
    y2 = parse_float(row["ball_y2"], "ball_y2", clip_id, frame_index)
    cx = parse_float(row["ball_center_x"], "ball_center_x", clip_id, frame_index)
    cy = parse_float(row["ball_center_y"], "ball_center_y", clip_id, frame_index)

    if min(x1, y1, x2, y2) < 0:
        raise ValueError(f"{clip_id} frame {frame_index}: negative bbox coordinate")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{clip_id} frame {frame_index}: invalid bbox geometry")
    if x2 > width or y2 > height:
        raise ValueError(
            f"{clip_id} frame {frame_index}: bbox out of bounds for image {width}x{height}"
        )

    expected_cx = (x1 + x2) / 2.0
    expected_cy = (y1 + y2) / 2.0
    if abs(cx - expected_cx) > 1.0 or abs(cy - expected_cy) > 1.0:
        raise ValueError(
            f"{clip_id} frame {frame_index}: center mismatch "
            f"({cx}, {cy}) vs ({expected_cx}, {expected_cy})"
        )

    x_center = cx / width
    y_center = cy / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    values = [x_center, y_center, box_width, box_height]
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"{clip_id} frame {frame_index}: YOLO coordinates out of range {values}")

    return " ".join(["0", *[f"{value:.6f}" for value in values]])


def write_yaml() -> None:
    yaml = "\n".join(
        [
            f"path: {OUT_ROOT.as_posix()}",
            "train: train/images",
            "val: val/images",
            "nc: 1",
            "names:",
            "  0: ball",
            "",
        ]
    )
    (OUT_ROOT / "data.yaml").write_text(yaml, encoding="utf-8")


def main() -> None:
    reset_output()
    rows = load_rows()

    missing_images: list[str] = []
    metadata_rows: list[dict[str, str]] = []
    split_counts = {
        "train": Counter(),
        "val": Counter(),
    }
    clip_counts: dict[str, int] = defaultdict(int)
    split_clip_membership: dict[str, set[str]] = defaultdict(set)

    for ref in rows:
        src_path = image_source_path(ref)
        if not src_path.exists():
            missing_images.append(str(src_path))
            continue

        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            missing_images.append(f"unreadable image: {src_path}")
            continue

        clip_id = ref.row["clip_id"]
        frame_index = int(ref.row["frame_index"])
        split = split_for_clip(clip_id)
        split_clip_membership[clip_id].add(split)

        suffix = src_path.suffix.lower() or ".jpg"
        yolo_stem = f"{ref.source_batch}__{clip_id}__{frame_index:04d}"
        yolo_image_name = f"{yolo_stem}{suffix}"
        yolo_label_name = f"{yolo_stem}.txt"

        out_image_path = OUT_ROOT / split / "images" / yolo_image_name
        out_label_path = OUT_ROOT / split / "labels" / yolo_label_name

        label_line = validate_and_build_label(ref, width=img.shape[1], height=img.shape[0])

        shutil.copy2(src_path, out_image_path)
        out_label_path.write_text(label_line + ("\n" if label_line else ""), encoding="utf-8")

        ball_visible = ref.row["ball_visible"].strip().lower()
        split_counts[split]["images"] += 1
        split_counts[split]["positive" if ball_visible == "yes" else "negative"] += 1
        clip_counts[clip_id] += 1

        metadata_rows.append(
            {
                "source_batch": ref.source_batch,
                "clip_id": clip_id,
                "frame_index": str(frame_index),
                "original_image_file": ref.row["image_file"],
                "yolo_image_file": f"{split}/images/{yolo_image_name}",
                "split": split,
                "ball_visible": ref.row["ball_visible"],
                "is_release_pose_frame": ref.row["is_release_pose_frame"],
                "is_ball_release_frame": ref.row["is_ball_release_frame"],
                "label_confidence": ref.row["label_confidence"],
                "ball_visibility_quality": ref.row["ball_visibility_quality"],
                "occlusion": ref.row["occlusion"],
                "notes": ref.row["notes"],
            }
        )

    if missing_images:
        preview = "\n".join(missing_images[:20])
        raise FileNotFoundError(
            f"Missing or unreadable source images ({len(missing_images)} total):\n{preview}"
        )

    metadata_path = OUT_ROOT / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metadata_rows)

    write_yaml()
    run_checks(metadata_rows, split_clip_membership)

    total_images = len(metadata_rows)
    total_positive = sum(1 for row in metadata_rows if row["ball_visible"].strip().lower() == "yes")
    total_negative = sum(1 for row in metadata_rows if row["ball_visible"].strip().lower() == "no")

    print(f"total_images={total_images}")
    print(f"train_images={split_counts['train']['images']}")
    print(f"val_images={split_counts['val']['images']}")
    print(f"positive_images={total_positive}")
    print(f"negative_images={total_negative}")
    print(
        "train_distribution="
        f"positive:{split_counts['train']['positive']} negative:{split_counts['train']['negative']}"
    )
    print(
        "val_distribution="
        f"positive:{split_counts['val']['positive']} negative:{split_counts['val']['negative']}"
    )
    for clip_id in sorted(clip_counts):
        print(f"clip_rows[{clip_id}]={clip_counts[clip_id]}")
    print(f"metadata_rows={len(metadata_rows)}")


def run_checks(metadata_rows: list[dict[str, str]], split_clip_membership: dict[str, set[str]]) -> None:
    images_total = 0
    labels_total = 0
    empty_label_ok = 0
    non_empty_label_ok = 0
    val_clips = set()

    for split in ("train", "val"):
        image_dir = OUT_ROOT / split / "images"
        label_dir = OUT_ROOT / split / "labels"
        image_files = sorted(image_dir.iterdir())
        label_files = sorted(label_dir.iterdir())
        images_total += len(image_files)
        labels_total += len(label_files)
        if len(image_files) != len(label_files):
            raise RuntimeError(f"{split}: image/label count mismatch")

    for row in metadata_rows:
        image_path = OUT_ROOT / row["yolo_image_file"]
        label_path = image_path.with_suffix(".txt").parents[1] / "labels" / f"{image_path.stem}.txt"
        if not image_path.exists():
            raise RuntimeError(f"Missing generated image: {image_path}")
        if not label_path.exists():
            raise RuntimeError(f"Missing generated label: {label_path}")

        label_text = label_path.read_text(encoding="utf-8").strip()
        ball_visible = row["ball_visible"].strip().lower()
        if not label_text:
            if ball_visible != "no":
                raise RuntimeError(f"Empty label file for positive frame: {image_path.name}")
            empty_label_ok += 1
        else:
            if ball_visible != "yes":
                raise RuntimeError(f"Non-empty label file for negative frame: {image_path.name}")
            parts = label_text.split()
            if len(parts) != 5 or parts[0] != "0":
                raise RuntimeError(f"Invalid YOLO label line in {label_path}")
            coords = [float(value) for value in parts[1:]]
            if not all(0.0 <= value <= 1.0 for value in coords):
                raise RuntimeError(f"YOLO coordinates out of range in {label_path}")
            non_empty_label_ok += 1

        if row["split"] == "val":
            val_clips.add(row["clip_id"])

    if len(metadata_rows) != images_total or len(metadata_rows) != labels_total:
        raise RuntimeError("metadata/image/label totals do not match")

    if val_clips != {"BILI_003_A", "NEW_012"}:
        raise RuntimeError(f"Unexpected val clips: {sorted(val_clips)}")

    leaked = {clip_id: sorted(splits) for clip_id, splits in split_clip_membership.items() if len(splits) > 1}
    if leaked:
        raise RuntimeError(f"Clip leakage across splits: {leaked}")

    if empty_label_ok + non_empty_label_ok != len(metadata_rows):
        raise RuntimeError("label validation accounting mismatch")


if __name__ == "__main__":
    main()

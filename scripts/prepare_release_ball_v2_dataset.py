"""Build the frozen, leak-free release_ball_v2 YOLO dataset.

This deliberately consumes only manually reviewed batch001 and batch003 labels.
The R-CAM2 localization frames from IMG_7221.MOV and IMG_7222.MP4 are excluded
at source-video level; auto-assisted batch002 material is never an input.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_TMP = Path(r"C:\Users\20825\Documents\Codex\2026-05-21\1-2-ai-app-3-ai\tmp")
OUT_ROOT = ROOT / "datasets" / "processed" / "yolo_release_ball_v2"
MODEL_ROOT = ROOT / "models" / "release_ball_v2"

# This is an operational split, frozen before training.  Splitting is by source
# clip, never individual frames.
SPLIT_BY_CLIP = {"BILI_003_A": "val", "NEW_012": "val"}
RCAM2_EXCLUDED_SOURCE_VIDEOS = {"IMG_7221.MOV", "IMG_7222.MP4"}

SOURCE_BATCHES = {
    "release_ball_batch_001": {
        "labels": ROOT / "datasets" / "annotations" / "release_ball_batch_001" / "labels.csv",
        "frames": HISTORICAL_TMP / "release_ball_annotation_batch_001",
        "source_videos": {
            "BILI_001_A": "BILI_001_A_BV14u411J7qS.mp4",
            "BILI_003_A": "BILI_003_A_BV1d84y1G7zq.mp4",
            "BILI_005_A": "BILI_005_A_BV1Re4y1K7Ey.mp4",
        },
    },
    "release_ball_batch_003": {
        "labels": ROOT / "datasets" / "annotations" / "release_ball_batch_003" / "labels.csv",
        "frames": HISTORICAL_TMP / "release_ball_annotation_batch_003",
        "source_videos": {
            "NEW_001": "IMG_7212.MOV", "NEW_002": "IMG_7215.MOV", "NEW_003": "IMG_7216.MOV",
            "NEW_004": "IMG_7218.MOV", "NEW_005": "IMG_7219.MP4", "NEW_006": "IMG_7221.MOV",
            "NEW_009": "IMG_7226.MP4", "NEW_010": "IMG_7227.MP4", "NEW_012": "IMG_7235.MOV",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def yolo_label(row: dict[str, str], width: int, height: int) -> str:
    if row["ball_visible"].lower() == "no":
        if any(row[field].strip() for field in ("ball_x1", "ball_y1", "ball_x2", "ball_y2")):
            raise ValueError(f"negative frame has bbox: {row['clip_id']}:{row['frame_index']}")
        return ""
    values = {key: float(row[key]) for key in ("ball_x1", "ball_y1", "ball_x2", "ball_y2")}
    if not (0 <= values["ball_x1"] < values["ball_x2"] <= width and 0 <= values["ball_y1"] < values["ball_y2"] <= height):
        raise ValueError(f"invalid bbox: {row['clip_id']}:{row['frame_index']}")
    cx = (values["ball_x1"] + values["ball_x2"]) / 2 / width
    cy = (values["ball_y1"] + values["ball_y2"]) / 2 / height
    bw = (values["ball_x2"] - values["ball_x1"]) / width
    bh = (values["ball_y2"] - values["ball_y1"]) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen dataset: {OUT_ROOT}")
    rows: list[tuple[str, dict[str, str], Path, str]] = []
    exclusions: list[dict[str, str]] = []
    for batch_name, config in SOURCE_BATCHES.items():
        with config["labels"].open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                clip_id = row["clip_id"]
                source_video = config["source_videos"][clip_id]
                if source_video in RCAM2_EXCLUDED_SOURCE_VIDEOS:
                    exclusions.append({"source_batch": batch_name, "clip_id": clip_id, "source_video": source_video, "reason": "R-CAM2 localization validation source-video exclusion"})
                    continue
                image = config["frames"] / clip_id / row["image_file"]
                if not image.is_file():
                    raise FileNotFoundError(image)
                rows.append((batch_name, row, image, source_video))

    if {item["source_video"] for item in exclusions} != {"IMG_7221.MOV"} or len(exclusions) != 31:
        raise RuntimeError(f"unexpected R-CAM2 exclusion audit: {exclusions}")
    if any(video in RCAM2_EXCLUDED_SOURCE_VIDEOS for _, _, _, video in rows):
        raise RuntimeError("R-CAM2 source leaked into training dataset")

    for split in ("train", "val"):
        (OUT_ROOT / split / "images").mkdir(parents=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True)
    metadata: list[dict[str, str]] = []
    for batch_name, row, image, source_video in rows:
        split = SPLIT_BY_CLIP.get(row["clip_id"], "train")
        frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"unreadable image: {image}")
        stem = f"{batch_name}__{row['clip_id']}__{int(row['frame_index']):04d}"
        destination = OUT_ROOT / split / "images" / f"{stem}.jpg"
        shutil.copy2(image, destination)
        (OUT_ROOT / split / "labels" / f"{stem}.txt").write_text(yolo_label(row, frame.shape[1], frame.shape[0]) + "\n", encoding="utf-8")
        metadata.append({"source_batch": batch_name, "clip_id": row["clip_id"], "source_video": source_video, "frame_index": row["frame_index"], "split": split, "ball_visible": row["ball_visible"], "image": str(destination.relative_to(OUT_ROOT)).replace("\\", "/")})

    with (OUT_ROOT / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader(); writer.writerows(metadata)
    (OUT_ROOT / "data.yaml").write_text(f"path: {OUT_ROOT.as_posix()}\ntrain: train/images\nval: val/images\nnc: 1\nnames:\n  0: ball\n", encoding="utf-8")
    counts = {split: sum(item["split"] == split for item in metadata) for split in ("train", "val")}
    manifest = {
        "version": "release_ball_v2", "purpose": "detector training only; R-CAM2 validation is excluded",
        "architecture": "YOLO11n", "image_size": 640, "seed": 0, "evaluation_confidence_threshold": 0.15,
        "training_args": {"epochs": 100, "batch": 16, "optimizer": "auto", "pretrained": True},
        "dataset": {"manual_label_batches": ["release_ball_batch_001", "release_ball_batch_003"], "rows": len(metadata), "split_counts": counts, "split_by_clip": SPLIT_BY_CLIP, "metadata_sha256": sha256(OUT_ROOT / "metadata.csv")},
        "rcam2_exclusion": {"excluded_source_videos": sorted(RCAM2_EXCLUDED_SOURCE_VIDEOS), "excluded_rows": len(exclusions), "excluded_rows_detail": exclusions[:1], "rule": "exclude every training frame from a R-CAM2 validation source video"},
    }
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    (MODEL_ROOT / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = ["# RELEASE_BALL_V2_TRAINING_MANIFEST", "", "Frozen before training; do not use it to tune against R-CAM2.", "", "## Configuration", "", "- Architecture: YOLO11n", "- Image size: 640", "- Seed: 0", "- Evaluation confidence threshold: 0.15", "- Training: 100 epochs, batch 16, optimizer auto, pretrained initialization", "", "## Dataset and split", "", f"- Manual reviewed rows: {len(metadata)} (train {counts['train']}, val {counts['val']})", "- Inputs: release_ball_batch_001 and release_ball_batch_003 historical reviewed extracts", "- Validation clips: BILI_003_A, NEW_012", "- Training clips are source-disjoint from validation clips.", "", "## R-CAM2 exclusion", "", "- Excluded whole source videos: IMG_7221.MOV and IMG_7222.MP4.", "- IMG_7221.MOV occurred as NEW_006 and all 31 of its labeled frames were removed.", "- IMG_7222.MP4 had no rows in the formal label inputs.", "- Auto-assisted batch002 frames are not training inputs.", "", f"- Dataset metadata SHA256: `{manifest['dataset']['metadata_sha256']}`", ""]
    (MODEL_ROOT / "RELEASE_BALL_V2_TRAINING_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

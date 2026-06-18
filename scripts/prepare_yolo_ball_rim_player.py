"""Create a unified YOLO dataset for ball/rim/player detection.

Input datasets:
- datasets/raw/public/roboflow/basketball-and-hoop-detection
- datasets/raw/public/roboflow/basketball-hoop-ball-and-player

Output:
- datasets/processed/yolo_ball_rim_player

Unified classes:
0 ball
1 rim
2 player
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "datasets" / "raw" / "public" / "roboflow"
OUT_ROOT = ROOT / "datasets" / "processed" / "yolo_ball_rim_player"

SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNIFIED_NAMES = ["ball", "rim", "player"]


@dataclass(frozen=True)
class SourceDataset:
    name: str
    root: Path
    class_map: dict[int, int]


SOURCES = [
    SourceDataset(
        name="bahd",
        root=RAW_ROOT / "basketball-and-hoop-detection",
        class_map={
            1: 0,  # Basketball -> ball
            2: 1,  # Net -> rim-ish target area
        },
    ),
    SourceDataset(
        name="bhbp",
        root=RAW_ROOT / "basketball-hoop-ball-and-player",
        class_map={
            1: 0,  # ball -> ball
            3: 1,  # hoop -> rim
            6: 2,  # player -> player
        },
    ),
]


def reset_output() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in SPLITS:
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def convert_label_line(line: str, class_map: dict[int, int]) -> str | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        source_class = int(float(parts[0]))
    except ValueError:
        return None
    if source_class not in class_map:
        return None
    return " ".join([str(class_map[source_class]), *parts[1:]])


def process_source(source: SourceDataset) -> dict[str, int]:
    stats = {
        "images": 0,
        "labels": 0,
        "boxes": 0,
        "empty_labels": 0,
    }

    for split in SPLITS:
        image_dir = source.root / split / "images"
        label_dir = source.root / split / "labels"
        if not image_dir.exists():
            continue

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            stem = f"{source.name}_{image_path.stem}"
            output_image = OUT_ROOT / split / "images" / f"{stem}{image_path.suffix.lower()}"
            output_label = OUT_ROOT / split / "labels" / f"{stem}.txt"
            source_label = label_dir / f"{image_path.stem}.txt"

            converted: list[str] = []
            if source_label.exists():
                for line in source_label.read_text(encoding="utf-8").splitlines():
                    new_line = convert_label_line(line, source.class_map)
                    if new_line is not None:
                        converted.append(new_line)

            shutil.copy2(image_path, output_image)
            output_label.write_text("\n".join(converted), encoding="utf-8")

            stats["images"] += 1
            stats["labels"] += 1
            stats["boxes"] += len(converted)
            if not converted:
                stats["empty_labels"] += 1

    return stats


def write_yaml() -> None:
    yaml = "\n".join(
        [
            f"path: {OUT_ROOT.as_posix()}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            f"nc: {len(UNIFIED_NAMES)}",
            f"names: {UNIFIED_NAMES}",
            "",
        ]
    )
    (OUT_ROOT / "data.yaml").write_text(yaml, encoding="utf-8")


def main() -> None:
    reset_output()
    total = {"images": 0, "labels": 0, "boxes": 0, "empty_labels": 0}
    per_source = {}

    for source in SOURCES:
        stats = process_source(source)
        per_source[source.name] = stats
        for key in total:
            total[key] += stats[key]

    write_yaml()

    print("Unified dataset created:", OUT_ROOT)
    print("Classes:", UNIFIED_NAMES)
    print("Per source:", per_source)
    print("Total:", total)


if __name__ == "__main__":
    main()

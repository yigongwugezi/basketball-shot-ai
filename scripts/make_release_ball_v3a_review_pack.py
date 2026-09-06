#!/usr/bin/env python3
"""Build a read-only visual review pack from frozen V3A pseudo-label evidence."""

from __future__ import annotations

import csv
import html
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "datasets/annotations/release_ball_v3a_pseudo/pseudo_labels.json"
CLIP_MANIFEST_PATH = REPO_ROOT / "datasets/bilibili_clip_manifest.csv"
OUTPUT_ROOT = REPO_ROOT / "artifacts/release_ball_v3a_pseudo_review_final"
THUMBNAIL_WIDTH = 320
CONTACT_SHEET_COLUMNS = 4

STATUS_LABELS = {
    "pseudo_positive": "pseudo-positive",
    "unresolved": "unresolved",
    # Pseudo-negatives stay withheld from labels; the reviewer sees the requested
    # three-state vocabulary rather than an inferred training label.
    "pseudo_negative": "unlabeled",
}
COLORS = {
    "v2": (255, 155, 0),
    "coco": (0, 190, 0),
    "final": (30, 30, 230),
}


def load_source_names() -> dict[str, str]:
    names: dict[str, str] = {}
    with CLIP_MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = row.get("file_name") or row.get("clip_file_path")
            if not source:
                continue
            for clip_id in ("BILI_001_C", "BILI_002_A", "BILI_006_A", "BILI_008_A", "BILI_010_A", "BILI_010_B", "BILI_010_D"):
                if source.startswith(f"{clip_id}_"):
                    names[clip_id] = source
    return names


def image_relpath(row: dict) -> Path:
    return Path(row["clip_id"]) / Path(row["image_path"]).name


def draw_box(image: np.ndarray, bbox: list[float], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = (round(value) for value in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text_y = max(18, y1 - 6)
    cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def candidates(row: dict, model: str) -> list[dict]:
    field = "v2_candidates" if model == "v2" else "teacher_sports_ball_candidates"
    raw = row.get(field, [])
    return raw if isinstance(raw, list) else []


def source_summary(row: dict) -> str:
    if row["label_type"] == "pseudo_positive":
        return "V2 + COCO + cross-model agreement + temporal support"
    v2_count = len(candidates(row, "v2"))
    coco_count = len(candidates(row, "teacher"))
    if v2_count and coco_count:
        return "V2 + COCO (not accepted)"
    if v2_count:
        return "V2 only"
    if coco_count:
        return "COCO only"
    return "no detector bbox"


def confidence_summary(row: dict) -> str:
    values: list[str] = []
    for label, model in (("V2", "v2"), ("COCO", "teacher")):
        confidences = [candidate.get("confidence") for candidate in candidates(row, model)]
        confidences = [value for value in confidences if isinstance(value, (int, float))]
        if confidences:
            values.append(f"{label} {max(confidences):.3f}")
    return " | ".join(values) if values else "n/a"


def annotate(row: dict, source_name: str, original: np.ndarray) -> np.ndarray:
    image = original.copy()
    status = row["label_type"]
    if status == "pseudo_positive" and row.get("bbox"):
        draw_box(image, row["bbox"], COLORS["final"], f"BALL / {confidence_summary(row)}")
    else:
        # Unresolved frames retain every available detector candidate for human review.
        for candidate in candidates(row, "v2"):
            if candidate.get("bbox"):
                draw_box(image, candidate["bbox"], COLORS["v2"], f"V2 {candidate.get('confidence', 0):.3f}")
        for candidate in candidates(row, "teacher"):
            if candidate.get("bbox"):
                draw_box(image, candidate["bbox"], COLORS["coco"], f"COCO {candidate.get('confidence', 0):.3f}")

    lines = [
        f"{STATUS_LABELS[status]} | {row['clip_id']} | frame {row['frame_index']}",
        f"source: {source_name}",
        f"bbox evidence: {source_summary(row)}",
    ]
    scale = max(0.45, min(0.72, image.shape[1] / 1600))
    line_height = round(26 * scale)
    panel_height = line_height * len(lines) + 18
    cv2.rectangle(image, (0, 0), (image.shape[1], panel_height), (18, 18, 18), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (10, 18 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return image


def thumbnail(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scaled_height = round(height * THUMBNAIL_WIDTH / width)
    return cv2.resize(image, (THUMBNAIL_WIDTH, scaled_height), interpolation=cv2.INTER_AREA)


def create_contact_sheet(images: list[np.ndarray], destination: Path) -> None:
    thumbs = [thumbnail(image) for image in images]
    tile_height = max(image.shape[0] for image in thumbs)
    rows = (len(thumbs) + CONTACT_SHEET_COLUMNS - 1) // CONTACT_SHEET_COLUMNS
    sheet = np.full((rows * tile_height, CONTACT_SHEET_COLUMNS * THUMBNAIL_WIDTH, 3), 248, dtype=np.uint8)
    for index, image in enumerate(thumbs):
        row, column = divmod(index, CONTACT_SHEET_COLUMNS)
        y, x = row * tile_height, column * THUMBNAIL_WIDTH
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    cv2.imwrite(str(destination), sheet)


def page_header(title: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
a {{ color: #075985; }} .nav {{ margin-bottom: 20px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:16px; }}
.card {{ background:white; border:1px solid #cbd5e1; border-radius:8px; padding:10px; }}
.card img {{ display:block; width:100%; height:auto; }} .meta {{ font-size:13px; line-height:1.45; margin-top:8px; }}
.pseudo-positive {{ color:#b91c1c; font-weight:700; }} .unresolved {{ color:#a16207; font-weight:700; }} .unlabeled {{ color:#475569; font-weight:700; }}
</style></head><body><h1>{html.escape(title)}</h1>"""


def card(row: dict, prefix: str, source_name: str) -> str:
    rel = image_relpath(row).as_posix()
    display_status = STATUS_LABELS[row["label_type"]]
    return f"""<article class=\"card\">
<a href=\"{prefix}originals/{rel}\"><img src=\"{prefix}overlays/{rel}\" alt=\"{html.escape(row['clip_id'])} frame {row['frame_index']}\"></a>
<div class=\"meta\"><span class=\"{display_status}\">{display_status}</span><br>
clip/source/frame: {html.escape(row['clip_id'])} / {html.escape(source_name)} / {row['frame_index']}<br>
bbox source: {html.escape(source_summary(row))}<br>
confidence: {html.escape(confidence_summary(row))}<br>
<a href=\"{prefix}originals/{rel}\">open original image</a></div></article>"""


def write_page(destination: Path, title: str, rows: list[dict], prefix: str, source_names: dict[str, str], nav: str) -> None:
    cards = "\n".join(card(row, prefix, source_names.get(row["clip_id"], "unknown source")) for row in rows)
    destination.write_text(f"{page_header(title)}<nav class=\"nav\">{nav}</nav><main class=\"grid\">{cards}</main></body></html>", encoding="utf-8")


def main() -> None:
    rows = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if len(rows) != 217:
        raise ValueError(f"Expected frozen 217-frame input, found {len(rows)}")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Review output already exists: {OUTPUT_ROOT}")
    source_names = load_source_names()
    original_dir = OUTPUT_ROOT / "originals"
    overlay_dir = OUTPUT_ROOT / "overlays"
    sheet_dir = OUTPUT_ROOT / "contact_sheets"
    clip_dir = OUTPUT_ROOT / "clips"
    for directory in (original_dir, overlay_dir, sheet_dir, clip_dir):
        directory.mkdir(parents=True, exist_ok=True)

    by_clip: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        relative = image_relpath(row)
        source_path = Path(row["image_path"])
        original_path = original_dir / relative
        overlay_path = overlay_dir / relative
        original_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, original_path)
        original = cv2.imread(str(source_path))
        if original is None:
            raise ValueError(f"Could not read {source_path}")
        overlay = annotate(row, source_names.get(row["clip_id"], "unknown source"), original)
        if not cv2.imwrite(str(overlay_path), overlay):
            raise ValueError(f"Could not write {overlay_path}")
        by_clip[row["clip_id"]].append(row)

    clips = sorted(by_clip)
    for index, clip_id in enumerate(clips):
        clip_rows = by_clip[clip_id]
        overlays = [cv2.imread(str(overlay_dir / image_relpath(row))) for row in clip_rows]
        create_contact_sheet(overlays, sheet_dir / f"{clip_id}.jpg")
        previous_link = f'<a href="{clips[index - 1]}.html">previous clip</a>' if index else ""
        next_link = f'<a href="{clips[index + 1]}.html">next clip</a>' if index + 1 < len(clips) else ""
        nav = f'<a href="../index.html">index</a> | {previous_link} {next_link}'
        write_page(clip_dir / f"{clip_id}.html", f"{clip_id} review", clip_rows, "../", source_names, nav)

    positives = [row for row in rows if row["label_type"] == "pseudo_positive"]
    unresolved = [row for row in rows if row["label_type"] == "unresolved"]
    write_page(
        OUTPUT_ROOT / "pseudo_positive.html",
        "Pseudo-positive review",
        positives,
        "",
        source_names,
        '<a href="index.html">index</a> | <a href="unresolved.html">unresolved</a>',
    )
    write_page(
        OUTPUT_ROOT / "unresolved.html",
        "Unresolved review",
        unresolved,
        "",
        source_names,
        '<a href="index.html">index</a> | <a href="pseudo_positive.html">pseudo-positive</a>',
    )
    counts = Counter(STATUS_LABELS[row["label_type"]] for row in rows)
    clip_links = "".join(
        f'<li><a href="clips/{clip_id}.html">{clip_id}</a> '
        f'(<a href="contact_sheets/{clip_id}.jpg">contact sheet</a>; {len(by_clip[clip_id])} frames)</li>'
        for clip_id in clips
    )
    index = f"""{page_header("Release ball V3A review pack")}
<p>Read-only visualization of the frozen V3A pseudo-label evidence. No training label or detector threshold was changed.</p>
<p>Frames: {len(rows)} | pseudo-positive: {counts["pseudo-positive"]} | unresolved: {counts["unresolved"]} | unlabeled: {counts["unlabeled"]}</p>
<p><a href="pseudo_positive.html">pseudo-positive only</a> | <a href="unresolved.html">unresolved only</a></p>
<h2>Clips</h2><ul>{clip_links}</ul></body></html>"""
    (OUTPUT_ROOT / "index.html").write_text(index, encoding="utf-8")
    review_manifest = {
        "kind": "read_only_visual_review",
        "input": str(INPUT_PATH),
        "frame_count": len(rows),
        "display_status_counts": dict(counts),
        "clip_count": len(clips),
        "notes": [
            "Original images are copied without annotation under originals/.",
            "Overlays are derived previews only; no pseudo-label input was modified.",
            "pseudo_negative is displayed as unlabeled to avoid presenting it as training ground truth.",
        ],
    }
    (OUTPUT_ROOT / "review_manifest.json").write_text(json.dumps(review_manifest, indent=2), encoding="utf-8")
    overlay_count = len(list(overlay_dir.rglob("*.jpg")))
    original_count = len(list(original_dir.rglob("*.jpg")))
    if (overlay_count, original_count, len(list(sheet_dir.glob("*.jpg")))) != (217, 217, len(clips)):
        raise AssertionError("Review pack output counts are incomplete")
    print(f"Review pack created: {OUTPUT_ROOT}")
    print(f"frames={len(rows)} clips={len(clips)} pseudo_positive={len(positives)} unresolved={len(unresolved)}")


if __name__ == "__main__":
    main()

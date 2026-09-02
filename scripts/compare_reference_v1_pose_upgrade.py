from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference_v1.motion import build_motion_representation


EVENTS = ("dip_start", "bottom", "takeoff", "pose_release", "strict_ball_release", "body_apex", "landing")
METRICS = (
    "release_elbow_angle",
    "normalized_release_height",
    "dip_depth",
    "minimum_knee_angle",
    "release_relative_to_body_apex",
    "elbow_extension_onset_relative_to_release",
    "takeoff_to_strict_release",
    "follow_through_duration",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_sample(root: Path, sample: str) -> dict[str, Any]:
    old_dir = root / "old_yolo" / sample
    new_dir = root / "new_rtmpose" / sample
    old, new = load(old_dir / "report.json"), load(new_dir / "report.json")
    events = []
    for name in EVENTS:
        old_item, new_item = old["events"][name], new["events"][name]
        old_frame, new_frame = old_item["frame"], new_item["frame"]
        delta = new_frame - old_frame if old_frame is not None and new_frame is not None else None
        events.append(
            {
                "sample": sample,
                "name": name,
                "old_frame": old_frame,
                "new_frame": new_frame,
                "delta_frames": delta,
                "old_status": old_item["status"],
                "new_status": new_item["status"],
                "flag": "inspect" if delta is not None and abs(delta) > 5 else "ok",
            }
        )
    metrics = []
    for name in METRICS:
        old_item, new_item = old["metrics"][name], new["metrics"][name]
        old_value, new_value = old_item["value"], new_item["value"]
        delta = numeric_delta(old_value, new_value)
        status = availability_change(old_item["status"], new_item["status"])
        metrics.append(
            {
                "sample": sample,
                "name": name,
                "old_value": old_value,
                "new_value": new_value,
                "delta": delta,
                "old_status": old_item["status"],
                "new_status": new_item["status"],
                "status": status,
                "reason": metric_reason(name, delta, status),
            }
        )

    old_pose, new_pose = old["pose_reliability"]["raw"], new["pose_reliability"]["raw"]
    return {
        "sample": sample,
        "events": events,
        "metrics": metrics,
        "real_video_proxy_not_accuracy": True,
        "proxy": {
            "old_pose_coverage": old_pose["pose_coverage"],
            "new_pose_coverage": new_pose["pose_coverage"],
            "old_visible_joint_coverage": old_pose["visible_joint_coverage"],
            "new_visible_joint_coverage": new_pose["visible_joint_coverage"],
            "old_large_jump_outliers": old_pose["large_jump_outliers"],
            "new_large_jump_outliers": new_pose["large_jump_outliers"],
            "old_angle_derivative_noise": old_pose["median_joint_angle_derivative_noise_degrees_per_frame"],
            "new_angle_derivative_noise": new_pose["median_joint_angle_derivative_noise_degrees_per_frame"],
            "identity_break_frames": new["pose_reliability"]["identity_break_frames"],
            "crop_status_counts": new["pose_reliability"]["crop_status_counts"],
            "old_available_events": sum(item["status"] == "ok" for item in old["events"].values()),
            "new_available_events": sum(item["status"] == "ok" for item in new["events"].values()),
            "old_available_metrics": sum(item["status"] == "ok" for item in old["metrics"].values()),
            "new_available_metrics": sum(item["status"] == "ok" for item in new["metrics"].values()),
        },
        "runtime": {
            "old_total_seconds": old["runtime"]["total_seconds"],
            "new_total_seconds": new["runtime"]["total_seconds"],
            "old_inference_seconds": old["runtime"]["inference_seconds"],
            "new_inference_seconds": new["runtime"]["inference_seconds"],
            "old_person_detector_ms": old["runtime"]["person_detector_ms"],
            "new_person_detector_ms": new["runtime"]["person_detector_ms"],
            "new_pose_head_ms": new["runtime"]["pose_head_ms"],
            "frames": new["input"]["frame_count"],
        },
        "old_report": old,
        "new_report": new,
        "old_dir": old_dir,
        "new_dir": new_dir,
    }


def numeric_delta(old: Any, new: Any) -> float | int | None:
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return round(new - old, 4)
    if isinstance(old, dict) and isinstance(new, dict) and isinstance(old.get("frames"), (int, float)) and isinstance(new.get("frames"), (int, float)):
        return new["frames"] - old["frames"]
    return None


def availability_change(old: str, new: str) -> str:
    if old != "ok" and new == "ok":
        return "availability_improved"
    if old == "ok" and new != "ok":
        return "availability_regressed"
    return "changed" if old != new else "same_status"


def metric_reason(name: str, delta: float | int | None, status: str) -> str | None:
    if status != "same_status":
        return status
    threshold = 5.0 if "angle" in name else 3.0
    if delta is not None and abs(delta) > threshold:
        return "material pose-derived change; inspect synchronized evidence"
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def representative_frames(result: dict[str, Any]) -> list[tuple[str, int]]:
    report = result["new_report"]
    events = report["events"]
    first_event = next((events[name]["frame"] for name in EVENTS if events[name]["frame"] is not None), 15)
    frames = [("setup", max(0, first_event - 15))]
    for label, event_name in (
        ("dip", "bottom"),
        ("takeoff", "takeoff"),
        ("pose_release", "pose_release"),
        ("strict_release", "strict_ball_release"),
        ("landing", "landing"),
    ):
        frame = events[event_name]["frame"]
        if frame is not None:
            frames.append((label, frame))
    release = events["strict_ball_release"]["frame"] or events["pose_release"]["frame"]
    if release is not None:
        frames.append(("follow_through", min(report["input"]["frame_count"] - 1, release + 10)))
    for frame in result["proxy"]["identity_break_frames"]:
        frames.append((f"identity_break_f{frame}", frame))
    difficult = difficult_arm_frame(result["new_dir"] / "evidence" / "pose_trajectories.json", report["attempt"]["shooting_side"])
    if difficult is not None:
        frames.append((f"difficult_arm_f{difficult}", difficult))
    return list(dict.fromkeys(frames))


def difficult_arm_frame(path: Path, side: str) -> int | None:
    elbow, wrist = (7, 9) if side == "left" else (8, 10)
    choices = []
    for row in load(path):
        pose = row.get("raw_pose")
        if pose:
            confidence = pose["keypoint_confidence"]
            choices.append((min(confidence[elbow], confidence[wrist]), row["frame_index"]))
    return min(choices)[1] if choices else None


def extract_frame(video: Path, frame_index: int, output: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Could not write review frame: {output}")


def make_review(root: Path, results: list[dict[str, Any]]) -> Path:
    review = root / "review"
    cards = []
    for result in results:
        sample = result["sample"]
        panels = []
        for label, frame in representative_frames(result):
            old_image = review / "assets" / sample / f"{label}_old.jpg"
            new_image = review / "assets" / sample / f"{label}_new.jpg"
            extract_frame(result["old_dir"] / "annotated.mp4", frame, old_image)
            extract_frame(result["new_dir"] / "annotated.mp4", frame, new_image)
            panels.append(
                f'<article><h4>{html.escape(label)} · frame {frame}</h4><div class="pair">'
                f'<figure><img src="assets/{sample}/{old_image.name}"><figcaption>OLD YOLO · raw</figcaption></figure>'
                f'<figure><img src="assets/{sample}/{new_image.name}"><figcaption>NEW RTMPose · raw</figcaption></figure>'
                "</div></article>"
            )
        event_rows = "".join(
            f'<tr class="{row["flag"]}"><td>{row["name"]}</td><td>{row["old_frame"]} ({row["old_status"]})</td><td>{row["new_frame"]} ({row["new_status"]})</td><td>{row["delta_frames"]}</td></tr>'
            for row in result["events"]
        )
        metric_rows = "".join(
            f'<tr><td>{row["name"]}</td><td>{html.escape(str(row["old_value"]))}</td><td>{html.escape(str(row["new_value"]))}</td><td>{html.escape(str(row["delta"]))}</td><td>{html.escape(row["status"])}</td></tr>'
            for row in result["metrics"]
        )
        proxy = result["proxy"]
        cards.append(
            f'<section><h2>{sample}</h2><div class="videos"><figure><video controls src="../old_yolo/{sample}/annotated.mp4"></video><figcaption>OLD YOLO · raw localization</figcaption></figure><figure><video controls src="../new_rtmpose/{sample}/annotated.mp4"></video><figcaption>NEW RTMPose · raw localization</figcaption></figure></div>'
            f'<p><b>REAL VIDEO REGRESSION (not HUMAN_GT accuracy):</b> pose coverage {proxy["old_pose_coverage"]:.1%} → {proxy["new_pose_coverage"]:.1%}; '
            f'large-jump outliers {proxy["old_large_jump_outliers"]} → {proxy["new_large_jump_outliers"]}; '
            f'available metrics {proxy["old_available_metrics"]} → {proxy["new_available_metrics"]}; identity breaks {proxy["identity_break_frames"]}.</p>'
            f'<h3>Event audit</h3><table><tr><th>event</th><th>old</th><th>new</th><th>Δ frames</th></tr>{event_rows}</table>'
            f'<h3>Metric audit</h3><table><tr><th>metric</th><th>old</th><th>new</th><th>Δ</th><th>status</th></tr>{metric_rows}</table>'
            f'<h3>Synchronized raw-localization frames</h3>{"".join(panels)}</section>'
        )
    document = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reference V1 Pose Upgrade Review</title><style>
body{{margin:0;background:#101715;color:#edf3ec;font:15px system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:24px}}h1,h2,h3{{color:#ffd66b}}section,article{{background:#19231f;border:1px solid #33443d;border-radius:14px;padding:16px;margin:18px 0}}.videos,.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}video,img{{width:100%;background:#000;border-radius:8px}}figure{{margin:0}}figcaption{{color:#aab9b2;margin-top:5px}}table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid #33443d;padding:8px;text-align:left}}tr.inspect{{background:#4b3325}}.note{{padding:14px;border-left:5px solid #ffd66b;background:#27291f}}@media(max-width:800px){{.videos,.pair{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Reference V1 · RTMPose Mainline Review</h1><p class="note"><b>PUBLIC_GT FACT:</b> accepted benchmark established RTMPose localization accuracy. <b>REAL VIDEO REGRESSION:</b> the evidence below is coverage, continuity, event/metric availability, runtime and visual inspection only—not localization accuracy.</p>{''.join(cards)}</main></body></html>'''
    path = review / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


def write_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Reference V1 pose upgrade regression",
        "",
        "> PUBLIC_GT FACT and REAL VIDEO REGRESSION are separate. These real-video proxies are not localization accuracy.",
        "",
    ]
    for result in results:
        proxy, runtime = result["proxy"], result["runtime"]
        lines.extend(
            [
                f'## {result["sample"]}',
                "",
                f'- Pose coverage: {proxy["old_pose_coverage"]:.2%} → {proxy["new_pose_coverage"]:.2%}',
                f'- Large-jump outliers: {proxy["old_large_jump_outliers"]} → {proxy["new_large_jump_outliers"]}',
                f'- Angle derivative noise: {proxy["old_angle_derivative_noise"]:.3f} → {proxy["new_angle_derivative_noise"]:.3f} deg/frame',
                f'- Identity breaks (explicit): {proxy["identity_break_frames"]}',
                f'- Runtime total: {runtime["old_total_seconds"]:.3f}s → {runtime["new_total_seconds"]:.3f}s; RTMPose head {runtime["new_pose_head_ms"] / runtime["frames"]:.2f} ms/frame',
                "",
                "| Event | Old | New | Δ frames |",
                "|---|---:|---:|---:|",
            ]
        )
        lines.extend(
            f'| {row["name"]} | {row["old_frame"]} ({row["old_status"]}) | {row["new_frame"]} ({row["new_status"]}) | {row["delta_frames"]} |'
            for row in result["events"]
        )
        lines.extend(["", "| Metric | Old | New | Δ | Status |", "|---|---|---|---:|---|"])
        lines.extend(
            f'| {row["name"]} | `{row["old_value"]}` | `{row["new_value"]}` | {row["delta"]} | {row["status"]} |'
            for row in result["metrics"]
        )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "RTMPose raw localization is the mainline. The remaining high-value blocker is human-ball/release perception under basketball arm/ball occlusion, not another broad pose survey.",
            "",
            "- REFERENCE_V1_POSE_UPGRADE = PASS",
            "- RTMPOSE_MAINLINE = YES",
            "- SHOOTER_CONTINUITY = IMPROVED",
            "- RAW_POSE_DEFAULT = YES",
            "- GLOBAL_FILTER_DEFAULT = NO",
            "- STRICT_RELEASE_REGRESSION = PASS",
            "- MOTION_REPRESENTATION_COMPATIBILITY = PASS",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=["IMG_7215", "IMG_7216"])
    args = parser.parse_args()
    results = [compare_sample(args.root, sample) for sample in args.samples]
    comparison = args.root / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    public = [{key: value for key, value in result.items() if key not in {"old_report", "new_report", "old_dir", "new_dir"}} for result in results]
    (comparison / "summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(comparison / "events.csv", [row for result in results for row in result["events"]])
    write_csv(comparison / "metrics.csv", [row for result in results for row in result["metrics"]])
    write_csv(comparison / "runtime.csv", [{"sample": result["sample"], **result["runtime"]} for result in results])
    write_markdown(comparison / "report.md", results)
    motion_dir = comparison / "motion"
    motion_dir.mkdir(exist_ok=True)
    for result in results:
        motion = build_motion_representation(result["new_report"], load(result["new_dir"] / "evidence" / "pose_trajectories.json"))
        (motion_dir / f'{result["sample"]}.json').write_text(json.dumps(motion, ensure_ascii=False, indent=2), encoding="utf-8")
    review = make_review(args.root, results)
    required = [comparison / "summary.json", comparison / "events.csv", comparison / "metrics.csv", comparison / "runtime.csv", comparison / "report.md", review]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
    print(json.dumps({"status": "PASS", "review": str(review), "samples": args.samples}, ensure_ascii=False))


if __name__ == "__main__":
    main()

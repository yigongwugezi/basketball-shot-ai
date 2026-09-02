from __future__ import annotations

import html
import json
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

PHASE_COLORS = {
    "preparation": (168, 126, 60),
    "dip": (55, 133, 205),
    "upward_drive": (37, 176, 118),
    "follow_through": (32, 179, 229),
    "landing_recovery": (128, 114, 196),
}


def render_annotated_video(
    input_path: Path,
    output_path: Path,
    evidence_dir: Path,
    rows: list[dict[str, Any]],
    phases: dict[str, dict[str, Any]],
    events: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    *,
    pose_key: str = "analysis_pose",
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not reopen video for rendering: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not create annotated.mp4")

    rows_by_frame = {int(row["frame_index"]): row for row in rows}
    event_frames: dict[int, list[str]] = {}
    for name, item in events.items():
        if item.get("frame") is not None:
            event_frames.setdefault(int(item["frame"]), []).append(name)

    trajectory: deque[tuple[int, int]] = deque(maxlen=14)
    frame_index = 0
    written = 0
    evidence_written: list[str] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        row = rows_by_frame.get(frame_index)
        if row and row.get("ball"):
            center = row["ball"]["center"]
            trajectory.append((round(center[0]), round(center[1])))
        annotated = draw_overlay(
            frame,
            row,
            frame_index,
            _phase_at_frame(phases, frame_index),
            event_frames.get(frame_index, []),
            metrics,
            list(trajectory),
            pose_key,
        )
        writer.write(annotated)
        written += 1
        for event_name in event_frames.get(frame_index, []):
            path = evidence_dir / f"{event_name}.jpg"
            cv2.imwrite(str(path), annotated)
            evidence_written.append(path.name)
        frame_index += 1

    capture.release()
    writer.release()
    return {
        "frames_written": written,
        "fps": fps,
        "evidence_images": sorted(set(evidence_written)),
        "codec": "mp4v",
    }


def draw_overlay(
    frame: np.ndarray,
    row: dict[str, Any] | None,
    frame_index: int,
    phase_name: str | None,
    event_names: list[str],
    metrics: dict[str, dict[str, Any]],
    trajectory: list[tuple[int, int]],
    pose_key: str = "analysis_pose",
) -> np.ndarray:
    canvas = frame.copy()
    scale = max(0.55, min(canvas.shape[0], canvas.shape[1]) / 900)
    line = max(2, round(scale * 3))
    phase_color = PHASE_COLORS.get(phase_name, (180, 180, 180))
    pose = row.get(pose_key) if row else None
    if pose:
        points = np.asarray(pose["keypoints"], dtype=float)
        confidence = np.asarray(pose["confidence"], dtype=float)
        for first, second in SKELETON:
            if confidence[first] >= 0.25 and confidence[second] >= 0.25:
                cv2.line(canvas, tuple(points[first].astype(int)), tuple(points[second].astype(int)), phase_color, line)
        for index, point in enumerate(points):
            if confidence[index] >= 0.25:
                cv2.circle(canvas, tuple(point.astype(int)), line + 2, (246, 231, 105), -1)

    for first, second in zip(trajectory, trajectory[1:]):
        cv2.line(canvas, first, second, (43, 211, 255), line)
    if row and row.get("ball"):
        ball = row["ball"]
        x1, y1, x2, y2 = [round(value) for value in ball["bbox"]]
        center = tuple(round(value) for value in ball["center"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (31, 145, 255), line)
        cv2.circle(canvas, center, max(4, line + 2), (22, 235, 255), -1)
        cv2.putText(
            canvas,
            f"ball {ball['confidence']:.2f}",
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.65,
            (22, 235, 255),
            line,
            cv2.LINE_AA,
        )

    pose_release = metrics["release_elbow_angle"]
    if pose_release.get("frame") == frame_index and pose_release["status"] == "ok" and pose:
        side = "left" if pose.get("shooting_side") == "left" else "right"
        indices = (5, 7, 9) if side == "left" else (6, 8, 10)
        points = np.asarray(pose["keypoints"], dtype=float)
        for first, second in zip(indices, indices[1:]):
            cv2.line(canvas, tuple(points[first].astype(int)), tuple(points[second].astype(int)), (72, 255, 151), line + 2)
        elbow = tuple(points[indices[1]].astype(int))
        cv2.putText(canvas, f"{pose_release['value']:.1f} deg 2D", (elbow[0] + 12, elbow[1]), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, (72, 255, 151), line, cv2.LINE_AA)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], max(58, round(82 * scale))), (18, 28, 25), -1)
    canvas = cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0)
    title = phase_name.replace("_", " ").upper() if phase_name else "PHASE UNAVAILABLE"
    cv2.putText(canvas, f"{title}  |  frame {frame_index}", (18, round(34 * scale) + 12), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.78, (245, 243, 228), line, cv2.LINE_AA)
    if event_names:
        event_text = "  +  ".join(name.replace("_", " ").upper() for name in event_names)
        cv2.putText(canvas, event_text, (18, round(65 * scale) + 12), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.62, (43, 211, 255), line, cv2.LINE_AA)
    return canvas


def write_report_html(report: dict[str, Any], output_path: Path) -> None:
    input_name = html.escape(report["input"]["name"])
    status = html.escape(report["attempt"]["analysis_status"])
    quality_warnings = [item for item in report["quality"]["checks"] if item["status"] != "ok"]
    quality_html = "".join(
        f'<li><b>{html.escape(item["name"])}</b>: {html.escape(item["detail"])}</li>'
        for item in quality_warnings
    ) or "<li>No blocking input warning was detected.</li>"

    phase_html = "".join(_timeline_phase(item, report["input"]["frame_count"]) for item in report["phases"].values())
    event_html = "".join(_timeline_event(item) for item in report["events"].values())
    metric_html = "".join(_metric_card(item) for item in report["metrics"].values())
    observation_html = "".join(f'<li>{html.escape(item["text"])}</li>' for item in report["observations"]) or "<li>No supported factual observation.</li>"
    suggestion_html = "".join(f'<li>{html.escape(item["text"])}</li>' for item in report["suggestions"]) or "<li>No suggestion was added without sufficient evidence.</li>"
    risk_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in report["risks"]) or "<li>No additional pipeline risk flag.</li>"
    evidence_html = "".join(
        f'<figure><img src="evidence/{html.escape(name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name.removesuffix(".jpg").replace("_", " "))}</figcaption></figure>'
        for name in report["artifacts"].get("evidence_images", [])
    ) or "<p>No event image was available.</p>"
    runtime = report["runtime"]

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reference V1 · {input_name}</title>
<style>
:root{{--ink:#15231d;--paper:#f3eddf;--card:#fffdf7;--green:#1d6b4d;--gold:#d69b2d;--red:#b94a38;--line:#d8cfbc;--muted:#6c6a60}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 12% 0,#fff5c9 0,transparent 28%),linear-gradient(135deg,#ebe3d4,#f7f3e8 58%,#e8efe7);color:var(--ink);font-family:"Microsoft YaHei UI","Noto Sans SC",sans-serif}}
header{{padding:24px clamp(18px,4vw,54px);background:#15231df2;color:#fff;display:flex;justify-content:space-between;gap:20px;align-items:end}}header h1{{margin:0;font-family:Georgia,"Microsoft YaHei UI",serif;font-size:clamp(25px,4vw,48px)}}header p{{margin:6px 0 0;color:#d8e2dc}}.status{{border:1px solid #ffffff55;border-radius:999px;padding:8px 14px;text-transform:uppercase;letter-spacing:.08em}}
main{{padding:24px clamp(14px,3vw,44px) 50px}}.hero{{display:grid;grid-template-columns:minmax(0,7fr) minmax(280px,3fr);gap:18px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 12px 30px #3a33251c}}video{{width:100%;max-height:72vh;background:#101713;border-radius:11px}}h2{{font-family:Georgia,"Microsoft YaHei UI",serif;margin:0 0 12px}}h3{{margin:5px 0 9px}}ul{{padding-left:20px;line-height:1.65}}.timeline{{margin-top:18px}}.phase-track{{display:flex;gap:4px;min-height:72px}}.phase{{flex:1;padding:10px;border-radius:8px;background:#dce8df;border-left:6px solid var(--green);font-size:12px}}.phase.bad{{background:#eee5dc;border-color:var(--red)}}.events{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}}button.event{{border:1px solid var(--line);background:#fff9e8;border-radius:999px;padding:8px 11px;cursor:pointer}}button.event.bad{{opacity:.62;cursor:default}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:18px}}.metric{{min-height:155px}}.metric .value{{font-size:25px;font-weight:800;color:var(--green);word-break:break-word}}.metric.bad .value{{font-size:15px;color:var(--red)}}.tag{{display:inline-block;border-radius:999px;padding:4px 8px;background:#e4ece6;font-size:11px;margin-bottom:9px}}.tag.bad{{background:#f4ddd7;color:#7b271d}}.fine{{font-size:12px;color:var(--muted);line-height:1.5}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}}.evidence{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}}figure{{margin:0}}figure img{{width:100%;border-radius:9px;border:1px solid var(--line)}}figcaption{{font-size:12px;color:var(--muted);margin-top:4px}}details{{margin-top:18px}}code{{font-family:Consolas,monospace}}@media(max-width:900px){{.hero,.two{{grid-template-columns:1fr}}header{{align-items:start;flex-direction:column}}}}
</style></head><body>
<header><div><h1>Reference V1 · 投篮动作拆解</h1><p>{input_name} · Evidence-first experimental report</p></div><div class="status">{status}</div></header>
<main><section class="hero"><div class="card"><video id="video" controls preload="metadata" src="annotated.mp4"></video></div><aside class="card"><h2>Attempt</h2><p><b>ID</b><br><code>{html.escape(report["attempt"]["attempt_id"])}</code></p><p><b>Shooting side</b><br>{html.escape(str(report["attempt"]["shooting_side"]))}</p><p><b>Outcome</b><br>unknown</p><h3>Quality warnings</h3><ul>{quality_html}</ul><p class="fine">总运行时间 {runtime["total_seconds"]:.2f}s；推理 {runtime["inference_seconds"]:.2f}s；渲染 {runtime["render_seconds"]:.2f}s。</p></aside></section>
<section class="card timeline"><h2>Phase / Event Timeline</h2><div class="phase-track">{phase_html}</div><div class="events">{event_html}</div></section>
<section class="metrics">{metric_html}</section>
<section class="two"><div class="card"><h2>事实描述</h2><ul>{observation_html}</ul></div><div class="card"><h2>保守建议</h2><ul>{suggestion_html}</ul></div></section>
<section class="card" style="margin-top:18px"><h2>关键证据帧</h2><div class="evidence">{evidence_html}</div><details><summary>风险与缺失原因</summary><ul>{risk_html}</ul></details><details><summary>Ball evidence</summary><p class="fine">{len(report["ball_evidence"]["center_observations"])} 个 release-window detector observations；tracker_used={str(report["ball_evidence"]["tracker_used"]).lower()}；详情见 <code>evidence/ball_motion.json</code>。</p></details></section>
</main><script>function seekVideo(seconds){{if(seconds===null)return;const video=document.getElementById('video');video.currentTime=Number(seconds);video.play();}}</script></body></html>"""
    output_path.write_text(document, encoding="utf-8")


def _phase_at_frame(phases: dict[str, dict[str, Any]], frame: int) -> str | None:
    for name, item in phases.items():
        if item.get("start_frame") is not None and item.get("end_frame") is not None:
            if int(item["start_frame"]) <= frame <= int(item["end_frame"]):
                return name
    return None


def _timeline_phase(item: dict[str, Any], frame_count: int) -> str:
    bad = " bad" if item["status"] != "ok" else ""
    frames = (
        f'{item["start_frame"]}–{item["end_frame"]}'
        if item.get("start_frame") is not None and item.get("end_frame") is not None
        else "unavailable"
    )
    return f'<div class="phase{bad}"><b>{html.escape(item["label_zh"])}</b><br>{html.escape(frames)}<br>{html.escape(item["status"])}</div>'


def _timeline_event(item: dict[str, Any]) -> str:
    bad = " bad" if item["status"] != "ok" else ""
    time_value = item.get("timestamp_seconds")
    label = f'{item["label_zh"]} · f{item["frame"]}' if item.get("frame") is not None else f'{item["label_zh"]} · {item["status"]}'
    action = f"seekVideo({time_value})" if time_value is not None else ""
    return f'<button class="event{bad}" onclick="{action}">{html.escape(label)}</button>'


def _metric_card(item: dict[str, Any]) -> str:
    bad = " bad" if item["status"] != "ok" else ""
    value = item["value"]
    if isinstance(value, dict):
        value_text = " · ".join(f"{key}: {value[key]}" for key in value)
    elif value is None:
        value_text = item.get("reason") or "unavailable"
    else:
        value_text = f"{value} {item.get('unit') or ''}".strip()
    provenance = ", ".join(item.get("provenance", [])) or "none"
    confidence = item.get("confidence")
    confidence_text = f"{confidence:.2f}" if confidence is not None else "n/a"
    return f'<article class="card metric{bad}"><span class="tag{bad}">{html.escape(item["status"])}</span><h3>{html.escape(item["label_zh"])}</h3><div class="value">{html.escape(str(value_text))}</div><p class="fine">confidence {confidence_text}<br>provenance: {html.escape(provenance)}</p></article>'

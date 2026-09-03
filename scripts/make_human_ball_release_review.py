from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import cv2


SAMPLES = ("IMG_7215", "IMG_7216", "BILI_010_A", "BILI_002_A", "BILI_010_B")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_frame(video: Path, frame_index: int, output: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not extract frame {frame_index} from {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise RuntimeError(f"Could not write {output}")


def event_frame(report: dict[str, Any], name: str) -> int | None:
    value = report["events"][name].get("frame")
    return int(value) if value is not None else None


def comparison_row(name: str, old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    human_ball = new["human_ball_release"]
    old_pose = event_frame(old, "pose_release")
    old_strict = event_frame(old, "strict_ball_release")
    new_pose = event_frame(new, "pose_release")
    new_strict = event_frame(new, "strict_ball_release")
    return {
        "sample": name,
        "old_pose_release": old_pose,
        "new_pose_release": new_pose,
        "pose_frame_delta": new_pose - old_pose if None not in (new_pose, old_pose) else None,
        "old_strict_release": old_strict,
        "new_strict_release": new_strict,
        "strict_frame_delta": new_strict - old_strict if None not in (new_strict, old_strict) else None,
        "old_ball_status": old["ball_evidence"]["status"],
        "new_ball_status": human_ball["ball_track_status"],
        "ball_track_coverage": human_ball["ball_track_quality"]["coverage"],
        "contact_state_coverage": human_ball["contact_state_coverage"],
        "old_strict_abstained": old_strict is None,
        "new_strict_abstained": new_strict is None,
        "event_ordering": "pose<=strict" if new_strict is not None and new_pose <= new_strict else "strict_abstained" if new_strict is None else "needs_review",
    }


def make_review(root: Path) -> None:
    baseline_root = root / "baseline_eb87cc0"
    new_root = root / "new_human_ball"
    review_root = root / "review"
    assets = review_root / "assets"
    comparison_root = root / "comparison"
    assets.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)
    rows = []
    sections = []
    for name in SAMPLES:
        old = load_json(baseline_root / name / "report.json")
        new = load_json(new_root / name / "report.json")
        row = comparison_row(name, old, new)
        rows.append(row)
        human_ball = new["human_ball_release"]
        state_by_frame = {item["frame"]: item for item in human_ball["contact_state_sequence"]}
        pose = event_frame(new, "pose_release")
        strict = event_frame(new, "strict_ball_release")
        center = pose if pose is not None else strict
        if center is None:
            window = human_ball["release_window"]
            center = round(sum(window) / 2) if window else 0
        start = max(0, min(pose if pose is not None else center, strict if strict is not None else center) - 6)
        end = max(pose if pose is not None else center, strict if strict is not None else center) + 6
        original_video = Path(new["input"]["path"])
        annotated_video = new_root / name / "annotated.mp4"
        cards = []
        timeline = []
        for frame in range(start, end + 1):
            state = state_by_frame.get(frame)
            original_rel = Path("assets") / f"{name}_f{frame:04d}_rgb.jpg"
            overlay_rel = Path("assets") / f"{name}_f{frame:04d}_overlay.jpg"
            extract_frame(original_video, frame, review_root / original_rel)
            extract_frame(annotated_video, frame, review_root / overlay_rel)
            markers = []
            if frame == pose:
                markers.append("POSE RELEASE")
            if frame == strict:
                markers.append("STRICT RELEASE")
            status = state["ball_status"] if state else "OUTSIDE_WINDOW"
            contact = state["contact_state"] if state else "UNKNOWN"
            distance = state.get("wrist_ball_distance_diameters") if state else None
            reliability = state.get("evidence_reliability") if state else None
            marker_html = " ".join(f"<b>{html.escape(value)}</b>" for value in markers)
            cards.append(
                f'<article class="frame {contact.lower()}"><h4>f{frame} {marker_html}</h4>'
                f'<div class="pair"><figure><img src="{original_rel.as_posix()}"><figcaption>Original RGB</figcaption></figure>'
                f'<figure><img src="{overlay_rel.as_posix()}"><figcaption>RTMPose arm + tracked ball</figcaption></figure></div>'
                f'<p><span>{html.escape(status)}</span> · {html.escape(contact)} · d={html.escape(str(distance))} ball diameters · reliability={html.escape(str(reliability))}</p></article>'
            )
            timeline.append(f'<div class="tick {contact.lower()}"><b>f{frame}</b><br>{html.escape(contact)}<br><small>{html.escape(status)}</small></div>')
        support = human_ball["supporting_evidence"]
        diagnosis = (
            f'pose-only candidate f{human_ball["release_pose"]["pose_only_frame"]} '
            f'(score {support["pose_only_score"]}); selected pose f{pose} '
            f'(score {support["selected_pose_score"]}); strict {strict if strict is not None else "abstained"}; '
            f'last likely contact {support["last_likely_contact_frame"]}.'
        )
        sections.append(
            f'<section><h2>{html.escape(name)}</h2><p>{html.escape(diagnosis)}</p>'
            f'<div class="timeline">{"".join(timeline)}</div><div class="frames">{"".join(cards)}</div></section>'
        )

    comparison_json = comparison_root / "old_vs_new.json"
    comparison_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (comparison_root / "old_vs_new.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table_head = "".join(f"<th>{html.escape(key)}</th>" for key in rows[0])
    table_rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in row) + "</tr>" for row in rows)
    document = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reference V1 Human-Ball Release Review</title><style>
:root{{--ink:#eaf3ed;--bg:#0d1713;--card:#17251f;--line:#385145;--gold:#f2c14e;--green:#58d68d;--blue:#66b3ff;--red:#ff7b72}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,"Microsoft YaHei UI",sans-serif}}header{{position:sticky;top:0;z-index:2;background:#0d1713ee;border-bottom:1px solid var(--line);padding:18px 3vw}}main{{padding:18px 3vw 50px}}h1,h2{{margin:0 0 9px}}section{{margin-top:30px;border-top:1px solid var(--line);padding-top:22px}}.summary{{overflow:auto}}table{{border-collapse:collapse;min-width:1100px}}th,td{{padding:7px;border:1px solid var(--line);font-size:12px;text-align:left}}.timeline{{display:flex;overflow:auto;gap:4px;margin:12px 0}}.tick{{min-width:100px;padding:8px;border-radius:8px;background:#26352e;text-align:center;font-size:11px}}.likely_contact{{border:2px solid var(--gold)}}.separating{{border:2px solid var(--blue)}}.no_contact{{border:2px solid var(--green)}}.unknown{{border:2px solid var(--red)}}.frames{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px}}.frame{{background:var(--card);border-radius:12px;padding:10px}}.frame h4,.frame p{{margin:4px 0 8px}}.frame h4 b{{color:var(--gold)}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}figure{{margin:0}}img{{display:block;width:100%;border-radius:7px}}figcaption,small{{color:#aebdb5;font-size:11px}}@media(max-width:620px){{.frames{{grid-template-columns:1fr}}.pair{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Human-Ball / Release Perception V1</h1><p>Original RGB beside RTMPose shooting arm, tracked ball, wrist relation and explicit contact state. Missing and ambiguous evidence remains visible.</p></header><main>
<section class="summary"><h2>eb87cc0 → Human-Ball V1</h2><table><thead><tr>{table_head}</tr></thead><tbody>{table_rows}</tbody></table></section>{''.join(sections)}</main></body></html>'''
    (review_root / "index.html").write_text(document, encoding="utf-8")
    assert (review_root / "index.html").is_file()
    assert all((new_root / name / "report.json").is_file() for name in SAMPLES)
    print(f"review: {review_root / 'index.html'}")
    print(f"comparison: {comparison_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    make_review(args.root.resolve())


if __name__ == "__main__":
    main()

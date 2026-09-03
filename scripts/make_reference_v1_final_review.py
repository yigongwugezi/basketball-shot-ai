from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import cv2


REQUIRED = ("annotated.mp4", "report.html", "report.json", "timeline.csv", "evidence")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract(video: Path, frame_index: int, output: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 86]):
        raise RuntimeError(f"Could not extract frame {frame_index} from {video}")


def make_review(root: Path, manifest_path: Path) -> None:
    manifest = load(manifest_path)
    review = root / "review"
    assets = review / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    sections, results = [], []
    for sample in manifest["samples"]:
        name = sample["id"]
        run = root / name
        missing = [item for item in REQUIRED if not (run / item).exists()]
        if missing:
            raise RuntimeError(f"{name} missing artifacts: {missing}")
        report = load(run / "report.json")
        motion = report["motion_representation"]
        event_cards = []
        for event_name in ("dip_bottom", "release_pose", "strict_ball_release", "landing"):
            event = motion["events"][event_name]
            if event["frame"] is None:
                event_cards.append(f'<article class="missing"><h4>{html.escape(event_name)}</h4><strong>INSUFFICIENT EVIDENCE</strong><p>{html.escape(str(event.get("reason") or event["status"]))}</p></article>')
                continue
            relative = Path("assets") / f'{name}_{event_name}_f{event["frame"]}.jpg'
            extract(run / "annotated.mp4", event["frame"], review / relative)
            event_cards.append(f'<article><h4>{html.escape(event_name)} · f{event["frame"]}</h4><img src="{relative.as_posix()}"><p>{event["reliability"]} · t={event["normalized_shot_time"]}</p></article>')
        metrics = motion["kinematics"]["metrics"]
        metric_text = "; ".join(f'{name}: {item["value"]} {item.get("unit") or ""} [{item["reliability"]}]' for name, item in metrics.items() if item["status"] in {"ok", "low_confidence"})
        unavailable = [f'{name}: {item.get("reason") or item["status"]}' for name, item in metrics.items() if item["status"] not in {"ok", "low_confidence"}]
        relations = [f'{item["from_event"]} → {item["to_event"]}: {item["delta_frames"]}f' for item in motion["temporal_relations"].values() if item["status"] in {"ok", "low_confidence"}]
        strict = motion["events"]["strict_ball_release"]
        result = {"sample": name, "classification": sample["classification"],
                  "analysis_status": report["attempt"]["analysis_status"],
                  "strict_release_status": strict["status"], "strict_release_frame": strict["frame"],
                  "ball_track_status": report["human_ball_release"]["ball_track_status"],
                  "artifacts_complete": not missing, "runtime_seconds": report["runtime"]["total_seconds"]}
        results.append(result)
        insufficient = strict["frame"] is None or unavailable
        sections.append(f'''<section><div class="heading"><h2>{html.escape(name)}</h2><span>{html.escape(sample["classification"])}</span></div>
<p><b>Result:</b> {html.escape(report["attempt"]["analysis_status"])} · ball {html.escape(str(result["ball_track_status"]))} · strict release {html.escape(str(strict["frame"]) if strict["frame"] is not None else "ABSTAINED")}</p>
<div class="frames">{"".join(event_cards)}</div><h3>Factual metrics</h3><p>{html.escape(metric_text or "No supported metric")}</p>
<h3>Temporal relations</h3><p>{html.escape("; ".join(relations) or "No supported relation")}</p>
<div class="{'warning' if insufficient else 'ok'}"><b>{'INSUFFICIENT EVIDENCE IS EXPLICIT' if insufficient else 'SUPPORTED EVIDENCE'}</b><br>{html.escape('; '.join(unavailable) or 'No metric abstention')}</div>
<p class="links"><a href="../{name}/report.html">report.html</a> · <a href="../{name}/report.json">report.json</a> · <a href="../{name}/timeline.csv">timeline.csv</a></p></section>''')
    (root / "acceptance_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    total_runtime = sum(item["runtime_seconds"] for item in results)
    document = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reference V1 Final Acceptance</title><style>
:root{{--bg:#0c1512;--card:#16231e;--line:#365044;--ink:#edf5f0;--muted:#a8bbb1;--gold:#f0bd55;--red:#ff8274;--green:#62d99b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,"Microsoft YaHei UI"}}header,main{{max-width:1500px;margin:auto;padding:24px}}header{{border-bottom:1px solid var(--line)}}h1,h2,h3{{margin:0 0 10px}}section{{margin:28px 0;padding:20px;background:var(--card);border:1px solid var(--line);border-radius:14px}}.heading{{display:flex;justify-content:space-between;gap:12px}}.heading span{{color:var(--gold)}}.frames{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}}article{{padding:10px;background:#0e1915;border-radius:9px}}img{{width:100%;display:block;border-radius:6px}}.missing,.warning{{border:2px solid var(--red);color:#ffd4cf;padding:14px}}.ok{{border:2px solid var(--green);color:#c7ffe2;padding:14px}}p{{line-height:1.55;color:var(--muted)}}a{{color:#7dc7ff}}@media(max-width:650px){{.heading{{display:block}}}}
</style></head><body><header><h1>Reference V1 · Final Acceptance</h1><p>Default CLI · {len(results)} representative real videos · total runtime {total_runtime:.2f}s. Abstention and missing evidence are first-class results.</p></header><main>{''.join(sections)}</main></body></html>'''
    (review / "index.html").write_text(document, encoding="utf-8")
    print(review / "index.html")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    make_review(args.root.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main()

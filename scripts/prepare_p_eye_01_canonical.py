#!/usr/bin/env python3
"""Freeze CANONICAL_EYE_TEST_01 and create its human GT review page."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2


DEFAULT_VIDEO = Path(
    r"E:\BasketballShotAI\raw\selected_shot_material\clips\part01_0000_0005_first_5s.webm"
)
DEFAULT_OBSERVATIONS = Path(
    r"C:\Users\20825\.codex\visualizations\2026\09\05\01a06f5f-f474-7a20-9e9f-029373fa5b0a"
    r"\p-ball-obs1-canonical.json"
)
DEFAULT_OUTPUT = Path(
    r"E:\BasketballShotAI\benchmarks\P-EYE-01\CANONICAL_EYE_TEST_01"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_source_frames(video: Path, frames_dir: Path, expected_count: int) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {video}")
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        destination = frames_dir / f"frame_{count:06d}.jpg"
        if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Cannot write {destination}")
        count += 1
    capture.release()
    if count != expected_count:
        raise ValueError(f"Expected {expected_count} source frames, extracted {count}")


def build_rows(observations: dict, frames_dir: Path) -> list[dict]:
    dense = observations["dense_ball_observations"]
    rows = []
    for item in dense["frames"]:
        frame_index = int(item["frame_index"])
        filename = f"frame_{frame_index:06d}.jpg"
        frame_path = frames_dir / filename
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_s": item["timestamp"],
                "image": f"frames/{filename}",
                "review_status": "TODO",
                "ball_visible": None,
                "gt_bbox_xyxy": None,
                "gt_center_xy": None,
                "gt_apparent_diameter_px": None,
                "occluded": None,
                "ambiguous": None,
                "near_hand": None,
                "free_flight": None,
                "near_rim": None,
                "notes": "",
                "preannotation_non_gt": {
                    "source": "current_general_ball",
                    "status": item["general_ball_status"],
                    "detections": item["general_ball_candidates"],
                },
            }
        )
    return rows


def build_manifest(video: Path, observations: dict, rows: list[dict]) -> dict:
    metadata = observations["metadata"]
    return {
        "benchmark_id": "CANONICAL_EYE_TEST_01",
        "purpose": "held-out pretrained basketball perception test",
        "held_out_test": True,
        "training_allowed": False,
        "fine_tuning_allowed": False,
        "threshold_tuning_allowed": False,
        "source_video": str(video.resolve()),
        "source_video_sha256": sha256(video),
        "source_video_size_bytes": video.stat().st_size,
        "frame_count": len(rows),
        "fps": metadata["fps"],
        "width_px": metadata["width"],
        "height_px": metadata["height"],
        "duration_s": metadata["duration"],
        "gt_status": "AWAITING_HUMAN_VERIFICATION",
        "preannotations_are_ground_truth": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    frames = observations["dense_ball_observations"]["frames"]
    if len(frames) != 150:
        raise ValueError(f"Expected 150 canonical frames, found {len(frames)}")
    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    args.output.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output / "frames"
    frames_dir.mkdir(exist_ok=True)
    extract_source_frames(args.video, frames_dir, len(frames))
    rows = build_rows(observations, frames_dir)
    manifest = build_manifest(args.video, observations, rows)
    (args.output / "canonical_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gt_path = args.output / "canonical_gt_draft.json"
    if not gt_path.exists():
        gt_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "gt_review.html").write_text(review_page(rows), encoding="utf-8")
    (args.output / "README.txt").write_text(
        "CANONICAL_EYE_TEST_01 is held-out test data (留出测试数据).\n"
        "Open gt_review.html and review all 150 frames. Model boxes are hints only, never GT.\n"
        "Use Export verified GT (导出已核实真值) when progress reaches 150/150.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "frames": len(rows), "sha256": manifest["source_video_sha256"]}, indent=2))


def review_page(rows: list[dict]) -> str:
    data = json.dumps(rows, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>CANONICAL_EYE_TEST_01 GT（真实标注）</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;font-family:system-ui,sans-serif;color:#162033;background:#edf2f7}}
header{{padding:12px 18px;background:#14213d;color:#fff;display:flex;gap:18px;align-items:center;flex-wrap:wrap}}
header h1{{font-size:19px;margin:0}} main{{display:grid;grid-template-columns:minmax(620px,1fr) 390px;gap:14px;padding:14px}}
.stage{{background:#090f1b;border-radius:9px;min-height:650px;display:flex;align-items:center;justify-content:center;overflow:auto}}
canvas{{max-width:100%;max-height:80vh;cursor:crosshair}} aside{{background:#fff;border-radius:9px;padding:14px;overflow:auto;max-height:86vh}}
button,.button{{font:inherit;padding:8px 10px;margin:3px;border:1px solid #94a3b8;border-radius:6px;background:#fff;cursor:pointer}}
button.primary{{background:#087f5b;color:#fff;border-color:#087f5b}} button.no{{background:#b42318;color:#fff;border-color:#b42318}}
button.amb{{background:#b54708;color:#fff;border-color:#b54708}} input[type=range]{{width:100%}}
.group{{padding:9px 0;border-top:1px solid #e2e8f0}} .meta,.small{{font-size:13px;line-height:1.5}} .small{{color:#526075}}
label.flag{{display:inline-block;margin:4px 8px 4px 0}} textarea{{width:100%;min-height:60px}}
@media(max-width:950px){{main{{grid-template-columns:1fr}}.stage{{min-height:430px}}}}
</style></head><body>
<header><h1>CANONICAL_EYE_TEST_01 · Human GT Review（人工真值审核）</h1><strong id="progress"></strong><span id="title"></span></header>
<main><section class="stage"><canvas id="canvas"></canvas></section><aside>
<div><button id="prev">← Previous（上一帧）</button><button id="next">Next（下一帧）→</button></div>
<input id="slider" type="range" min="0" max="{len(rows)-1}" value="0"><div id="meta" class="meta"></div>
<div class="group"><b>Ball visibility（篮球可见性）</b><br>
<button class="primary" id="yes">YES + draw bbox（可见并标框）</button><button class="no" id="no">NO（不可见）</button><button class="amb" id="unclear">AMBIGUOUS（不确定）</button><button id="clear">Clear（清除）</button></div>
<div class="group"><label><input id="hint" type="checkbox"> Show model hints（显示模型提示框，非真值）</label><br>
<label class="flag"><input id="occluded" type="checkbox"> Occluded（遮挡）</label>
<label class="flag"><input id="ambiguous" type="checkbox"> Ambiguous（模糊）</label><br>
<label class="flag"><input id="near_hand" type="checkbox"> Near hand（靠近手部）</label>
<label class="flag"><input id="free_flight" type="checkbox"> Free flight（自由飞行）</label>
<label class="flag"><input id="near_rim" type="checkbox"> Near rim（靠近篮筐）</label></div>
<div class="group"><label>Notes（备注）<textarea id="notes"></textarea></label></div>
<div class="group"><button id="export">Export verified GT（导出已核实真值）</button><label class="button">Import（导入）<input id="import" type="file" accept="application/json" hidden></label></div>
<p class="small">红框是人工 GT（真实标注）；青框仅是 General Ball（通用篮球检测）的预标注提示。选择 YES（可见）后拖动画框。←/→ 翻帧，Y=可见，N=不可见，A=不确定。</p>
</aside></main><script>
const INITIAL={data}, KEY="p-eye-01-canonical-human-gt-v1";
let rows=JSON.parse(localStorage.getItem(KEY)||"null")||INITIAL,index=0,img=new Image(),start=null;
const canvas=document.querySelector("#canvas"),ctx=canvas.getContext("2d"),slider=document.querySelector("#slider"),notes=document.querySelector("#notes");
const flags=["occluded","ambiguous","near_hand","free_flight","near_rim"],cur=()=>rows[index];
function save(){{localStorage.setItem(KEY,JSON.stringify(rows))}}
function done(){{return rows.filter(r=>r.review_status==="VERIFIED").length}}
function box(b,color,label){{if(!b)return;ctx.strokeStyle=color;ctx.lineWidth=3;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.fillStyle=color;ctx.font="16px sans-serif";ctx.fillText(label,b[0],Math.max(18,b[1]-5))}}
function paint(){{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0);if(document.querySelector("#hint").checked)cur().preannotation_non_gt.detections.forEach(d=>box(d.bbox,"#22d3ee","MODEL HINT（模型提示） "+d.confidence.toFixed(3)));box(cur().gt_bbox_xyxy,"#ff334f","HUMAN GT（人工真值）")}}
img.onload=paint;
function render(){{const r=cur();slider.value=index;document.querySelector("#progress").textContent="Verified（已核实） "+done()+" / "+rows.length;document.querySelector("#title").textContent="Frame（帧） "+r.frame_index+" · "+r.timestamp_s.toFixed(3)+" s · "+r.review_status;document.querySelector("#meta").innerHTML="<b>Frame（帧）</b> "+r.frame_index+" / "+(rows.length-1)+"<br><b>Timestamp（时间戳）</b> "+r.timestamp_s.toFixed(3)+" s<br><b>Visibility（可见性）</b> "+(r.ball_visible??"TODO（待审核）")+"<br><b>Model hint status（模型提示状态）</b> "+r.preannotation_non_gt.status;notes.value=r.notes||"";flags.forEach(f=>document.querySelector("#"+f).checked=r[f]===true);img.src=r.image}}
function go(n){{index=Math.max(0,Math.min(rows.length-1,n));start=null;render()}}
function verify(value){{const r=cur();r.review_status="VERIFIED";r.ball_visible=value;if(value!==true){{r.gt_bbox_xyxy=null;r.gt_center_xy=null;r.gt_apparent_diameter_px=null}}save();render()}}
document.querySelector("#prev").onclick=()=>go(index-1);document.querySelector("#next").onclick=()=>go(index+1);slider.oninput=e=>go(+e.target.value);
document.querySelector("#yes").onclick=()=>verify(true);document.querySelector("#no").onclick=()=>verify(false);document.querySelector("#unclear").onclick=()=>{{verify(null);cur().ambiguous=true;save();render()}};
document.querySelector("#clear").onclick=()=>{{Object.assign(cur(),{{review_status:"TODO",ball_visible:null,gt_bbox_xyxy:null,gt_center_xy:null,gt_apparent_diameter_px:null,occluded:null,ambiguous:null,near_hand:null,free_flight:null,near_rim:null,notes:""}});save();render()}};
document.querySelector("#hint").onchange=paint;flags.forEach(f=>document.querySelector("#"+f).onchange=e=>{{cur()[f]=e.target.checked;save()}});notes.onchange=()=>{{cur().notes=notes.value;save()}};
function point(e){{const r=canvas.getBoundingClientRect();return[(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]}}
canvas.onmousedown=e=>{{if(cur().ball_visible!==true)return;start=point(e)}};canvas.onmousemove=e=>{{if(!start)return;paint();const p=point(e);ctx.strokeStyle="#ff334f";ctx.lineWidth=3;ctx.strokeRect(start[0],start[1],p[0]-start[0],p[1]-start[1])}};
canvas.onmouseup=e=>{{if(!start)return;const p=point(e),b=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])];start=null;if(b[2]-b[0]<2||b[3]-b[1]<2)return;cur().gt_bbox_xyxy=b.map(v=>+v.toFixed(2));cur().gt_center_xy=[+((b[0]+b[2])/2).toFixed(2),+((b[1]+b[3])/2).toFixed(2)];cur().gt_apparent_diameter_px=+Math.min(b[2]-b[0],b[3]-b[1]).toFixed(2);save();paint()}};
document.onkeydown=e=>{{if(e.target.tagName==="TEXTAREA")return;if(e.key==="ArrowLeft")go(index-1);if(e.key==="ArrowRight")go(index+1);if(e.key.toLowerCase()==="y")verify(true);if(e.key.toLowerCase()==="n")verify(false);if(e.key.toLowerCase()==="a")document.querySelector("#unclear").click()}};
document.querySelector("#export").onclick=()=>{{const incomplete=rows.filter(r=>r.review_status!=="VERIFIED").length;if(incomplete&&!confirm("Still "+incomplete+" TODO frames（仍有待审核帧）. Export draft anyway（仍导出草稿）?"))return;const blob=new Blob([JSON.stringify(rows,null,2)],{{type:"application/json"}}),a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=incomplete?"canonical_gt_incomplete.json":"canonical_gt_verified.json";a.click();URL.revokeObjectURL(a.href)}};
document.querySelector("#import").onchange=e=>{{const f=e.target.files[0];if(!f)return;const rd=new FileReader();rd.onload=()=>{{const x=JSON.parse(rd.result);if(!Array.isArray(x)||x.length!==150)throw Error("Expected 150 frames（必须为150帧）");rows=x;save();render()}};rd.readAsText(f)}};render();
</script></body></html>"""


if __name__ == "__main__":
    main()

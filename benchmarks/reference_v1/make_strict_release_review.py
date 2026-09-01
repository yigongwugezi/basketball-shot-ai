from __future__ import annotations

import argparse
import csv
import json
import tempfile
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LABEL_ROOT = ROOT / "datasets" / "annotations"
PROCESSED_ROOT = ROOT / "datasets" / "processed" / "yolo_release_ball"
DEFAULT_RANGES = Path(
    r"E:\BasketballShotAI\tools\locateanything_local\batch_runs"
    r"\release_ball_v2_usable_ranges\phase0_range_select\selected_ranges.csv"
)
DEFAULT_FRAME_MANIFEST = Path(
    r"E:\BasketballShotAI\tools\locateanything_local\batch_runs"
    r"\release_ball_v2_usable_ranges\phase1_frames.csv"
)

SOURCE_MAP = {
    "BILI_003_A": r"E:\BasketballShotAI\raw\confirmed_videos\BILI_003_A_BV1d84y1G7zq.mp4",
    "BILI_005_A": r"E:\BasketballShotAI\raw\confirmed_videos\BILI_005_A_BV1Re4y1K7Ey.mp4",
    "NEW_001": r"E:\BasketballShotAI\raw\confirmed_videos\IMG_7212.MOV",
    "NEW_002": r"E:\BasketballShotAI\raw\confirmed_videos\IMG_7215.MOV",
    "NEW_003": r"E:\BasketballShotAI\raw\confirmed_videos\IMG_7216.MOV",
    "NEW_004": r"E:\BasketballShotAI\raw\confirmed_videos\IMG_7218.MOV",
    "NEW_005": r"E:\BasketballShotAI\raw\confirmed_videos\IMG_7219.MP4",
    "NEW_006": r"E:\BasketballShotAI\raw\confirmed_videos\IMG_7221.MOV",
    "NEW_009": r"E:\BasketballShotAI\raw\confirmed_videos\IMG_7226.MP4",
    "NEW_010": r"E:\BasketballShotAI\raw\confirmed_videos\IMG_7227.MP4",
    "NEW_012": r"E:\BasketballShotAI\raw\confirmed_videos\IMG_7235.MOV",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _file_url(path: Path) -> str:
    return path.resolve().as_uri() if path.is_file() else ""


def _metadata() -> dict[tuple[str, int], Path]:
    rows = _read_csv(PROCESSED_ROOT / "metadata.csv")
    return {
        (row["clip_id"], int(row["frame_index"])): PROCESSED_ROOT / row["yolo_image_file"]
        for row in rows
    }


def _phase_frames(path: Path) -> tuple[dict[tuple[str, int, int], dict[str, str]], dict[str, float]]:
    frames: dict[tuple[str, int, int], dict[str, str]] = {}
    fps_by_name: dict[str, float] = {}
    if not path.is_file():
        return frames, fps_by_name
    for row in _read_csv(path):
        key = (row["source_video_name"], int(row["shot_number"]), int(row["frame_index"]))
        image_path = path.parent / row["image_path"]
        frames[key] = {**row, "image_url": _file_url(image_path)}
        fps_by_name[row["source_video_name"]] = float(row["fps"])
    return frames, fps_by_name


def collect_candidates(ranges_path: Path, frame_manifest: Path) -> list[dict[str, Any]]:
    processed = _metadata()
    phase_frames, fps_by_name = _phase_frames(frame_manifest)
    candidates: list[dict[str, Any]] = []

    for batch in ("release_ball_batch_001", "release_ball_batch_003"):
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in _read_csv(LABEL_ROOT / batch / "labels.csv"):
            grouped[row["clip_id"]].append(row)
        for clip_id, rows in sorted(grouped.items()):
            rows.sort(key=lambda row: int(row["frame_index"]))
            strict = next(
                (int(row["frame_index"]) for row in rows if row["is_ball_release_frame"] == "yes"),
                None,
            )
            pose = next(
                (int(row["frame_index"]) for row in rows if row["is_release_pose_frame"] == "yes"),
                None,
            )
            strict_confidence = next(
                (row["label_confidence"] for row in rows if row["is_ball_release_frame"] == "yes"),
                "",
            )
            source_path = SOURCE_MAP.get(clip_id, "")
            source_name = Path(source_path).name if source_path else ""
            frames = []
            for row in rows:
                frame_index = int(row["frame_index"])
                frames.append(
                    {
                        "frame_index": frame_index,
                        "image_url": _file_url(processed.get((clip_id, frame_index), Path("missing"))),
                        "ball_visible": row["ball_visible"],
                        "ball_occluded": row["occlusion"],
                        "ball_uncertain": row["ball_visibility_quality"] == "poor",
                        "ball_center_x": row["ball_center_x"],
                        "ball_center_y": row["ball_center_y"],
                    }
                )
            candidates.append(
                {
                    "sample_id": clip_id,
                    "source_video_id": clip_id,
                    "source_video_path": source_path,
                    "source_video_url": _file_url(Path(source_path)) if source_path else "",
                    "clip_start_frame": int(rows[0]["frame_index"]),
                    "clip_end_frame": int(rows[-1]["frame_index"]),
                    "fps": fps_by_name.get(source_name),
                    "pose_release_frame": pose,
                    "contact_last_supported_frame": None,
                    "separation_candidate_frame": None,
                    "strict_release_frame": strict,
                    "uncertainty_frames": 0 if strict_confidence == "high" else 1 if strict is not None else None,
                    "visibility_status": "reviewed_legacy",
                    "review_status": "legacy_manual_annotation",
                    "reviewer_1": "unknown_legacy_annotator",
                    "reviewer_1_status": "imported_legacy",
                    "reviewer_2": "",
                    "reviewer_2_status": "pending",
                    "reviewer_2_contact_last_supported_frame": None,
                    "reviewer_2_separation_candidate_frame": None,
                    "reviewer_2_strict_release_frame": None,
                    "reviewer_2_uncertainty_frames": None,
                    "reviewer_2_visibility_status": "pending",
                    "reviewer_2_review_status": "pending",
                    "split_role": "development_training_overlap",
                    "notes": "Existing human release/ball labels; used by v1 detector and not independent evaluation.",
                    "frames": frames,
                }
            )

    if ranges_path.is_file():
        for row in _read_csv(ranges_path):
            if row["source_video_name"] not in {"IMG_7221.MOV", "IMG_7222.MP4"}:
                continue
            shot = int(row["shot_number"])
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            frames = []
            for frame_index in range(start, end + 1):
                frame = phase_frames.get((row["source_video_name"], shot, frame_index), {})
                frames.append(
                    {
                        "frame_index": frame_index,
                        "image_url": frame.get("image_url", ""),
                        "ball_visible": "uncertain",
                        "ball_occluded": "unknown",
                        "ball_uncertain": True,
                        "ball_center_x": "",
                        "ball_center_y": "",
                    }
                )
            candidates.append(
                {
                    "sample_id": f"{Path(row['source_video_name']).stem}_shot_{shot:02d}",
                    "source_video_id": row["source_video_id"],
                    "source_video_path": row["source_video_path"],
                    "source_video_url": _file_url(Path(row["source_video_path"])),
                    "clip_start_frame": start,
                    "clip_end_frame": end,
                    "fps": fps_by_name.get(row["source_video_name"]),
                    "pose_release_frame": None,
                    "contact_last_supported_frame": None,
                    "separation_candidate_frame": None,
                    "strict_release_frame": None,
                    "uncertainty_frames": None,
                    "visibility_status": "pending",
                    "review_status": "pending",
                    "reviewer_1": "",
                    "reviewer_1_status": "pending",
                    "reviewer_2": "",
                    "reviewer_2_status": "pending",
                    "reviewer_2_contact_last_supported_frame": None,
                    "reviewer_2_separation_candidate_frame": None,
                    "reviewer_2_strict_release_frame": None,
                    "reviewer_2_uncertainty_frames": None,
                    "reviewer_2_visibility_status": "pending",
                    "reviewer_2_review_status": "pending",
                    "split_role": (
                        "development" if row["source_video_name"] == "IMG_7221.MOV" else "frozen_evaluation"
                    ),
                    "notes": "Human-selected shot range; strict release and ball centers pending.",
                    "frames": frames,
                }
            )
    return candidates


SHOT_COLUMNS = [
    "sample_id",
    "source_video_id",
    "source_video_path",
    "clip_start_frame",
    "clip_end_frame",
    "fps",
    "pose_release_frame",
    "contact_last_supported_frame",
    "separation_candidate_frame",
    "strict_release_frame",
    "uncertainty_frames",
    "visibility_status",
    "review_status",
    "reviewer_1",
    "reviewer_1_status",
    "reviewer_2",
    "reviewer_2_status",
    "reviewer_2_contact_last_supported_frame",
    "reviewer_2_separation_candidate_frame",
    "reviewer_2_strict_release_frame",
    "reviewer_2_uncertainty_frames",
    "reviewer_2_visibility_status",
    "reviewer_2_review_status",
    "split_role",
    "notes",
]


def write_seed_csv(candidates: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHOT_COLUMNS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in SHOT_COLUMNS} for row in candidates)


HTML = r"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strict release micro-GT review</title>
<style>
body{margin:0;background:#eee8dc;color:#18221d;font:14px Segoe UI,sans-serif}header{position:sticky;top:0;z-index:2;background:#18221d;color:white;padding:10px;display:flex;gap:8px;align-items:center}header strong{margin-right:auto}.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(330px,1fr);gap:16px;padding:16px}.card{background:#fffdf7;border:1px solid #d2c8b5;border-radius:12px;padding:14px}.image-wrap{position:relative;width:fit-content;max-width:100%;margin:auto}img{display:block;max-width:100%;max-height:70vh;cursor:crosshair;background:#111}.marker{position:absolute;width:14px;height:14px;border:3px solid #ffcf25;border-radius:50%;transform:translate(-50%,-50%);box-shadow:0 0 0 2px #111;pointer-events:none}input,select,textarea,button{font:inherit;padding:7px}label{display:grid;gap:4px;margin:8px 0}.fields{display:grid;grid-template-columns:1fr 1fr;gap:8px}textarea{min-height:70px}.frames{display:flex;gap:5px;align-items:center;justify-content:center;margin:10px}.meta{font-size:12px;word-break:break-all;color:#655}.pending{color:#a23}.ok{color:#176b4d}@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style>
<header><strong>Strict release micro-GT</strong><span id="progress"></span><button onclick="moveShot(-1)">Previous shot</button><button onclick="moveShot(1)">Next shot</button><button onclick="downloadAll()">Export CSV</button></header>
<main class="grid"><section class="card"><div class="image-wrap"><img id="image"><i id="marker" class="marker" hidden></i></div><div class="frames"><button onclick="moveFrame(-1)">-1</button><b id="frame"></b><button onclick="moveFrame(1)">+1</button></div><div class="meta" id="meta"></div><p>Click the basketball center. Use visible/occluded/uncertain honestly; do not force a strict frame when evidence is ambiguous.</p><div class="fields"><label>Ball visible<select id="visible"><option>yes</option><option>no</option><option>uncertain</option></select></label><label>Occlusion<select id="occlusion"><option>none</option><option>partial</option><option>heavy</option><option>unknown</option></select></label></div></section><aside class="card"><h2 id="name"></h2><div class="fields"><label>Contact last supported<input id="contact" type="number"></label><label>Separation candidate<input id="separation" type="number"></label><label>Strict release<input id="strict" type="number" placeholder="leave blank if unknown"></label><label>Uncertainty frames<input id="uncertainty" type="number" min="0"></label><label>Visibility status<select id="visibility"><option>clear</option><option>occluded</option><option>blurred</option><option>ambiguous</option><option>insufficient_data</option><option>pending</option><option>reviewed_legacy</option></select></label><label>Review status<select id="status"><option>pending</option><option>reviewed</option><option>ambiguous</option><option>insufficient_data</option><option>legacy_manual_annotation</option></select></label><label>Reviewer 1<input id="reviewer1"></label><label>Reviewer 2<input id="reviewer2" placeholder="independent reviewer only"></label></div><details><summary>Independent reviewer 2 labels</summary><div class="fields"><label>R2 contact last<input id="r2contact" type="number"></label><label>R2 separation<input id="r2separation" type="number"></label><label>R2 strict release<input id="r2strict" type="number"></label><label>R2 uncertainty<input id="r2uncertainty" type="number" min="0"></label><label>R2 visibility<select id="r2visibility"><option>pending</option><option>clear</option><option>occluded</option><option>blurred</option><option>ambiguous</option><option>insufficient_data</option></select></label><label>R2 status<select id="r2status"><option>pending</option><option>reviewed</option><option>ambiguous</option><option>insufficient_data</option></select></label></div></details><label>Notes<textarea id="notes"></textarea></label><p id="split"></p><p>Reviewer 2 remains pending unless a real second person independently reviews it. Algorithm output is never reviewer 2.</p></aside></main>
<script>
const rows=__DATA__,key="strict-release-micro-gt-v1";let shot=0,frame=0;
const ids=["contact","separation","strict","uncertainty","visibility","status","reviewer1","reviewer2","r2contact","r2separation","r2strict","r2uncertainty","r2visibility","r2status","notes"];
function restore(){try{const saved=JSON.parse(localStorage.getItem(key)||"{}");rows.forEach(r=>Object.assign(r,saved[r.sample_id]||{}))}catch(_){}}
function save(){const out={};rows.forEach(r=>out[r.sample_id]={contact_last_supported_frame:r.contact_last_supported_frame,separation_candidate_frame:r.separation_candidate_frame,strict_release_frame:r.strict_release_frame,uncertainty_frames:r.uncertainty_frames,visibility_status:r.visibility_status,review_status:r.review_status,reviewer_1:r.reviewer_1,reviewer_1_status:r.reviewer_1_status,reviewer_2:r.reviewer_2,reviewer_2_status:r.reviewer_2_status,reviewer_2_contact_last_supported_frame:r.reviewer_2_contact_last_supported_frame,reviewer_2_separation_candidate_frame:r.reviewer_2_separation_candidate_frame,reviewer_2_strict_release_frame:r.reviewer_2_strict_release_frame,reviewer_2_uncertainty_frames:r.reviewer_2_uncertainty_frames,reviewer_2_visibility_status:r.reviewer_2_visibility_status,reviewer_2_review_status:r.reviewer_2_review_status,notes:r.notes,frames:r.frames});localStorage.setItem(key,JSON.stringify(out))}
function current(){return rows[shot]}function currentFrame(){return current().frames[frame]}
function renderMarker(){const f=currentFrame(),m=document.querySelector("#marker"),img=document.querySelector("#image"),valid=f.ball_center_x!==""&&f.ball_center_y!==""&&img.naturalWidth;m.hidden=!valid;if(valid){m.style.left=`${100*Number(f.ball_center_x)/img.naturalWidth}%`;m.style.top=`${100*Number(f.ball_center_y)/img.naturalHeight}%`}}
function render(){const r=current(),f=currentFrame();document.querySelector("#progress").textContent=`${shot+1}/${rows.length} | reviewed ${rows.filter(x=>x.review_status!=="pending").length}`;document.querySelector("#name").textContent=r.sample_id;document.querySelector("#frame").textContent=`Frame ${f.frame_index}`;document.querySelector("#meta").textContent=`${r.source_video_path} | ${r.fps??"fps pending"} fps | center ${f.ball_center_x||"?"}, ${f.ball_center_y||"?"}`;document.querySelector("#image").src=f.image_url;document.querySelector("#visible").value=f.ball_visible;document.querySelector("#occlusion").value=f.ball_occluded;document.querySelector("#contact").value=r.contact_last_supported_frame??"";document.querySelector("#separation").value=r.separation_candidate_frame??"";document.querySelector("#strict").value=r.strict_release_frame??"";document.querySelector("#uncertainty").value=r.uncertainty_frames??"";document.querySelector("#visibility").value=r.visibility_status;document.querySelector("#status").value=r.review_status;document.querySelector("#reviewer1").value=r.reviewer_1;document.querySelector("#reviewer2").value=r.reviewer_2;document.querySelector("#r2contact").value=r.reviewer_2_contact_last_supported_frame??"";document.querySelector("#r2separation").value=r.reviewer_2_separation_candidate_frame??"";document.querySelector("#r2strict").value=r.reviewer_2_strict_release_frame??"";document.querySelector("#r2uncertainty").value=r.reviewer_2_uncertainty_frames??"";document.querySelector("#r2visibility").value=r.reviewer_2_visibility_status;document.querySelector("#r2status").value=r.reviewer_2_review_status;document.querySelector("#notes").value=r.notes;document.querySelector("#split").textContent=`Split role: ${r.split_role}; reviewer 2: ${r.reviewer_2_status}`;renderMarker()}
function moveShot(d){shot=Math.max(0,Math.min(rows.length-1,shot+d));frame=0;render()}function moveFrame(d){frame=Math.max(0,Math.min(current().frames.length-1,frame+d));render()}
function numberOrNull(v){return v===""?null:Number(v)}
ids.forEach(id=>document.querySelector("#"+id).addEventListener("change",e=>{const r=current(),map={contact:"contact_last_supported_frame",separation:"separation_candidate_frame",strict:"strict_release_frame",uncertainty:"uncertainty_frames",visibility:"visibility_status",status:"review_status",reviewer1:"reviewer_1",reviewer2:"reviewer_2",r2contact:"reviewer_2_contact_last_supported_frame",r2separation:"reviewer_2_separation_candidate_frame",r2strict:"reviewer_2_strict_release_frame",r2uncertainty:"reviewer_2_uncertainty_frames",r2visibility:"reviewer_2_visibility_status",r2status:"reviewer_2_review_status",notes:"notes"},field=map[id];r[field]=["contact","separation","strict","uncertainty","r2contact","r2separation","r2strict","r2uncertainty"].includes(id)?numberOrNull(e.target.value):e.target.value;if(id==="status"&&e.target.value!=="pending")r.reviewer_1_status="reviewed";if(id==="r2status"&&e.target.value!=="pending")r.reviewer_2_status="reviewed";save();render()}));
document.querySelector("#visible").addEventListener("change",e=>{currentFrame().ball_visible=e.target.value;save()});document.querySelector("#occlusion").addEventListener("change",e=>{currentFrame().ball_occluded=e.target.value;save()});document.querySelector("#image").addEventListener("click",e=>{const rect=e.target.getBoundingClientRect(),f=currentFrame();f.ball_center_x=Math.round((e.clientX-rect.left)*e.target.naturalWidth/rect.width);f.ball_center_y=Math.round((e.clientY-rect.top)*e.target.naturalHeight/rect.height);f.ball_visible="yes";f.ball_uncertain=false;save();render()});
document.querySelector("#image").addEventListener("load",renderMarker);
function cell(v){return `"${String(v??"").replaceAll('"','""')}"`}function download(name,columns,data){const lines=[columns.map(cell).join(","),...data.map(r=>columns.map(c=>cell(r[c])).join(","))],a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["\ufeff"+lines.join("\r\n")],{type:"text/csv"}));a.download=name;a.click();URL.revokeObjectURL(a.href)}
function downloadAll(){const shots=rows.map(r=>{const x={...r};delete x.frames;return x}),balls=rows.flatMap(r=>r.frames.map(f=>({sample_id:r.sample_id,source_video_id:r.source_video_id,...f,reviewer_1:r.reviewer_1,review_status:r.review_status})));download("strict_release_reviewer_1.csv",__SHOT_COLUMNS__,shots);setTimeout(()=>download("strict_release_ball_centers.csv",["sample_id","source_video_id","frame_index","ball_visible","ball_occluded","ball_uncertain","ball_center_x","ball_center_y","reviewer_1","review_status"],balls),300)}
restore();render();
</script>"""


def write_review_page(candidates: list[dict[str, Any]], path: Path) -> None:
    payload = json.dumps(candidates, ensure_ascii=False).replace("</", "<\\/")
    columns = json.dumps(SHOT_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HTML.replace("__DATA__", payload).replace("__SHOT_COLUMNS__", columns), encoding="utf-8")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "review.html"
        row = {key: "" for key in SHOT_COLUMNS}
        row.update({"sample_id": "test", "review_status": "pending", "frames": [{"frame_index": 1, "image_url": ""}]})
        write_review_page([row], output)
        assert "strict_release_reviewer_1.csv" in output.read_text(encoding="utf-8")
    print("strict release review self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the strict-release micro-GT review page")
    parser.add_argument("--ranges", type=Path, default=DEFAULT_RANGES)
    parser.add_argument("--frame-manifest", type=Path, default=DEFAULT_FRAME_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp" / "reference_validation_closure")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    candidates = collect_candidates(args.ranges, args.frame_manifest)
    if not 20 <= len(candidates) <= 40:
        raise ValueError(f"Expected 20-40 micro-GT candidates, found {len(candidates)}")
    seed = args.output_dir / "micro_gt_seed.csv"
    review = args.output_dir / "strict_release_review.html"
    write_seed_csv(candidates, seed)
    write_review_page(candidates, review)
    print(f"candidates: {len(candidates)}")
    print(f"seed: {seed}")
    print(f"review: {review}")
    if args.open:
        webbrowser.open(review.resolve().as_uri())


if __name__ == "__main__":
    main()

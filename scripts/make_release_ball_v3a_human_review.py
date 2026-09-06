#!/usr/bin/env python3
"""Create a browser-based human bbox review task for the frozen 217 V3A frames."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PSEUDO_LABELS = ROOT / "datasets/annotations/release_ball_v3a_pseudo/pseudo_labels.json"
OUTPUT = ROOT / "artifacts/release_ball_v3a_human_review"


def main() -> None:
    rows = json.loads(PSEUDO_LABELS.read_text(encoding="utf-8"))
    if len(rows) != 217:
        raise ValueError(f"Expected 217 frozen V3A frames, found {len(rows)}")
    if OUTPUT.exists():
        raise FileExistsError(f"Review task already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    task_rows = [
        {
            "clip_id": row["clip_id"],
            "frame_index": row["frame_index"],
            "image": f"../release_ball_v3a_pseudo_review_final/originals/{row['clip_id']}/{Path(row['image_path']).name}",
            "human_status": None,
            "human_bbox": None,
            "notes": "",
            "reference": {
                "pseudo_status": row["label_type"],
                "pseudo_bbox": row["bbox"],
                "v2_candidates": row["v2_candidates"],
                "coco_candidates": row["teacher_sports_ball_candidates"],
                "cross_model_iou": row["cross_model_iou"],
                "temporal_support": row["temporal_support"],
            },
        }
        for row in rows
    ]
    (OUTPUT / "task.json").write_text(json.dumps(task_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "README.txt").write_text(
        "Open annotation.html. Labels stay in browser storage until Export downloads a JSON copy; Import resumes that JSON.\n"
        "This task never changes pseudo_labels.json or a training dataset.\n",
        encoding="utf-8",
    )
    (OUTPUT / "annotation.html").write_text(page(task_rows), encoding="utf-8")
    print(f"Human review task created: {OUTPUT}\nframes={len(task_rows)}")


def page(rows: list[dict]) -> str:
    data = json.dumps(rows, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Release Ball V3A 人工复核</title>
<style>
body {{ margin:0; font-family:system-ui,sans-serif; color:#172033; background:#edf1f7; }}
header {{ padding:14px 22px; background:#172033; color:white; display:flex; gap:20px; align-items:center; flex-wrap:wrap; }}
header h1 {{ font-size:20px; margin:0; }} main {{ display:grid; grid-template-columns:minmax(600px,1fr) 330px; gap:16px; padding:16px; }}
.stage {{ background:#111827; border-radius:8px; min-height:660px; display:flex; justify-content:center; align-items:center; overflow:auto; }}
canvas {{ max-width:100%; max-height:78vh; cursor:crosshair; }} aside {{ background:white; border-radius:8px; padding:16px; }}
button,.button {{ font:inherit; padding:9px 12px; margin:4px 3px 4px 0; border:1px solid #94a3b8; border-radius:6px; background:white; cursor:pointer; }}
button.primary {{ background:#0f766e; color:white; border-color:#0f766e; }} button.warn {{ background:#b45309; color:white; border-color:#b45309; }}
.meta {{ line-height:1.55; font-size:14px; margin:10px 0; }} .hint {{ font-size:13px; padding:10px; background:#f1f5f9; border-radius:6px; white-space:pre-wrap; }}
textarea {{ width:100%; box-sizing:border-box; min-height:88px; }} input[type=range] {{ width:100%; }} .small {{ font-size:12px; color:#475569; }} kbd {{ background:#e2e8f0; padding:2px 5px; border-radius:3px; }}
@media(max-width:900px) {{ main {{ grid-template-columns:1fr; }} .stage {{ min-height:420px; }} }}
</style></head><body>
<header><h1>Release Ball V3A — 217 帧人工复核</h1><strong id="progress"></strong><span id="frameTitle"></span></header>
<main><section class="stage"><canvas id="canvas"></canvas></section><aside>
<div><button id="prev">← 上一张</button><button id="next">下一张 →</button></div>
<input id="slider" type="range" min="0" max="{len(rows) - 1}" value="0">
<div class="meta" id="meta"></div>
<div><button class="primary" id="yes">有球：拖动标框</button><button class="warn" id="no">无球</button><button id="clear">清除此帧</button></div>
<div><label><input id="showHints" type="checkbox"> 显示模型参考框（非人工标签）</label></div>
<p class="small">先选“有球”，再在图上拖框；无球直接点按钮。<kbd>←</kbd>/<kbd>→</kbd> 翻页，<kbd>N</kbd> 无球，<kbd>Y</kbd> 有球。</p>
<label>备注<textarea id="notes" placeholder="可选"></textarea></label>
<p><button id="export">导出人工复核 JSON</button><label class="button">导入已有 JSON<input id="import" type="file" accept="application/json" hidden></label></p>
<div class="hint" id="hint"></div>
</aside></main>
<script>
const INITIAL = {data};
const STORAGE_KEY = "release-ball-v3a-human-review-v1";
let rows = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null") || INITIAL;
let index = 0, image = new Image(), drawing = null;
const canvas = document.querySelector("#canvas"), ctx = canvas.getContext("2d");
const slider = document.querySelector("#slider"), notes = document.querySelector("#notes");
const current = () => rows[index];
function save() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(rows)); }}
function statusText(row) {{ return row.human_status === "yes" ? "有球（已人工标注）" : row.human_status === "no" ? "无球（已人工确认）" : "待复核"; }}
function progress() {{ return rows.filter(r => r.human_status).length; }}
function drawBox(box, color, label) {{
  if (!box) return; const sx=canvas.width/image.naturalWidth, sy=canvas.height/image.naturalHeight;
  const x1=box[0],y1=box[1],x2=box[2],y2=box[3]; ctx.strokeStyle=color; ctx.lineWidth=3; ctx.strokeRect(x1*sx,y1*sy,(x2-x1)*sx,(y2-y1)*sy);
  ctx.fillStyle=color; ctx.font="16px sans-serif"; ctx.fillText(label,x1*sx,Math.max(17,y1*sy-5));
}}
function render() {{
  const row=current(); slider.value=index; notes.value=row.notes || "";
  document.querySelector("#progress").textContent="已完成 "+progress()+" / "+rows.length;
  document.querySelector("#frameTitle").textContent=row.clip_id+" · frame "+row.frame_index+" · "+statusText(row);
  document.querySelector("#meta").innerHTML="<b>Clip</b> "+row.clip_id+"<br><b>Frame</b> "+row.frame_index+"<br><b>人工状态</b> "+statusText(row);
  const ref=row.reference, v2=ref.v2_candidates.length, coco=ref.coco_candidates.length;
  document.querySelector("#hint").textContent="模型参考（不会自动写入人工标签）\\n原 pseudo 状态："+ref.pseudo_status+"\\nV2 candidates："+v2+"；COCO candidates："+coco+"\\n一致性 IoU："+ref.cross_model_iou.toFixed(3)+"；时序支持："+(ref.temporal_support ? "是" : "否");
  image.src=row.image;
}}
function paint() {{
  canvas.width=image.naturalWidth; canvas.height=image.naturalHeight; ctx.drawImage(image,0,0);
  if (document.querySelector("#showHints").checked) {{
    const r=current().reference;
    r.v2_candidates.forEach(c=>drawBox(c.bbox,"#38bdf8","V2 "+c.confidence.toFixed(3)));
    r.coco_candidates.forEach(c=>drawBox(c.bbox,"#facc15","COCO "+c.confidence.toFixed(3)));
  }}
  drawBox(current().human_bbox,"#ef4444","人工篮球框");
}}
image.onload=paint;
function goto(next) {{ index=Math.max(0,Math.min(rows.length-1,next)); drawing=null; render(); }}
document.querySelector("#prev").onclick=()=>goto(index-1); document.querySelector("#next").onclick=()=>goto(index+1);
slider.oninput=e=>goto(Number(e.target.value));
document.querySelector("#yes").onclick=()=>{{ current().human_status="yes"; save(); render(); }};
document.querySelector("#no").onclick=()=>{{ current().human_status="no"; current().human_bbox=null; save(); render(); }};
document.querySelector("#clear").onclick=()=>{{ current().human_status=null; current().human_bbox=null; current().notes=""; save(); render(); }};
document.querySelector("#showHints").onchange=paint;
notes.onchange=()=>{{ current().notes=notes.value; save(); }};
function point(event) {{ const r=canvas.getBoundingClientRect(); return [(event.clientX-r.left)*canvas.width/r.width,(event.clientY-r.top)*canvas.height/r.height]; }}
canvas.onmousedown=event=>{{ if(current().human_status!=="yes") return; drawing=point(event); }};
canvas.onmousemove=event=>{{ if(!drawing) return; paint(); const p=point(event); ctx.strokeStyle="#ef4444";ctx.lineWidth=3;ctx.strokeRect(drawing[0],drawing[1],p[0]-drawing[0],p[1]-drawing[1]); }};
canvas.onmouseup=event=>{{ if(!drawing) return; const p=point(event), sx=image.naturalWidth/canvas.width, sy=image.naturalHeight/canvas.height; const x1=Math.min(drawing[0],p[0])*sx,y1=Math.min(drawing[1],p[1])*sy,x2=Math.max(drawing[0],p[0])*sx,y2=Math.max(drawing[1],p[1])*sy; if(x2-x1>2&&y2-y1>2) current().human_bbox=[+x1.toFixed(2),+y1.toFixed(2),+x2.toFixed(2),+y2.toFixed(2)]; drawing=null;save();paint(); }};
document.onkeydown=e=>{{ if(e.target.tagName==="TEXTAREA") return; if(e.key==="ArrowLeft") goto(index-1); if(e.key==="ArrowRight") goto(index+1); if(e.key.toLowerCase()==="n") {{current().human_status="no";current().human_bbox=null;save();render();}} if(e.key.toLowerCase()==="y") {{current().human_status="yes";save();render();}} }};
document.querySelector("#export").onclick=()=>{{ const blob=new Blob([JSON.stringify(rows,null,2)],{{type:"application/json"}}); const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="release_ball_v3a_human_review.json";a.click();URL.revokeObjectURL(a.href); }};
document.querySelector("#import").onchange=e=>{{ const f=e.target.files[0];if(!f)return;const reader=new FileReader();reader.onload=()=>{{const incoming=JSON.parse(reader.result);if(!Array.isArray(incoming)||incoming.length!==INITIAL.length)throw new Error("不是本任务的 217 帧导出文件");rows=incoming;save();render();}};reader.readAsText(f); }};
render();
</script></body></html>"""


if __name__ == "__main__":
    main()

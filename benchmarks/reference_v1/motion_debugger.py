from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reference_v1.motion import build_motion_representation, trajectory_signals  # noqa: E402


def build(run_dir: Path, *, slow_motion: bool = False, contaminated: bool = False) -> tuple[Path, Path]:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    trajectories = json.loads((run_dir / "evidence" / "pose_trajectories.json").read_text(encoding="utf-8"))
    representation = build_motion_representation(
        report,
        trajectories,
        slow_motion=slow_motion,
        contaminated_research_only=contaminated,
    )
    representation_path = run_dir / "motion_representation_v0.json"
    representation_path.write_text(json.dumps(representation, ensure_ascii=False, indent=2), encoding="utf-8")
    signals = trajectory_signals(trajectories, report["attempt"]["shooting_side"])
    ball = report.get("ball_evidence", {}).get("center_observations", [])
    data: dict[str, Any] = {
        "fps": report["input"]["fps"],
        "frame_count": report["input"]["frame_count"],
        "events": report["events"],
        "phases": report["phases"],
        "signals": {
            "wrist_y": signals["wrist_y"],
            "elbow_angle": signals["elbow_angle"],
            "trunk_angle": signals["trunk_angle"],
            "ball_y": [[row["frame"], row["center"][1]] for row in ball],
        },
        "representation": representation,
    }
    html_path = run_dir / "motion_debugger.html"
    html_path.write_text(render_html(data), encoding="utf-8")
    return representation_path, html_path


def render_html(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Motion Representation V0 Debugger</title>
<style>body{{margin:0;background:#10151d;color:#eaf0f7;font:14px system-ui}}main{{max-width:1200px;margin:auto;padding:18px}}.grid{{display:grid;grid-template-columns:2fr 1fr;gap:16px}}.card{{background:#19212c;border:1px solid #364254;border-radius:9px;padding:12px}}video{{width:100%;max-height:70vh;background:#000}}#timeline{{position:relative;height:90px;background:#0b0f15;margin-top:12px;cursor:pointer}}.phase{{position:absolute;height:25px;top:8px;background:#276a7d99;border-radius:3px;padding:4px;overflow:hidden}}.event{{position:absolute;top:40px;width:2px;height:35px;background:#ffcc66}}.event span{{position:absolute;top:34px;transform:translateX(-50%);white-space:nowrap;font-size:11px}}canvas{{width:100%;height:260px;background:#0b0f15}}table{{width:100%;border-collapse:collapse}}td,th{{padding:6px;border-bottom:1px solid #34404f;text-align:left}}.bad{{color:#ffb17a}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>Shot Motion Representation V0 — Developer Debugger</h1><div class=\"grid\"><section class=\"card\"><video id=\"video\" controls src=\"annotated.mp4\"></video><div id=\"timeline\"></div><h3>Evidence curves</h3><canvas id=\"plot\" width=\"1000\" height=\"260\"></canvas></section><aside class=\"card\"><h2>Selected frame</h2><div id=\"selected\"></div><h2>Events</h2><table id=\"events\"></table><p class=\"bad\">Developer evidence only. No coaching or good/bad action labels.</p></aside></div></main>
<script>const D={encoded},v=document.getElementById('video'),timeline=document.getElementById('timeline'),plot=document.getElementById('plot'),ctx=plot.getContext('2d');const pct=f=>100*f/Math.max(1,D.frame_count-1);for(const [name,p] of Object.entries(D.phases)){{if(p.start_frame==null||p.end_frame==null)continue;const el=document.createElement('div');el.className='phase';el.style.left=pct(p.start_frame)+'%';el.style.width=(pct(p.end_frame)-pct(p.start_frame))+'%';el.textContent=name;timeline.append(el)}}for(const [name,e] of Object.entries(D.events)){{if(e.frame==null)continue;const el=document.createElement('div');el.className='event';el.style.left=pct(e.frame)+'%';el.innerHTML='<span>'+name+'<br>f'+e.frame+'</span>';el.onclick=x=>{{x.stopPropagation();seek(e.frame)}};timeline.append(el)}}timeline.onclick=e=>{{const r=timeline.getBoundingClientRect();seek(Math.round((e.clientX-r.left)/r.width*(D.frame_count-1)))}};function seek(frame){{v.currentTime=frame/D.fps;document.getElementById('selected').textContent='frame '+frame+' · '+(frame/D.fps).toFixed(3)+' s';draw(frame)}}function draw(selected){{ctx.clearRect(0,0,plot.width,plot.height);const colors={{wrist_y:'#45d4ff',elbow_angle:'#ffac4a',trunk_angle:'#c18cff',ball_y:'#76e06f'}}, entries=Object.entries(D.signals).filter(x=>x[1].length);entries.forEach(([name,values],series)=>{{const ys=values.map(x=>x[1]),min=Math.min(...ys),max=Math.max(...ys);ctx.strokeStyle=colors[name];ctx.lineWidth=2;ctx.beginPath();values.forEach(([f,y],i)=>{{const x=f/(D.frame_count-1)*plot.width,py=20+series*55+(y-min)/Math.max(1,max-min)*40;i?ctx.lineTo(x,py):ctx.moveTo(x,py)}});ctx.stroke();ctx.fillStyle=colors[name];ctx.fillText(name,8,32+series*55)}});ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(selected/(D.frame_count-1)*plot.width,0);ctx.lineTo(selected/(D.frame_count-1)*plot.width,plot.height);ctx.stroke()}}v.ontimeupdate=()=>draw(Math.round(v.currentTime*D.fps));const table=document.getElementById('events');table.innerHTML='<tr><th>event</th><th>frame</th><th>status</th></tr>'+Object.entries(D.events).map(([n,e])=>'<tr><td>'+n+'</td><td>'+(e.frame??'—')+'</td><td>'+e.status+'</td></tr>').join('');seek(0);</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone Reference V1 motion debugger")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--slow-motion", action="store_true")
    parser.add_argument("--contaminated-research-only", action="store_true")
    args = parser.parse_args()
    representation, html = build(args.run_dir, slow_motion=args.slow_motion, contaminated=args.contaminated_research_only)
    print(representation)
    print(html)


if __name__ == "__main__":
    main()

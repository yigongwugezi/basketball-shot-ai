from __future__ import annotations

import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "datasets" / "bilibili_clip_plan.csv"
OUT_DIR = Path(r"E:\BasketballShotAI\clips\bilibili")
DOWNKYI_ROOT = Path.home() / "Desktop" / "工具" / "b站视频下载" / "找到downkyi.exe"
DOWNKYI_FFMPEG = DOWNKYI_ROOT / "ffmpeg" / "ffmpeg.exe"


def has_clip_time(row: dict[str, str]) -> bool:
    return bool(row.get("downloaded_file_path") and row.get("start_time") and row.get("end_time"))


def main() -> None:
    if not DOWNKYI_FFMPEG.exists():
        raise SystemExit(f"ffmpeg not found: {DOWNKYI_FFMPEG}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with PLAN_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    cut_count = 0
    for row in rows:
        if not has_clip_time(row):
            continue

        source = Path(row["downloaded_file_path"].strip().strip('"'))
        if not source.exists():
            print(f"SKIP missing file: {row['clip_id']} -> {source}")
            continue

        output_name = f"{row['clip_id']}_{row['bv']}.mp4"
        output_path = OUT_DIR / output_name

        command = [
            str(DOWNKYI_FFMPEG),
            "-y",
            "-ss",
            row["start_time"],
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        if row["end_time"].strip().lower() not in {"end", "last", "finish"}:
            command[4:4] = ["-to", row["end_time"]]
        print(f"CUT {row['clip_id']}: {source.name} -> {output_path.name}")
        subprocess.run(command, check=True)
        cut_count += 1

    print(f"Done. Clips created: {cut_count}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()

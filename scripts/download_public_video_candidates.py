from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "datasets" / "video_sources.csv"
OUT_DIR = ROOT / "datasets" / "raw" / "public" / "video" / "public_candidates"
YTDLP_FALLBACK = Path.home() / ".agent-reach-venv" / "Scripts" / "yt-dlp.exe"


def safe_slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts)[:80] or "video"


def find_ytdlp() -> str:
    found = shutil.which("yt-dlp")
    if found:
        return found
    if YTDLP_FALLBACK.exists():
        return str(YTDLP_FALLBACK)
    raise SystemExit("yt-dlp not found. Install yt-dlp or check ~/.agent-reach-venv/Scripts/yt-dlp.exe")


def load_rows(priority: str, limit: int, platform: str | None) -> list[dict[str, str]]:
    with SOURCE_CSV.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    candidates = [
        row
        for row in rows
        if row.get("download_status") == "not_downloaded"
        and row.get("priority") == priority
        and row.get("platform") in {"YouTube", "Bilibili", "NBA Jr", "University of Richmond"}
        and (platform is None or row.get("platform") == platform)
        and row.get("source_type") != "search_page"
    ]
    return candidates[:limit]


def download(row: dict[str, str], ytdlp: str, dry_run: bool, cookies_from_browser: str | None) -> None:
    platform = row["platform"]
    title = row["title"]
    slug = safe_slug(f"{platform}-{title}")
    target_dir = OUT_DIR / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = target_dir / "source.json"
    metadata_path.write_text(
        json.dumps(row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    command = [
        ytdlp,
        "--no-playlist",
        "--write-info-json",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*[height<=720]+ba/b[height<=720]/best[height<=720]/best",
        "-o",
        str(target_dir / "%(id)s.%(ext)s"),
        row["url"],
    ]
    if cookies_from_browser:
        command[1:1] = ["--cookies-from-browser", cookies_from_browser]

    print(f"\n==> {title}")
    print(row["url"])
    if dry_run:
        print("DRY RUN:", " ".join(command))
        return

    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small batch of public video candidates.")
    parser.add_argument("--priority", default="high", choices=["high", "medium", "low"])
    parser.add_argument("--platform", choices=["YouTube", "Bilibili", "NBA Jr", "University of Richmond"])
    parser.add_argument("--limit", default=3, type=int)
    parser.add_argument(
        "--cookies-from-browser",
        choices=["edge", "chrome", "firefox"],
        help="Use browser cookies for platforms that require login or anti-bot verification.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ytdlp = find_ytdlp()
    rows = load_rows(args.priority, args.limit, args.platform)
    if not rows:
        raise SystemExit("No matching not_downloaded rows found.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        try:
            download(row, ytdlp, args.dry_run, args.cookies_from_browser)
        except subprocess.CalledProcessError as exc:
            print(f"FAILED: {row['title']} ({exc})")


if __name__ == "__main__":
    main()

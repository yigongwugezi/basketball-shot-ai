from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Basketball Shot AI Reference V1 experimental build")
    parser.add_argument("--input", type=Path, required=True, help="Trimmed single-shot video")
    parser.add_argument("--output", type=Path, required=True, help="Artifact output directory")
    parser.add_argument("--shot-type", choices=["jump_shot", "set_shot", "free_throw"])
    parser.add_argument("--pose-view", choices=["analysis", "raw"], default="analysis", help="Skeleton shown in annotated.mp4")
    args = parser.parse_args()
    try:
        report = run_pipeline(args.input, args.output, shot_type=args.shot_type, pose_view=args.pose_view)
    except Exception as exc:
        print(f"Reference V1 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(
        json.dumps(
            {
                "status": report["attempt"]["analysis_status"],
                "attempt_id": report["attempt"]["attempt_id"],
                "strict_release_frame": report["events"]["strict_ball_release"]["frame"],
                "available_metrics": sum(item["status"] == "ok" for item in report["metrics"].values()),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

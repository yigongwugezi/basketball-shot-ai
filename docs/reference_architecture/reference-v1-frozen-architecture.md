# Reference V1 frozen architecture

Status: `REFERENCE_V1 = DONE` · frozen 2026-09-03.

Reference V1 accepts a trimmed, normal-speed, single-shot basketball video with one unambiguous shooter and useful 2D view. Its default production chain is decode quality → shooter continuity → RTMPose-m → conservative temporal signals → event/phase decoding → ball and Human-Ball evidence → factual metrics → Motion Representation V1.

The canonical machine output is `report.json` with `shot_motion_representation_v1` (`motion_representation_version = 1`). Events are sparse points; phases are intervals; Pose Release and Strict Ball Release are distinct. Every measurement and relation carries explicit status, reliability, units, view dependence, and provenance. Unsupported evidence propagates to dependent facts instead of producing defaults.

The human output is `report.html`; traceable artifacts are `annotated.mp4`, `timeline.csv`, and `evidence/`, including `motion_representation_v1.json`. The system describes motion only. Coaching, quality judgement, 3D claims, and universal thresholds are outside V1.

Accepted limitations: partial ball tracking, contract-limited shooter identity, 2D/view dependence, and honest abstention under insufficient ball, landing, timing, or geometry evidence.

Next stage: **Basketball Intelligence**, starting with Basketball Metric Evidence Mapping.

# Reference V1 final acceptance — 2026-09-03

## Decision

`REFERENCE_V1 = DONE`. Reference V1 is frozen as an evidence-first, single-camera 2D description pipeline for one trimmed basketball shot. It reports what happened and why a fact is supported or unavailable; it does not judge technique or provide coaching.

## Supported input contract

A trimmed, normal-speed, single-shot video with one unambiguous shooter, useful side/diagonal geometry, sufficient body visibility and image quality, and visible ball evidence for ball-dependent facts. Severe person ambiguity or occlusion, poor lighting/blur, severe crop or distance, slow motion for real-time timing, insufficient ball evidence, and unsupported camera geometry produce reduced confidence, `unsupported_view`, or explicit abstention.

## Frozen architecture and output contract

The default CLI is `python -m reference_v1.cli --input VIDEO --output OUTPUT`. The production path is input/decode quality → YOLO person proposals and shooter continuity → RTMPose-m raw localization → conservative analysis pose signals → phase/event decoding → release_ball_v1 and Human-Ball contact/separation evidence → factual metrics → Motion Representation V1 → annotated video, HTML, JSON, CSV, and evidence artifacts.

Required output artifacts are `annotated.mp4`, `report.html`, `report.json`, `timeline.csv`, and `evidence/`. Motion V1 is embedded in `report.json` and written to `evidence/motion_representation_v1.json`. The timeline contains canonical motion events, temporal relations, Human-Ball state, and compact frame rows.

## Motion Representation V1

Schema identifier: `shot_motion_representation_v1`; `motion_representation_version = 1`. Sections are input quality, capture context, shooter identity, pose reliability, ball reliability, events, phases, motion primitives, Human-Ball relations, kinematics, temporal relations, uncertainty, and provenance.

Canonical events are `dip_start`, `dip_bottom`, `leg_drive_onset`, `ball_rise_start`, `elbow_extension_onset`, `takeoff`, `release_region_start`, `release_pose`, `strict_ball_release`, `body_apex`, `release_region_end`, and `landing`. Events are points. `release_pose` remains distinct from evidence-backed `strict_ball_release`.

Canonical phases are `setup`, `dip`, `drive`, `release`, `follow_through`, and `landing_recovery`. Phases are intervals and are never used as event substitutes.

Temporal relations cover the supported event pairs from dip bottom through drive, ball rise, elbow extension, takeoff, pose/strict release, body apex, landing, and follow-through end. A relation is emitted with null deltas and `insufficient_data` unless both endpoints exist. Supported relations carry frame, seconds, and normalized deltas. Slow-motion input preserves ordering but downgrades real-time timing.

Kinematics adapt the existing factual metric set: strict release frame, pose-to-strict delta, release elbow angle, normalized release height, dip depth, minimum knee angle, release relative to body apex, elbow-extension onset relative to release, takeoff-to-strict-release, and follow-through duration. Every metric carries value, unit, reference, status, qualitative reliability, confidence where present, view dependence, provenance, and reason. Geometric values remain explicitly 2D/view-dependent.

Reliability is separated into input, pose, ball, event evidence, measurement, and view suitability. `HIGH`, `MEDIUM`, `LOW`, and `INSUFFICIENT` are evidence categories, not calibrated probabilities. Missing strict release invalidates strict-release timing and separation facts; missing landing invalidates landing relations; insufficient ball track invalidates separation/flight facts; unsupported view invalidates affected geometric metrics.

## Acceptance set and results

Manifest: `reference_v1/acceptance_manifest.v1.json`. All cases used the default CLI and generated the complete artifact contract.

| Sample | Classification | Analysis | Strict release | Ball evidence | Runtime |
|---|---|---|---:|---|---:|
| IMG_7215 | SUPPORTED_GOOD_EVIDENCE | ok | f125 | ok | 247.00 s |
| IMG_7216 | SUPPORTED_GOOD_EVIDENCE | ok | f138 | ok | 112.41 s |
| BILI_010_A | EXPECTED_ABSTENTION | needs_review | abstained | insufficient_data | 24.07 s |
| BILI_002_A | EXPECTED_ABSTENTION | needs_review | abstained | insufficient_data | 23.03 s |
| BILI_010_B | EXPECTED_ABSTENTION | needs_review | abstained | insufficient_data | 16.73 s |

Total measured pipeline runtime was 423.23 s on CPU. The final review page is `E:/BasketballShotAI/analysis_runs/reference_v1_final_acceptance/review/index.html`; visual QA confirmed readable annotated frames, event/metric summaries, and conspicuous insufficient-evidence cards.

## Blocker disposition and limitations

The V1 blocker was the disconnected V0 representation: it lacked the final canonical vocabulary, explicit interval/point separation, supported endpoint relations, metric reliability/view semantics, and production artifact integration. These are closed.

Remaining V1 limitations are accepted: ball tracking remains partial; no arbitrary crowded-game identity guarantee; no calibrated 3D or metric-world biomechanics; unknown/unsupported camera geometry limits 2D measurements; timing from slow-motion/re-timed video is not real-time; landing and some primitives can remain unavailable when the clip ends or evidence is weak. These are local failures, not global fabricated outputs.

## V2 backlog

No V2 work was implemented. The next project stage is Basketball Intelligence, beginning with Basketball Metric Evidence Mapping. Potential later perception improvements remain targeted ball continuity, crowded-scene identity, calibrated/multiview geometry, and broader domain validation only when product evidence justifies them.

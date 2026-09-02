# Pose Ground Truth Benchmark — 2026-09-02

## Decision gate

`POSE_GT_PACKAGE_READY = YES`

`POSE_GT_CLOSURE = WAITING_FOR_HUMAN_LABELS`

The package closes the tooling and sampling gap, but it deliberately does not call any model prediction ground truth. A final pose-backbone decision is not permitted until a human has reviewed every selected frame.

## Benchmark design

The versioned manifest is `benchmarks/reference_v1/pose_gt_manifest.v1.json`. It fixes 42 high-value frames rather than taking a random or uniform sample:

| Source | Frames | Role | Timing restriction |
|---|---:|---|---|
| IMG_7215 | 16 | evaluation | normal-speed user clip |
| IMG_7216 | 16 | evaluation | normal-speed user clip |
| BILI_005_A | 10 | stress-test | slow-motion, contaminated/research-only; no generalization or real-time coordination claim |

The strata cover easy/static pose, dip/load, bottom, upward drive, rapid elbow extension, takeoff, release windows, follow-through, motion blur, ball-hand occlusion, partial occlusion, difficult configurations, and known identity ambiguity. The 42-frame overview was visually checked after extraction; the known late-clip identity failures in IMG_7215 and IMG_7216 remain in the package because they are decision-relevant failures, not noise to hide.

The core joint set is intentionally limited to shoulders, elbows, wrists, hips, knees, and ankles on both sides. Coordinates are original-frame pixels with a top-left origin. Each joint has exactly one visibility value:

- `visible`
- `occluded_but_inferable`
- `not_labelable`

`not_labelable` never requires coordinates and is excluded from accuracy denominators. The schema stores annotator, timestamps, revision, frame notes, frame metadata, and human review state.

## Data separation and leakage control

The generated package keeps three layers separate:

- `manifest.json`: immutable frame metadata and sampling roles.
- `model_predictions/*.json`: RAW YOLO11, FILTERED YOLO11, RTMPose, and RTMW predictions.
- `annotations/pose_gt.json`: human annotations only.

The UI may copy a selected model prediction into a draft, but the user must explicitly do so and explicitly mark the frame human-reviewed. The prediction overlay can be hidden. The evaluator rejects incomplete or unreviewed frames and never mutates, fills, or overwrites GT. No model parameter is tuned on these 42 frames. If filter redesign later becomes justified, a train/dev/test split must be introduced before tuning.

RTMPose and RTMW use the selected YOLO11 person bounding box. They are therefore pose-head/crop comparisons, not independent person-detector comparisons. All stored coordinates are mapped back to the original frame.

## Annotation tool

`benchmarks/reference_v1/pose_gt_annotation.html` is a dependency-free local page. The generated page supports:

- named left/right joints with distinct colors;
- mouse placement and dragging;
- previous/next frame shortcuts;
- per-joint visibility;
- model selection plus show/hide prediction;
- copy prediction to draft, reset one joint to the selected model, and clear one joint;
- zoom and scroll;
- browser-local autosave;
- JSON import and direct save/download;
- frame id, timestamp, tags, review progress, notes, annotator, and revision.

The starter file contains no GT coordinates and all 42 frames are `reviewed: false`.

## Accuracy evaluator

After human review, `benchmarks/reference_v1/pose_gt_benchmark.py evaluate` reports:

- pixel error and body-scale-normalized error;
- mean, median, P90, and P95 error;
- deterministic non-parametric 95% bootstrap intervals for median error;
- PCK@0.05, PCK@0.10, and PCK@0.20 body scale;
- failure/not-detected rate;
- easy, fast-motion, release-window, motion-blur, occlusion, shooting-arm, and lower-body groups;
- shooting wrist, elbow, and shoulder filter damage;
- `FILTER_HELP_RATE`, `FILTER_HARM_RATE`, and `FILTER_NEUTRAL_RATE`, using a declared ±2 px neutral band;
- ranked RAW-better and FILTERED-better examples plus contact sheets.

Temporal metrics remain in a separate `temporal_quality` section. They are proxy-only and never relabeled as accuracy.
Filter rates include detection transitions: RAW-detected/FILTERED-missing is harm for localization accuracy, FILTERED-detected/RAW-missing is help, and both missing is neutral while failure rates remain separately visible.

The prediction package currently reproduces these proxy-only P95 normalized second differences for IMG_7215 / IMG_7216 / BILI_005_A: RAW YOLO `0.396 / 0.325 / 0.055`, FILTERED YOLO `0.179 / 0.179 / 0.021`, RTMPose crop `0.124 / 0.082 / 0.048`, and RTMW crop `0.345 / 0.111 / 0.046`. Mean cached CPU times are about 68.5 ms for YOLO detector+pose, 12.5 ms for the RTMPose pose head, and 35.5 ms for the RTMW pose head. The latter two exclude YOLO bbox inference, so they are not end-to-end runtime comparisons.

## Human-review artifact

Generated outside Git:

`E:\BasketballShotAI\analysis_runs\pose_accuracy_closure\pose_gt_v1\annotate.html`

The same directory contains 42 frame images, four independent prediction files, a candidate-frame contact sheet, a four-model visual comparison sheet, a starter annotation JSON, and exact local instructions. Generated frames, images, predictions, and later results are intentionally excluded from the repository.

## Exact next action

Open `annotate.html` in Edge or Chrome. Enter the annotator name; for each frame press `B` if a model seed is useful, correct the 12 joints, set visibility, then check **Human-reviewed frame**. Save the result over `annotations/pose_gt.json`. Expected review time is approximately 20–30 minutes because most joints are pre-positioned by one click.

Then run from the repository root:

```powershell
.\.venv310\Scripts\python.exe benchmarks\reference_v1\pose_gt_benchmark.py evaluate --output "E:\BasketballShotAI\analysis_runs\pose_accuracy_closure\pose_gt_v1"
```

Until that succeeds, the only defensible backbone status is `POSE_BACKBONE_DECISION_FINAL = INSUFFICIENT_EVIDENCE`.

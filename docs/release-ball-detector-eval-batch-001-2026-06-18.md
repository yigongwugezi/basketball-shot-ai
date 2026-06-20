# Release Ball Detector Eval Batch 001 - 2026-06-18

## Background

This evaluation checks how the current detector performs on the first formal release-neighborhood annotation batch:
`datasets/annotations/release_ball_batch_001/labels.csv`.

The batch contains 3 clips:
`BILI_001_A`, `BILI_003_A`, and `BILI_005_A`.
It covers 93 frames in total, including 71 frames with `ball_visible=yes` target-shooter ball boxes and 22 frames with `ball_visible=no`.

The detector evaluation reuses the production detection logic from `backend/main.py` via `detect_frame()`.
No new detector was written, and no tracked source files were modified.

## Evaluation Method

For every row in `labels.csv`:

1. Read the original clip path from `datasets/bilibili_clip_plan.csv`.
2. Load the exact `frame_index` from the source video.
3. Run the current detector.
4. Compare predicted ball boxes with the manual ground-truth box.
5. Compute IoU and mark:
   - `matched` when IoU >= 0.3
   - `strong_match` when IoU >= 0.5
6. For `ball_visible=no` frames, record any detector output as a possible false positive, but do not count it as a true positive.

## Overall Metrics

| Metric | Value |
|---|---:|
| total_gt_visible_frames | 71 |
| matched_iou_0_3 | 0 |
| matched_iou_0_5 | 0 |
| recall_at_0_3 | 0.000 |
| recall_at_0_5 | 0.000 |
| miss_count | 71 |
| no_target_frames | 22 |
| no_target_frames_with_detection | 4 |

## Per Clip Metrics

| clip_id | gt_visible_frames | matched_iou_0_3 | matched_iou_0_5 | recall_at_0_3 | recall_at_0_5 | miss_count | ball_release_frame detected? | ball_release_frame matched@0.3? |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BILI_001_A | 16 | 0 | 0 | 0.000 | 0.000 | 16 | no | no |
| BILI_003_A | 29 | 0 | 0 | 0.000 | 0.000 | 29 | no | no |
| BILI_005_A | 26 | 0 | 0 | 0.000 | 0.000 | 26 | no | no |

## Release Window Results

The three strict release frames were all missed:

- `BILI_001_A` frame 55: no ball detection
- `BILI_003_A` frame 517: no ball detection
- `BILI_005_A` frame 224: no ball detection

The `+-3` frame windows around each release frame were also all misses.

## False Positives

Among the 22 `ball_visible=no` frames, 4 frames still produced a ball detection:

- `BILI_001_A` frame 56
- `BILI_001_A` frame 57
- `BILI_001_A` frame 67
- `BILI_001_A` frame 70

These are better interpreted as possible background-ball or wrong-object detections, not target-shooter-ball successes.

## Conclusion

The current detector has `0` recall on the 71 target-shooter ball frames in `release_ball_batch_001`.
The dominant failure mode is miss / low-recall, not a small localization offset.
This detector does not support strict `ball_release_frame` in its current form.

It is not worth wiring the detector into strict release timing yet.
We should not spend more effort tuning `wrist_y`, `elbow_angle`, or ROI / tracking parameters at this stage.
The blocker is the detector's inability to see the small ball near release.

## Next Steps

The best next move is to continue annotating the remaining 7 planned clips, or to prepare a detector fine-tuning set.
That gives us a larger small-ball sample before any attempt to change the detector weights.

## Notes

- No tracked source files were changed for this evaluation.
- No images, videos, or model weights are included in this report.
- The full temporary analysis output lives in `tmp/release_ball_detector_eval_batch_001_report.md`.

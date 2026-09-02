# Reference V1 Pose Reliability Pass

Date: 2026-09-02

## 1. Problem

Reference V1 previously sent the largest per-frame YOLO11 pose directly to analysis and rendering. The dominant visible failure on the two user clips was not only sub-pixel jitter: after the shooter moved out of the largest-person position, the skeleton jumped to another person and continued there. Local wrist, elbow, knee, and ankle noise also produced unstable second differences and angles.

The pass introduces an explicit, immutable `raw_pose -> analysis_pose` boundary. `keypoint_confidence` remains the model score; `temporal_reliability` and `correction_status` describe post-processing usability and are not measurement-accuracy probabilities.

## 2. Test clips and windows

| Clip | Window | Role |
|---|---:|---|
| `IMG_7215.MOV` | 75-145 | normal-speed drive/release/follow-through; prior pose release 122, strict release 125 |
| `IMG_7216.MOV` | 72-160 | normal-speed drive/release/follow-through; prior pose release 135, strict release 138 |
| `BILI_005_A_BV1Re4y1K7Ey.mp4` | 150-260 | slow-motion, crowd, occlusion, and upper-limb stress only |

The BILI window was not used for playback-timing or timing-accuracy claims.

## 3. Raw YOLO failure analysis

- IMG_7215 changes from the shooter to a left-side bystander at frame 133 and never returns inside the window.
- IMG_7216 changes person at frames 150-151 and remains on a bystander.
- These identity discontinuities created 49 and 51 large-jump outliers respectively, despite nominal 100% pose coverage.
- Raw P95 normalized second difference was 0.396 on IMG_7215 and 0.325 on IMG_7216.
- BILI did not show the same global identity failure. Its failure mode was local upper-limb ambiguity around ball/hand occlusion.

This explains why coverage alone looked healthy while the annotated skeleton visibly appeared to chase the action or attach to another person.

## 4. Filtering candidates

The implemented path is intentionally small:

```text
raw pose
  -> confidence gate
  -> isolated left/right and bone sanity checks
  -> temporal spike and identity-discontinuity rejection
  -> interpolation of internal gaps no longer than 3 frames
  -> adaptive symmetric three-frame coordinate smoothing
  -> analysis_pose + per-joint/frame status
```

Long gaps and permanent identity loss become `unavailable`; they are not extrapolated. A 5-frame symmetric aggregation remains only in the existing wrist/elbow release-event signal because removing it changed the event peak materially. It is zero-phase and does not affect the displayed skeleton coordinates. Strong moving averages and causal One-Euro filtering were not selected. No new runtime dependency was added.

## 5. Quantitative comparison

Primary reliability metrics are body-scale normalized. RTMPose/RTMW `_bbox` rows use the exact YOLO shooter bbox and the same accepted frames, so they compare pose heads rather than independent person detectors. Their runtime is pose-head-only and is not an end-to-end runtime.

| Window / candidate | usable coverage | P95 displacement | P95 second diff | median angle second diff (deg) | local angle-derivative noise (deg/frame) | large jumps | bone-length CV | runtime ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IMG_7215 raw YOLO | 1.000 | 0.550 | 0.396 | 8.146 | 3.568 | 49 | 0.256 | 71.8 |
| IMG_7215 filtered YOLO | 0.817 | 0.411 | 0.179 | 8.563 | 4.314 | 5 | 0.181 | 71.8 |
| IMG_7215 RTMPose bbox | 0.817 | 0.342 | 0.124 | 3.450 | 1.150 | 20 | 0.143 | 12.7* |
| IMG_7215 RTMW bbox | 0.817 | 0.452 | 0.345 | 4.408 | 1.325 | 33 | 0.168 | 35.3* |
| IMG_7216 raw YOLO | 1.000 | 0.456 | 0.325 | 7.206 | 3.881 | 51 | 0.261 | 69.9 |
| IMG_7216 filtered YOLO | 0.876 | 0.366 | 0.179 | 5.575 | 2.385 | 4 | 0.258 | 69.9 |
| IMG_7216 RTMPose bbox | 0.876 | 0.316 | 0.082 | 4.639 | 1.858 | 0 | 0.282 | 12.2* |
| IMG_7216 RTMW bbox | 0.876 | 0.335 | 0.111 | 4.938 | 1.707 | 13 | 0.275 | 35.6* |
| BILI raw YOLO | 1.000 | 0.065 | 0.055 | 0.361 | 0.098 | 0 | 0.108 | 65.4 |
| BILI filtered YOLO | 1.000 | 0.034 | 0.021 | 0.510 | 0.093 | 0 | 0.107 | 65.4 |
| BILI RTMPose bbox | 1.000 | 0.048 | 0.048 | 0.891 | 0.021 | 4 | 0.060 | 12.5* |
| BILI RTMW bbox | 1.000 | 0.052 | 0.046 | 0.674 | 0.005 | 0 | 0.069 | 35.6* |

For the normal-speed clips, filtered YOLO reduced large jumps by 90% and 92%, and P95 second-difference noise by 55% and 45%. IMG_7215 aggregate angle jitter remains mixed; the pass does not claim every joint metric improved. BILI P95 coordinate and second-difference noise improved, while its already-small median angle second difference increased by 0.15 degrees.

Full data: `E:\BasketballShotAI\analysis_runs\pose_reliability_pass\pose_reliability_benchmark.json`.

## 6. Visual review artifacts

Synchronized RAW | ANALYSIS videos:

- `E:\BasketballShotAI\analysis_runs\pose_reliability_pass\review\img_7215_release_drive_raw_vs_analysis.mp4`
- `E:\BasketballShotAI\analysis_runs\pose_reliability_pass\review\img_7216_release_drive_raw_vs_analysis.mp4`
- `E:\BasketballShotAI\analysis_runs\pose_reliability_pass\review\bili_005_a_difficult_release_raw_vs_analysis.mp4`

Human review found no visible skeleton phase shift around release. The major single-person failure is removed: when the shooter identity is lost, ANALYSIS becomes unavailable instead of drawing a confident skeleton on a bystander. The BILI arm motion remains fast and is not flattened into a slow trajectory.

Accepted full-pipeline outputs:

- `E:\BasketballShotAI\analysis_runs\pose_reliability_pass\accepted_img_7215\annotated.mp4`
- `E:\BasketballShotAI\analysis_runs\pose_reliability_pass\accepted_img_7216\annotated.mp4`

## 7. No-lag evaluation

Cross-correlation lag was measured on wrist X/Y, elbow angle, knee angle, and body vertical motion for every window.

| Window | median event-sensitive displacement | max event-sensitive displacement |
|---|---:|---:|
| IMG_7215 | 0 frames | 0 frames |
| IMG_7216 | 0 frames | 0 frames |
| BILI_005_A | 0 frames | 0 frames |

The filter is non-causal and symmetric. No tested signal showed a systematic delayed peak.

## 8. RTMPose / RTMW comparison

Full-frame RTMLib runs frequently selected a different person and therefore were not used as a backbone-quality result. The follow-up bbox-controlled comparison fixed the person input.

RTMPose was competitive on the two normal-speed windows and beat filtered YOLO on several angle/noise metrics, but it was mixed on BILI, had lower model scores, and still depended on a YOLO-derived person bbox. RTMW body joints did not show a consistent advantage over RTMPose; 133 points by itself was not treated as quality evidence. Without independent joint GT, these differences are insufficient to justify a production backbone change.

## 9. Event regression comparison

This is a regression comparison, not an accuracy result.

| Clip / field | prior raw pipeline | analysis_pose pipeline | delta / note |
|---|---:|---:|---|
| IMG_7215 dip | 75 | 75 | 0 |
| IMG_7215 bottom | 105 | 103 | -2 |
| IMG_7215 takeoff | unavailable | 119 | newly supported; no GT |
| IMG_7215 pose release | 122 | 125 | +3; stable arm peak now coincides with strict release, not claimed more accurate |
| IMG_7215 strict release | 125 | 125 | 0 |
| IMG_7215 apex | unavailable | 120 | newly supported; no GT |
| IMG_7215 landing | unavailable | unavailable | unchanged availability |
| IMG_7215 release elbow | 153.0° | 159.3° | +6.3° |
| IMG_7215 knee metric | 105.8° | 119.0° | +13.2° |
| IMG_7216 dip | 72 | 72 | 0 |
| IMG_7216 bottom | 91 | 92 | +1 |
| IMG_7216 takeoff | 112 | 111 | -1 |
| IMG_7216 pose release | 135 | 135 | 0 |
| IMG_7216 strict release | 138 | 138 | 0 |
| IMG_7216 apex | 135 | 135 | 0 |
| IMG_7216 landing | 150 | unavailable | frame 150 is an identity switch, so the prior landing was rejected |
| IMG_7216 release elbow | 122.7° | 126.0° | +3.3° |
| IMG_7216 knee metric | 127.2° | 123.0° | -4.2° |
| IMG_7216 strict release vs apex | +3 frames | +3 frames | 0 |

Strict release remains unchanged on both normal-speed clips. IMG_7215's pose-release shift is explained but not labeled as an accuracy improvement because there is no independent pose-event GT.

## 10. Failure cases

- Permanent shooter identity loss becomes unavailable rather than being repaired.
- Gaps longer than 3 frames are not interpolated.
- Some aggregate angle metrics, especially IMG_7215 knees, remain noisy.
- BILI is slow-motion and training-domain contaminated; its timing is not generalizable.
- RTMPose/RTMW comparisons use YOLO bbox control and do not evaluate independent person detection.
- There is no independent 2D joint GT, so lower jitter cannot prove lower localization error.
- 2D angles remain camera-view dependent.

## 11. Final pose backbone decision

`POSE_BACKBONE_DECISION = KEEP_YOLO_FILTERED`

Filtered YOLO is the smallest integration that removes the observed catastrophic failures, materially reduces high-percentile noise, preserves release timing, and adds no model/runtime dependency. RTMPose remains a future challenger only if independent joint GT or repeated human review shows the mixed metric advantage changes downstream event accuracy.

## 12. Recommended Reference V1 integration

- Keep `raw_pose` in every frame and write it to `evidence/pose_trajectories.json`.
- Use `analysis_pose` for pose-derived signals, events, angles, and metrics.
- Render `analysis_pose` by default.
- Use `--pose-view raw` for synchronized development/debug inspection.
- Preserve per-joint `keypoint_confidence`, `temporal_reliability`, `joint_status`, and frame `correction_status`.
- Treat unavailable intervals as missing evidence in downstream derivative/event code.

## 13. Remaining limitations

This pass is not a 3D, biomechanics, coaching, AQA, motion-understanding, or learned temporal model. It does not prove keypoint accuracy and does not repair long occlusions. The next evidence-changing step would be a small manually labeled 2D joint/event set, not another model sweep.

`POSE_RELIABILITY_READY = YES`

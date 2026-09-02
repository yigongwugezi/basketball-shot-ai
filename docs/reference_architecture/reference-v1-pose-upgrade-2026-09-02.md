# Reference V1 pose perception upgrade

Date: 2026-09-02
Decision: RTMPose mainline, legacy YOLO debug path retained

## Architecture

Before:

`YOLO11n-pose → largest person each frame → raw_pose → globally smoothed analysis_pose → events/metrics`

After:

`YOLO11n person candidates → deterministic shooter continuity → RTMPose-m Body7 256x192 → raw_pose localization evidence`

and, separately:

`raw_pose → confidence/anatomical checks + bounded interpolation (no global coordinate smoothing) → analysis_pose temporal signal → events/metrics`

The ball detector remains `release_ball_v1`. Ball and wrist coordinates are both original-video frame pixels. `pose_release` and `strict_ball_release` remain distinct.

The default command is unchanged:

```powershell
python -m reference_v1.cli --input VIDEO --output OUTPUT
```

It now defaults to `--pose-backbone rtmpose --pose-view raw`. The exact old path remains available with `--pose-backbone yolo`.

## Model and provenance

- RTMPose-m Body7 256x192, config identifier `rtmpose-m_simcc-body7_256x192-e48f03d0`
- RTMLib 0.0.16
- ONNX Runtime 1.23.2 CPU
- Existing E-drive cache; no model was copied to C:
- Every pose records provider, model/config, head runtime, person-box source, coordinate space, raw/derived status, shooter track id, selection confidence, identity break, and crop status

## Shooter continuity

The selector scores bbox IoU, center displacement, scale continuity, torso-pose center continuity, and visible-joint support. It keeps the current shooter when another person becomes larger. It refuses unsupported jumps for up to three frames; after a longer gap it explicitly reacquires with a new track id and `identity_break=true` instead of silently claiming continuity.

This remains deliberately scoped to trimmed, single-shot, unambiguous-shooter input. It is not a crowded-game tracker.

## Real-video regression

Normal-speed IMG_7215 and IMG_7216 were run through old and new paths. BILI_010_A was also run through the switch-free default command as a smoke test. Generated media and tables live at:

`E:\BasketballShotAI\analysis_runs\reference_v1_pose_upgrade\`

PUBLIC_GT established the localization decision. The following are real-video proxies, not accuracy measurements.

| Sample | Raw pose coverage | Large-jump outliers | Angle derivative noise (deg/frame) | Strict release old→new |
|---|---:|---:|---:|---:|
| IMG_7215 | 100.00% → 98.08% | 50 → 25 | 2.820 → 1.092 | 125 → 125 |
| IMG_7216 | 100.00% → 98.34% | 54 → 5 | 2.264 → 1.065 | 138 → 137 |

The small coverage reduction is intentional abstention during three-frame unsupported continuity gaps. Both clips later reacquire another visible person only after the original shooter leaves the supported track, and record the identity break instead of hiding it.

Most event deltas were within two frames. IMG_7215 `dip_start` moved +11 frames and `pose_release` moved -11 frames. Visual review shows that basketball arm/ball occlusion can still alter the pose-derived release candidate and therefore release-frame geometry. Strict release stayed at frame 125; the difference is retained as a risk flag, not smoothed away. A visually false IMG_7215 landing candidate at strict release was rejected by enforcing that landing must occur after strict release.

Pose-derived angle values changed materially, especially release elbow and minimum knee angle. Neither old nor new real-video values are ground truth; review them as localization/event-anchor changes. Exact event and metric tables are in `comparison/report.md`, `events.csv`, and `metrics.csv` under the E-drive run root.

## Runtime

Measured CPU RTMPose head cost was 23.31 ms/frame on IMG_7215 and 23.42 ms/frame on IMG_7216. Whole-pipeline totals were 73.366 s and 82.143 s respectively. End-to-end old/new totals varied substantially with the shared YOLO person and ball passes, so the saved per-component accounting should be used instead of claiming a stable speed ratio from two runs.

## Basketball-domain limits

- A top-down 2D body model can still confuse elbows/wrists when hands, ball, face, or the opposite arm overlap.
- RTMPose localization does not solve shooter identity after the shooter exits or a long unsupported crop gap.
- Camera pan and perspective can still contaminate ankle-based takeoff/landing and image-space heights.
- The current ball detector is independent but detector-only; strict release remains a conservative experimental geometry/contact-transition result and may abstain.
- Real-video comparisons have no HUMAN_GT and therefore do not establish localization accuracy.

## Decision

The pose upgrade passes the Reference V1 integration gate: RTMPose powers normal runs, raw localization is the default rendering, the old YOLO path remains available, strict release regressed by at most one frame on the two paired shots, and Motion Representation V0 serialized successfully for both new runs.

The single highest-value next perception step is a targeted human-ball/release closure: annotate and evaluate hand-ball contact/separation around release on normal-speed basketball clips, using the new RTMPose wrists and the existing independent ball evidence. Do not start another broad pose-framework survey.

```text
REFERENCE_V1_POSE_UPGRADE = PASS
RTMPOSE_MAINLINE = YES
SHOOTER_CONTINUITY = IMPROVED
RAW_POSE_DEFAULT = YES
GLOBAL_FILTER_DEFAULT = NO
STRICT_RELEASE_REGRESSION = PASS
MOTION_REPRESENTATION_COMPATIBILITY = PASS
REFERENCE_V1_REGRESSION = PASS
POSE_READY_FOR_MOTION_UNDERSTANDING = PARTIAL
NEXT_V1_PRIORITY = targeted human-ball / strict-release perception closure
LARGE_DATA_ON_C_DRIVE = NO
```

# Human-Ball / Release Perception V1 closure

> This closure record predates the final freeze. Its Motion Representation V0 references are historical; production now uses canonical `shot_motion_representation_v1` in `report.json` and `evidence/motion_representation_v1.json`.

Date: 2026-09-03

Starting mainline: `eb87cc0`

Decision: `HUMAN_BALL_PERCEPTION_V1 = PASS`

## What changed

The previous path selected the highest-confidence ball detection independently on every frame, measured RTMPose wrist-to-ball distance, found a global pose-only release maximum, and decoded strict release with a distance threshold around that pose candidate. It did not retain alternative ball candidates, explicitly distinguish missing/interpolated/ambiguous track points, or expose a per-frame contact state.

V1 now keeps all `release_ball_v1` candidates in a release-centered window and selects a short track using detector confidence, predicted spatial continuity, and pre-release shooting-wrist proximity. Jumps beyond seven ball diameters are rejected. Bounded gaps of at most two frames may be linearly interpolated; longer gaps stay missing. Every point is `DETECTED`, `INTERPOLATED`, `MISSING`, or `AMBIGUOUS`, with provenance.

Each supported frame records shoulder, elbow, wrist and ball geometry; distance in pixels, ball diameters and radii; ball, wrist and relative velocity; 2D elbow angle; relative position; and reliability. The wrist remains an explicit contact proxy, not a physical hand-contact measurement.

The temporal state machine is:

`UNKNOWN → LIKELY_CONTACT → SEPARATING → NO_CONTACT`

`SEPARATING` requires recent supported proximity, increasing separation, and distinct relative motion. Strict release is the earliest separating transition whose following three-frame interval supports persistent separation without a major return. Missing evidence never becomes `NO_CONTACT`; unsupported cases abstain.

Pose Release and Strict Ball Release remain separate events. Pose Release keeps the wrist-height, elbow-extension and wrist-velocity score, but when a supported contact/separation region exists it chooses the best pose score only inside that region. Without usable ball evidence it retains the pose-only candidate and records `pose_only_fallback`.

## Normal-speed validation

The old reports are the frozen `eb87cc0` artifacts. Frame changes are visual/proxy evidence, not ground-truth accuracy.

| Clip | Old pose | New pose | Old strict | New strict | Track coverage | Contact-state coverage | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| IMG_7215 | 114 | 125 | 125 | 125 | 89.74% | 76.92% | supported |
| IMG_7216 | 135 | 137 | 137 | 138 | 89.74% | 76.92% | supported |
| BILI_010_A | 55 | 55 | abstain | abstain | 2.56% | 0.00% | honest abstention |
| BILI_002_A | 32 | 32 | abstain | abstain | 12.82% | 0.00% | honest abstention |
| BILI_010_B | 28 | 28 | abstain | abstain | 15.38% | 12.82% | honest abstention |

IMG_7216 contains one defensible interpolated point. BILI_002_A rejected 23 discontinuous candidates. No long unsupported gap was bridged.

## IMG_7215 diagnosis

The old pose-only score peaked at frame 114 (`0.9583`) because the global wrist-height / elbow-extension / wrist-velocity combination favored the early arm-extension peak. The ball was still spatially coupled to the wrist through frame 124. Human-Ball V1 constrains the release region to frames 124–125 and selects frame 125 (`0.8664`) as the action release. At frame 125 the distance reaches 1.64 ball diameters with distinct relative motion and is labeled `SEPARATING`; frames 126–127 confirm persistent separation and frame 126 is visibly `NO_CONTACT`. The strict event therefore remains frame 125, validated by its supporting interval, rather than being copied from Pose Release.

The visual sequence supports the correction from the implausibly early frame 114. This is still visual/proxy validation, not annotated event ground truth.

## Downstream semantics and regression

- Pose/action measurements (`release_elbow_angle`, `normalized_release_height`, elbow-extension timing and follow-through duration) use Pose Release.
- Ball-separation measurements (`strict_release_frame`, pose-to-strict delta, release relative to body apex and takeoff-to-strict timing) require Strict Ball Release and abstain when it is absent.
- Phase fallback to Pose Release remains explicit through `pose_release_fallback` risk flags.
- Landing is rejected unless it occurs after Strict Release when both exist.
- Motion Representation V0 validates for all five clips. Its separation relation uses Strict Release and abstains on the three BILI clips; its generic action release may use Pose Release only with explicit provenance.

The main measurement changes are expected consequences of the corrected event semantics: IMG_7215 release elbow angle changes 178.7° → 138.5° and normalized release height 0.750 → 0.729; IMG_7216 changes 170.8° → 164.4° and 0.823 → 0.759. IMG_7216 strict-dependent timing moves by one frame. IMG_7215 continues to abstain on landing and follow-through duration because the apparent ankle return is not persistent. No metric silently substitutes a missing strict separation event.

## Runtime and artifacts

Human-Ball post-processing is small: matched analysis-stage overhead had a median delta of about 0.054 seconds across the five clips. End-to-end CPU wall time was higher in this run (median inference 174.5 seconds versus 63.5 seconds in the frozen baseline), but the increase occurred inside unchanged model inference, not Human-Ball analysis; the runs were not a controlled performance benchmark.

Primary artifacts:

- `E:\BasketballShotAI\analysis_runs\reference_v1_human_ball\review\index.html`
- `E:\BasketballShotAI\analysis_runs\reference_v1_human_ball\comparison\old_vs_new.json`
- `E:\BasketballShotAI\analysis_runs\reference_v1_human_ball\comparison\validation.json`
- Per-shot `report.json`, `timeline.csv`, `annotated.mp4`, `evidence/human_ball_release_v1.json`, and `evidence/motion_representation_v0.json`

All generated videos, frames, review assets and reports remain under `E:\BasketballShotAI`. Only source, tests, a review generator and this lightweight document are stored in the repository.

## Limitations

- Five unlabelled normal-speed clips are a closure set, not a ground-truth accuracy benchmark.
- The detector has useful release-window coverage on the two phone clips but insufficient coverage on all three BILI clips; this is the principal remaining perception failure.
- RTMPose wrist is not the fingers or physical contact point. Occlusion and unusual one-handed ball placement can bias normalized distance.
- The tracker is deliberately local and offline: future frames validate persistence. It is not a full-game or real-time tracker.
- Ball-rise onset is a conservative image-plane vertical trend and remains camera-motion/view sensitive.
- All geometry is single-camera 2D; depth, scale change, projection and motion blur can distort distance and velocity.
- Pose Release and Strict Release can share a frame while retaining different definitions, evidence and provenance.

No additional perception research survey was needed: the implementation blocker was resolved with existing detector candidates, RTMPose evidence and explicit temporal persistence. The highest-value next V1 step is **Motion Representation V1 plus final Reference V1 end-to-end acceptance**, not another pose or perception model sweep.

# Basketball Shot AI Reference Architecture V1

Date: 2026-09-01

## Architecture principle

V1 is an **evidence architecture**, not an end-to-end score generator. Each layer emits observations, confidence, provenance and failure reasons. Feedback may use only evidence whose upstream gates passed.

```text
Video
 -> ingest + quality/cut gate
 -> shot proposal and shooter selection
 -> body/whole-body + ball/rim observations
 -> person association + ball point continuity
 -> hand-ball interaction states
 -> strict-release event + ordered phases
 -> normalized 2D kinematics + uncertainty
 -> matched-attempt/exemplar comparison
 -> rule outcomes
 -> feedback-ready evidence JSON
```

## Layer 1: Ingest and quality gate

| Role | Choice |
|---|---|
| Baseline | Existing metadata/decode checks: FPS, frame count, dimensions, pose/ball visibility. |
| Selected candidate | PySceneDetect for cuts; explicit local blur/person scale/ball scale checks. |
| Fallback | Reject edited/variable/too-small clips with `insufficient_data`. |
| Future challenger | TransNetV2 for cuts; DOVER-Mobile as auxiliary technical-quality signal. |

Output: canonical frame timestamps, quality flags, usable ranges, camera-view hint, no silent frame-index assumptions.

## Layer 2: Shot proposal and shooter selection

| Role | Choice |
|---|---|
| Baseline | Current project shot interval/pose heuristics and person selection. |
| Selected candidate | Ball-near-person persistence plus coarse pose motion; retain user-confirmed range workflow for data preparation. |
| Fallback | Ask for a trimmed shot or mark ambiguous multiple-person clips unavailable. |
| Future challenger | ActionFormer-like long-video proposal model after source-level proposal labels exist. |

Output: one or more attempt intervals, shooter track ID and ambiguity score. Multiple attempts from one source retain the same `source_video_id`.

## Layer 3: Human perception

| Role | Choice |
|---|---|
| Baseline | YOLO11 pose already used by backend. |
| Selected candidate | RTMW whole-body benchmark; RTMPose as speed/control challenger. |
| Fallback | YOLO11 body pose; omit hand/foot metrics that fail visibility. |
| Future challenger | ViTPose++ accuracy ceiling; RTMW3D research-only. |

Output: body/foot/hand keypoints and per-joint confidence in the shooter coordinate system. Whole-body must win on temporal stability and usable release-window coverage, not only static AP.

## Layer 4: Object perception

| Role | Choice |
|---|---|
| Baseline | Existing YOLO11n V1 basketball detector, explicitly prototype evidence. |
| Selected candidate | Same detector with SAHI comparison; train/benchmark RF-DETR core only after trusted V2 split. |
| Fallback | Pose release evidence with detector status `no_detection`/`insufficient_data`; never fabricate ball position. |
| Future challenger | RF-DETR Nano/Small/Core; TrackNetV4-like multi-frame heatmap if box detection remains the bottleneck. |

Output: ball/rim/backboard observations, visibility, bbox/center confidence, source model and slicing mode.

## Layer 5: Temporal continuity

| Role | Choice |
|---|---|
| Baseline | Per-frame detector centers and current fusion diagnostic. |
| Selected candidate | Detector-initialized CoTracker3 over a short release window with confidence/visibility and detector re-anchors. |
| Fallback | Detector-only trajectory with explicit gaps and maximum interpolation limit. |
| Future challenger | TAPNext++; SAM2.1 mask propagation when pixel scale permits. |

Output: ball center trajectory, visibility state, anchor provenance, drift/reinitialization flags. Point tracking is accepted only if it improves independent release-window continuity without unacceptable drift.

## Layer 6: Human-object interaction and strict release

| Role | Choice |
|---|---|
| Baseline | Current pose release plus top-level `release_fusion` diagnostic with V1 detector evidence. |
| Selected candidate | Explicit state machine over hand/ball distance, overlap/mask when available, ball velocity separation, pose timing and persistence. |
| Fallback | Pose release retained as an estimate; strict release unavailable when contact transition is not observed. |
| Future challenger | Basketball-fine-tuned temporal HOI classifier plus T-DEED/AdaSpot precise event head. |

Selected states:

```text
unknown -> possession_candidate -> contact_confirmed
        -> separation_candidate -> released_confirmed
        -> flight / out_of_view / reacquired
```

`strict_ball_release_frame` is the first persistent supported no-contact frame after verified contact. `release_pose_frame` remains a different anatomical proxy. `release_fusion` continues to expose agreement/delta/risk instead of hiding disagreement.

## Layer 7: Ordered phases

| Role | Choice |
|---|---|
| Baseline | Existing five-phase heuristics/experimental phase work, not yet stable enough for demo claims. |
| Selected candidate | MS-TCN++ over cached pose/ball/contact/appearance features after ground truth is frozen. |
| Fallback | Emit only supported sparse events; do not force all five phases. |
| Future challenger | ASFormer; AdaSpot-style local high-resolution crops for exact events. |

Outputs: setup, dip, drive/jump, release, follow-through and optional landing with frame intervals, confidence and order violations. Strict release uses a separate exact-event loss/metric from phase segmentation.

## Layer 8: Motion reconstruction and kinematics

| Role | Choice |
|---|---|
| Baseline | Current normalized 2D pose metrics. |
| Selected candidate | Sports2D-style confidence-aware filtering, bounded interpolation and view-conditioned normalized features. |
| Fallback | Raw event timing and qualitative evidence only. |
| Future challenger | Controlled Pose2Sim/OpenCap multi-view validation; OpenCap Monocular/MotionBERT research benchmark. |

V1 exposes planar joint angles, phase timing, normalized positions and repeatability. Metric 3D speed/force/torque/injury/fatigue are unavailable unless a future validated capture mode supports them.

## Layer 9: Skill and action quality

| Role | Choice |
|---|---|
| Baseline | Existing heuristic metric interpretation. |
| Selected candidate | Matched-condition within-user comparison + pairwise expert exemplar + explicit rule graph. |
| Fallback | Evidence-only report without score. |
| Future challenger | HP-MCoRe-style phase-aware visual/pose AQA, FineCausal tests and NS-AQA-style neuro-symbolic report. |

No global learned AQA score is selected for V1. Comparison requires matching shot type, view, distance bucket and visibility. Subject/source-disjoint evaluation is mandatory.

## Layer 10: Feedback-ready evidence

```json
{
  "observation": "release elbow extension lower than matched accepted attempts",
  "phase": "strict_release",
  "frames": [214, 215, 216],
  "measurement_type": "normalized_2d_estimate",
  "confidence": 0.72,
  "uncertainty": "release +/-2 frames; hand partially occluded",
  "comparison": "same user, free throw, side view, n=5",
  "rule_id": "release_elbow_consistency_v1",
  "allowed_feedback": "test a more repeatable release timing under the same setup",
  "forbidden_claims": ["caused the miss", "injury risk"]
}
```

V1 uses constrained templates. An LLM may later rephrase validated fields but cannot invent observations, causes or measurements.

## Evaluation contract

- **Splits:** `source_video_id` isolated; independent test frozen; new-person/new-view/hard-visibility slices.
- **Ball:** release-window recall, center error, visibility F1, track continuity and drift.
- **Release:** precision/recall and absolute frame error at 0/1/2/3/5 frames.
- **Phases:** F1@IoU, edit, frame accuracy and order violations.
- **Pose:** usable-joint recall, PCK where labeled, temporal jitter and left/right swaps.
- **Kinematics:** repeatability plus reference error per metric/view.
- **Feedback:** coach evidence correctness, specificity, usefulness, abstention correctness and forbidden-claim rate.

## ONE RECOMMENDED STACK

1. **Ingest:** OpenCV/ffprobe metadata + PySceneDetect + explicit quality gates.
2. **Shooter:** existing person selection, with ByteTrack only when identity continuity is needed.
3. **Pose:** benchmark RTMW against current YOLO11 pose; select RTMW only if whole-body stability wins.
4. **Ball:** current YOLO11n V1 detector as baseline, SAHI as immediate inference challenger; RF-DETR core waits for V2 split.
5. **Ball continuity:** CoTracker3 initialized/re-anchored by detector observations.
6. **Hand evidence:** RTMW hands first; MediaPipe Hands only on viable release crops.
7. **Strict release:** explicit temporal contact/separation state machine, with PTS exact-frame evaluation.
8. **Phases:** MS-TCN++ on cached multimodal evidence once labels are frozen.
9. **Kinematics:** Sports2D-style 2D filtering/normalization with per-metric uncertainty.
10. **Quality/feedback:** matched-attempt comparison + expert-authored rule graph + constrained evidence templates.

This is one stack, with fallbacks. It intentionally does not select monocular 3D, end-to-end AQA or a generative coach for V1.

# Required final conclusions

## 1. Five most limiting previous assumptions

1. Treating release as a pose frame rather than a hand-object contact transition.
2. Treating per-frame ball detection as sufficient temporal evidence.
3. Treating five phases as a basketball-only heuristic problem rather than precise event plus temporal segmentation.
4. Treating a single monocular pose output as metric biomechanics without view/task validation.
5. Treating more auto-labeled frames or a whole-video quality score as a substitute for source-isolated trusted evidence.

## 2. Problems solved better outside basketball

- Tennis/badminton/table tennis: tiny fast-object heatmaps, blur and occlusion-aware tracking.
- Golf: ordered frame-accurate swing events.
- Soccer: long-video dense event spotting and low-label transfer.
- Surgery/industrial procedures: stable phase segmentation and mistake-aware sequences.
- Diving/fitness/rehabilitation: phase-aware, pose-guided, uncertainty-aware and rule-based AQA.
- HOI/robotics: contact state and transition semantics.
- Rehabilitation/biomechanics: validation protocols and scientific limits of markerless motion.

## 3. Modules that can stand on general technology

Video cuts/quality, person tracking, whole-body pose, open/closed-set detection frameworks, point tracking, temporal segmentation frameworks, 2D signal processing, feature matching and feedback evidence schemas.

## 4. Problems that still need basketball data

Tiny-ball detection in the project's views, hand-ball contact/release, shot attempt boundaries, five phase labels, matched basketball quality judgments, expert feedback rules and independent product evaluation.

## 5. Ten resources most likely to change the architecture

CoTracker3; TrackNetV4; 100DOH/Hands23 contact formulation; PTS; T-DEED; AdaSpot; RTMW; Sports2D; FineDiving/HP-MCoRe; NS-AQA/FLEX rule graphs.

## 6. Current components most likely to be replaced

- Pose-only strict release by contact-transition fusion.
- Per-frame detector-center trajectory by detector + CoTracker3.
- Current unstable phase heuristics by MS-TCN++/ASFormer after labels.
- Universal metric suggestions by matched-condition evidence rules.
- YOLO11 pose may be replaced by RTMW if the controlled benchmark wins; the V1 ball detector may be replaced by RF-DETR/TrackNet-like models only after trusted test evidence.

## 7. Best strict-release route

Detector-confirmed ball anchors + point continuity + whole-body/hand evidence + contact/separation state machine + exact-frame temporal decoder. Keep pose release and strict ball release separate; fusion reports disagreement and abstains when contact is invisible.

## 8. Best phase route

Create source-video-disjoint dense/sparse ground truth, cache appearance/pose/ball/contact features, establish MS-TCN++ baseline, then test ASFormer. Use a separate exact-event head for strict release.

## 9. Best action-quality route

Start with matched-attempt comparative evidence and phase-local rules. Only after stable phases and expert labels, benchmark HP-MCoRe-style pose/visual procedure AQA with uncertainty and causal checks.

## 10. Best coaching-feedback route

Expert-authored metric-to-error-to-feedback graph, evidence frames and constrained wording. Retrieve a matched exemplar. LLM verbalization is optional and downstream; generated expert motion is research-only.

## 11. Scientific boundary of ordinary monocular phone video

It can support event timing, visibility-qualified 2D normalized geometry, phase timing, image-plane trajectories and within-person consistency under matched views. It cannot safely claim metric 3D ball speed/height, forces, torques, joint loading, injury risk, fatigue or causation without calibration/reference validation.

## 12. Final Reference V1 components

Quality/cut gate; shot proposal; shooter association; whole-body pose; closed-set ball detector; point continuity; hand-ball interaction state machine; precise release event; dense phase model; confidence-aware 2D kinematics; matched-comparison rule graph; feedback-ready evidence API with abstention.

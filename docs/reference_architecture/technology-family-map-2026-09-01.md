# Technology Family Map

Date: 2026-09-01

Legend: `->` means a methodological successor or a clearly stronger generation; `+` means complementary rather than replacement; `x` means not selected for the current path.

## Detection

```text
YOLO11n project baseline
  + SAHI sliced inference (no retraining, immediate test)
  -> RF-DETR Nano/Small/Core (immediate closed-set challenger)

open vocabulary:
Grounding DINO -> Grounded SAM 2 (Grounding DINO + SAM2 video masks)
```

- **Keep:** YOLO11n as the controlled baseline; RF-DETR core as challenger.
- **Do not conflate:** SAM2 tracks prompted masks but does not replace semantic detection.
- **Research-only:** Grounding DINO/Grounded SAM2 for candidate prelabels and diagnostics.
- **License boundary:** RF-DETR core tiers and larger/restricted tiers are not interchangeable; every external weight needs its own review.

## Multi-object/person tracking

```text
SORT-style motion association
  -> ByteTrack (uses low-score detections)
  -> BoT-SORT (adds ReID + camera-motion compensation)
```

- **Current use:** ByteTrack-level association is sufficient for shooter identity in fixed-camera, low-crowd clips.
- **Challenger:** BoT-SORT only for crowded/moving-camera cases.
- **Not selected for ball:** both families assume box detections and motion regimes unlike one tiny fast ball.

## Point tracking

```text
TAP-Vid benchmark
  -> TAPIR (global match + local refinement)
  -> BootsTAPIR (real-video pseudo-label improvement)
  -> TAPNext (next-token propagation)
  -> TAPNext++ (long-memory, occlusion/re-detection)

CoTracker -> CoTracker2 -> CoTracker3 (joint tracks, visibility, online/offline)
```

- **Immediate:** CoTracker3 because it has a practical public interface and visibility output.
- **Research challenger:** TAPNext++.
- **Do not benchmark all generations:** TAPIR/BootsTAPIR/TAPNext are lineage references unless the latest asset is unavailable.
- **Basketball adaptation:** initialize from a detector-confirmed ball center; measure drift and visibility on release windows before any fine-tuning.

## Fast small-object sports tracking

```text
TrackNet (stacked frames -> heatmap)
  -> TrackNetV2
  -> TrackNetV3 (background + inpainting/rectification)
  -> TrackNetV4 (motion-attention)

TTNet: multi-task ball + table + hit/bounce events
BlurBall: blur length/orientation + multi-frame localization
```

- **Selected idea:** multi-frame heatmap and explicit visibility/blur, not per-frame box only.
- **Immediate action:** no training. First test detector + CoTracker3.
- **Escalation:** train a TrackNetV4-like basketball heatmap model only if point tracking fails and trusted labels justify it.
- **Not transferable directly:** shuttle/tennis/table geometry, checkpoints and event classes.

## Segmentation

```text
SAM (image prompts)
  -> SAM2 (video memory)
  -> SAM2.1 (checkpoint/training refinement)

Grounding DINO + SAM2 -> Grounded SAM2
```

- **Candidate use:** visible ball-mask propagation from one trusted prompt.
- **Fallback:** point tracking is cheaper when the ball is only a few pixels.
- **Reject as assumption:** a good box prompt does not guarantee stable mask identity through hand occlusion.

## Body pose

```text
17-joint real-time body pose (current YOLO11 pose)
  -> RTMPose (strong real-time body challenger)
  -> RTMW (whole-body, multi-scale body/feet/hands)
  -> RTMW3D (camera-relative whole-body 3D research branch)

ViTPose -> ViTPose++ (larger accuracy/generalization branch)
DWPose (distilled whole-body branch)
```

- **Immediate:** RTMPose and RTMW, with the same shooter crops and temporal-stability metrics.
- **Likely winner for V1 evidence:** RTMW if hand/foot landmarks remain usable at actual resolution; otherwise RTMPose plus a release-hand crop.
- **Research only:** RTMW3D/ViTPose++ until ordinary-phone validation justifies cost.

## Hand

```text
MediaPipe Hands (21 2D/approximate world landmarks, light)
  x no direct successor relation
HaMeR (MANO 3D mesh, heavy and richer)
```

- These solve different operating points, not generations of one family.
- **Immediate:** crop around RTMW/person wrist and test MediaPipe Hands.
- **Research challenger:** HaMeR only on high-resolution crops.
- **Failure rule:** tiny/occluded hand returns unavailable; never fabricate finger evidence.

## Human-object interaction

```text
100DOH: hand + contact state + contacted object
  -> Hands23: richer hand/object relation annotations

ContactHands: contact taxonomy
EgoHOS / VISOR: hand-active-object masks
ContactPose / HOT3D: calibrated 3D contact references
```

- **Selected abstraction:** `unknown -> possession -> contact -> separating -> released`.
- **Immediate implementation path:** explicit features from ball track, hand landmarks/masks and temporal persistence; no general HOI weight is assumed transferable.
- **Basketball-specific requirement:** contact/release labels and fine-tuning are unavoidable.

## Precise event spotting

```text
GolfDB/SwingNet (ordered event baseline, 2019)
PTS/E2E-Spot (precise event formulation, 2022)
  -> T-DEED (multi-scale temporal discrimination, 2024)
  -> AdaSpot (global low-res + task-aware high-res ROI, 2026)
```

- **Evaluation now:** PTS exact-frame tolerance protocol.
- **Model after labels:** T-DEED.
- **Research challenger:** AdaSpot.
- **GolfDB status:** valuable simple baseline but non-commercial license and older architecture; do not make it the final stack.

## Dense phase segmentation

```text
MS-TCN -> MS-TCN++ (multi-stage dilated TCN refinement)
  -> ASFormer (local-attention encoder-decoder)
  -> LTContext / ASQuery / EAST (newer long-context/end-to-end branches)

surgical online branch:
TeCNO -> Trans-SVNet / TUNeS-like spatial-temporal hybrids
```

- **Immediate baseline:** MS-TCN++ on cached evidence features.
- **Immediate challenger:** ASFormer only after the same split/labels are fixed.
- **Future:** EAST-style end-to-end fine-tuning needs much more trusted data.
- **Do not mix tasks:** strict release is a sparse precise event; five phases are dense segments. They can share features but require different losses/metrics.

## 3D body and markerless biomechanics

```text
2D validated tooling: Sports2D

monocular 3D representation:
MotionBERT (2D-to-3D temporal lifting)
4DHumans/HMR2.0 (mesh + tracking)
  -> WHAM (world-grounded moving-camera motion)
  -> GVHMR (gravity-view global motion)

multi-view validated workflow:
Pose2Sim + OpenSim
OpenCap (two+ smartphones)
  -> OpenCap Monocular (2026 research branch, not yet sports-shot validated)
```

- **V1:** 2D normalized, view-aware metrics with uncertainty.
- **Validation reference:** Pose2Sim/OpenCap multi-view.
- **Research challenger:** OpenCap Monocular, then GVHMR if global motion matters.
- **Forbidden shortcut:** model output labeled “3D” does not make metric angles/forces scientifically valid.

## AQA

```text
global score lineage:
MTL-AQA -> USDL/MUSDL (uncertainty) -> CoRe (comparative regression)

procedure-aware lineage:
FineDiving -> FineParser / HP-MCoRe (pose + visual + procedure)

interpretability lineage:
Fitness-AQA -> NS-AQA (neuro-symbolic rules/report)
FineCausal (causal interventions)
UIL-AQA (clip uncertainty)
FLEX (multimodal keystep-error-feedback graph)
```

- **V1 winner:** not a learned global scorer. Use comparative/rule-based evidence first.
- **First learned candidate:** HP-MCoRe-style phase-aware pose/visual model after labels.
- **Explanation guard:** NS-AQA-style explicit rules plus FineCausal-style tests that the claimed evidence actually affects output.

## Feedback

```text
timestamped expert commentary: Ego-Exo4D
  -> ExpertAF (actionable commentary + corrected demonstration)

expert analysis benchmark: ExAct
joint localization/quality: SkillSpotter
pairwise ranking: PROSKILL
```

- **V1:** deterministic evidence JSON -> rule outcomes -> constrained language templates.
- **Next:** retrieve matched expert/user exemplar and pairwise comparison.
- **Future:** LLM verbalization and generated demonstrations only after evidence correctness is independently evaluated.

## Video quality and camera

```text
classic cut detection: PySceneDetect
  -> learned cut detection: TransNetV2

FAST-VQA -> FasterVQA -> DOVER / DOVER-Mobile

feature matching: SuperPoint/DISK/ALIKED + LightGlue -> background homography

court calibration:
KaliCalib / DeepSportRadar
  -> TVCalib -> PnLCalib -> BroadTrack temporal calibration
```

- **V1:** explicit metadata/scale/blur checks plus PySceneDetect; learned VQA is auxiliary.
- **Camera:** fixed-camera recommendation first; LightGlue motion flag second; court calibration only for sufficiently visible wide views.

## Components explicitly not selected now

- Grounding DINO/Grounded SAM2 as trusted label generators.
- LocateAnything labels in trusted training.
- RF-DETR Plus/XL/2XL without separate license/cost review.
- MotionBricks for ball detection/release.
- End-to-end coaching LLM from raw pixels.
- Monocular force, torque, kinetic-chain or injury claims.
- Full Ego-Exo4D/BASKET downloads.
- Training an AQA model before independent phase/release evidence is reliable.

# Top 10 Experiments

Date: 2026-09-01
Priority heuristic: `Impact x Information Gain / Implementation Cost`. No experiment below should train on or tune against the independent test.

## 1. Freeze a strict-release micro-benchmark

- **Question:** can reviewers reliably identify contact, separation and strict release at frame level?
- **Method:** select a small source-video-disjoint set across clear/occluded/blurred views; two reviewers label contact state, first persistent no-contact frame and uncertainty.
- **Metrics:** inter-reviewer frame error, agreement by visibility, unresolved rate.
- **Decision unlocked:** whether strict release is learnable at current FPS/resolution and what tolerance to use.
- **Impact / information / cost:** 10 / 10 / 2.

## 2. Detector center versus CoTracker3 continuity

- **Question:** does point tracking bridge release-window detector gaps without harmful drift?
- **Method:** initialize CoTracker3 only from trusted detector anchors; compare 10-30-frame windows against manually marked ball centers/visibility.
- **Metrics:** center error, visibility F1, track survival, drift, re-anchor count and release evidence coverage.
- **Decision unlocked:** adopt point tracking or remain detector-only.
- **Impact / information / cost:** 10 / 9 / 3.

## 3. Native detector versus SAHI

- **Question:** is ball recall limited by inference resolution rather than training data/model family?
- **Method:** run identical YOLO11 weights at native resolution and a small SAHI grid; do not retrain.
- **Metrics:** release-window recall, center error, duplicate false positives, latency.
- **Decision unlocked:** cheap detector gain versus need for RF-DETR/new data.
- **Impact / information / cost:** 8 / 9 / 1.

## 4. YOLO11 pose versus RTMPose versus RTMW

- **Question:** which pose model produces the most usable temporal body/hand/foot evidence on actual clips?
- **Method:** fixed shooter crops and frame set; manually inspect/labelling a compact joint subset around release/landing.
- **Metrics:** usable-joint recall, temporal jitter, left/right swaps, occluded-joint behavior, latency and memory.
- **Decision unlocked:** V1 pose model and whether separate hand inference is justified.
- **Impact / information / cost:** 9 / 9 / 3.

## 5. Explicit contact-transition decoder

- **Question:** does a transparent state machine outperform pose-only release and current diagnostic fusion?
- **Method:** features from ball trajectory, hand distance/overlap, motion separation, pose release and persistence; threshold only on train/val, freeze test.
- **Metrics:** release precision/recall, error at 0/1/2/3/5 frames, abstention correctness and failure reasons.
- **Decision unlocked:** strict release V1 algorithm and data requirements for T-DEED.
- **Impact / information / cost:** 10 / 10 / 4.

## 6. Camera/view observability audit

- **Question:** which 2D metrics remain stable under side/diagonal/front views?
- **Method:** same or closely matched attempts from controlled views; calculate pose visibility, angle repeatability and camera-motion flags.
- **Metrics:** per-metric missing rate, within-view repeatability, cross-view bias and confidence calibration.
- **Decision unlocked:** view-specific metric allowlist and recording guidance.
- **Impact / information / cost:** 9 / 9 / 3.

## 7. Phase annotation reliability before model training

- **Question:** are setup/dip/drive-release/follow-through/landing definitions reviewable and ordered?
- **Method:** two reviewers label a small source-disjoint set using the current review tool; record uncertain/missing phases.
- **Metrics:** boundary frame agreement, segment IoU, order violations and annotation time.
- **Decision unlocked:** dense labels versus sparse timestamp supervision and any taxonomy revision.
- **Impact / information / cost:** 9 / 10 / 3.

## 8. MS-TCN++ phase baseline on cached evidence

- **Prerequisite:** Experiment 7 passes reliability threshold.
- **Question:** does learned temporal refinement beat current phase heuristics without end-to-end complexity?
- **Method:** cache identical appearance/pose/ball/contact features; source-video split; one controlled MS-TCN++ baseline.
- **Metrics:** F1@10/25/50, edit, frame accuracy, boundary error and order violations.
- **Decision unlocked:** whether ASFormer is worth testing.
- **Impact / information / cost:** 8 / 8 / 5.

## 9. Matched-attempt rule feedback pilot

- **Question:** can evidence-first rules produce useful feedback before learned AQA?
- **Method:** select 3-5 observable metrics, compare same-user/same-shot/same-view attempts, expose frames and uncertainty; qualified reviewer rates statements.
- **Metrics:** evidence correctness, specificity, usefulness, agreement, forbidden-claim rate and abstention quality.
- **Decision unlocked:** feedback schema and expert annotation needs.
- **Impact / information / cost:** 10 / 8 / 4.

## 10. Two-view validation of selected 2D metrics

- **Question:** what error do V1 planar metrics have against an accessible multiview reference?
- **Method:** small controlled capture with two synchronized phones; Pose2Sim/OpenCap-style calibration; compare only the metrics selected by Experiment 6.
- **Metrics:** bias, RMSE, ICC/Bland-Altman limits by metric/view/phase.
- **Decision unlocked:** allowed product wording and whether monocular 3D research is justified.
- **Impact / information / cost:** 9 / 10 / 7.

## Execution order

```text
1 -> 2,3,4 -> 5
6 -> 7 -> 8
5,6 -> 9
6 -> 10
```

Do not start RF-DETR training, T-DEED/AdaSpot, TrackNetV4-like training or HP-MCoRe before the corresponding data/evaluation experiment above establishes that the subsystem is the bottleneck.

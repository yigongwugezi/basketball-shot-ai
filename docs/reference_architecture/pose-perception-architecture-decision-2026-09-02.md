# Pose perception architecture decision — 2026-09-02

## Decision

Use a **HYBRID, evidence-driven perception architecture**. RTMPose-m is the best current public-GT body-pose backbone and the default reference challenger. It should sit behind an explicit shooter/person detection and identity stage. Do not make RTMW the default body backbone: its small wins on some LSP static and bent-lower-body subsets do not offset its weaker error tails and much higher CPU cost. Keep RTMW as a research-only whole-body/hand specialist until hand GT proves that the extra keypoints add product value.

Keep RAW YOLO evidence. The existing temporal filter remains useful for trajectory regularity, but it is not an accuracy oracle and must not replace raw pose. Its localization effect is mixed: small net gains are concentrated in JHMDB lower-body failures, while fast elbows include clear regressions.

The single highest-value next perception step is to establish a compact, independent **basketball release-window HUMAN_GT benchmark** for shooter crop plus shoulder/elbow/wrist. It should cover modern smartphone side/45-degree shooting, ball-hand occlusion and the release window. Until then, the public benchmark selects a strong challenger but does not validate the product domain.

## Evidence boundary

- Localization accuracy below is prediction versus accepted source `HUMAN_GT`.
- Temporal quality is trajectory regularity on complete JHMDB clips; it is not localization accuracy.
- Runtime is measured wall time on this CPU-only environment.
- Failure is explicit detection/crop/pose-head availability, not confidence-as-accuracy.
- JHMDB and LSP are research benchmark sources only. Their acceptance does not authorize commercial training.

The frozen evaluation set contains 66 samples and 762 labelable mapped core joints:

| Dataset/subset | Samples |
|---|---:|
| JHMDB / `ALL_JHMDB` | 36 |
| JHMDB / `BASKETBALL` | 9 |
| JHMDB / `THROWING_STRIKING` | 18 |
| JHMDB / `JUMPING` | 6 |
| JHMDB / `FAST_ARM` | 27 |
| JHMDB / `DIFFICULT_POSE` | 15 |
| LSP / `ALL_LSP` | 30 |
| LSP / `UPPER_BODY_DIFFICULT` | 20 |
| LSP / `LOWER_BODY_DIFFICULT` | 9 |
| LSP / `OCCLUDED` | 12 |
| LSP / `VISIBLE` | 18 |

JHMDB subsets use source action labels and recorded review tags. LSP difficulty uses only GT geometry: an upper-body sample has a wrist above its shoulder or an elbow angle below 100 degrees; a lower-body sample has a knee angle below 120 degrees. `OCCLUDED` and `VISIBLE` use the source LSP visibility flag. LSP has no defensible temporal ordering, so FILTERED is an explicit static pass-through and never contributes invented temporal evidence.

For temporal quality, all 346 frames from the 12 source JHMDB clips containing frozen samples were processed. Only the 36 frozen JHMDB frames enter localization accuracy.

## Models and reproducibility

| Pipeline | Exact pose weight | Evaluation form |
|---|---|---|
| RAW YOLO | `yolo11n-pose.pt`, SHA-256 `869e83f…9319dc0` | Ultralytics 8.4.67, integrated largest-person pose, 640 input |
| FILTERED YOLO | same raw YOLO evidence | confidence/anatomy checks, bounded interpolation, adaptive zero-phase smoothing on contiguous clips |
| RTMPose | `rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx`, SHA-256 `5c0a4bf…5683e3c` | RTMLib 0.0.16 / ONNX Runtime 1.23.2; first 17 COCO body joints; current YOLO crop |
| RTMW | `rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx`, SHA-256 `9a9bc17…43aa58` | RTMLib 0.0.16 / ONNX Runtime 1.23.2; first 17 of 133 whole-body points; current YOLO crop |

The manifest fingerprint is recorded in `summary.v1.json`. Image dimensions, top-left origin, x/y order, direct left/right name mapping, visibility, frame indices, GT bounds and RTM original-frame coordinates all passed automated sanity checks: 66/66 dimensions and 762/762 labelable joints were valid. Original source GT was not changed.

## HUMAN_GT localization

PCK denominators include every labelable joint, so a missing joint or failed crop remains an end-to-end failure. Error distribution statistics are over available predictions. Conditional PCK values in the JSON use available predictions only.

| Pipeline | Joint coverage | Median px | Mean px | P90 px | P95 px | Median normalized | PCK@0.10 | PCK@0.20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW YOLO | 95.80% | 5.089 | 9.810 | 20.707 | 39.668 | 0.1172 | 39.50% | 69.95% |
| FILTERED YOLO | 95.80% | 4.954 | 9.515 | 19.344 | 39.199 | 0.1146 | 40.16% | 71.39% |
| RTMPose | 96.72% | **2.681** | **4.303** | **7.311** | **11.086** | 0.0634 | **71.00%** | **88.71%** |
| RTMW | **96.85%** | 2.706 | 5.047 | 8.051 | 13.570 | **0.0631** | 69.29% | 88.32% |

RTMPose is the overall winner. Its median pixel error is 47% lower than RAW YOLO, and its P95 is 72% lower. RTMW has a negligible normalized-median edge and one additional valid joint, but RTMPose has materially better mean, P90, P95, PCK@0.10 and PCK@0.20.

### Joint-level results

Cells are median pixel error / P95 pixel error / end-to-end PCK@0.10.

| Pipeline | Wrist | Elbow | Shoulder | Hip | Knee | Ankle |
|---|---|---|---|---|---|---|
| RAW YOLO | 6.014 / 56.238 / 29.23% | 5.997 / 33.923 / 32.58% | 4.521 / 14.855 / 47.73% | 5.447 / 14.726 / 36.51% | 4.647 / 18.796 / 46.03% | 4.585 / 37.868 / 45.69% |
| FILTERED YOLO | 6.036 / 54.233 / 30.77% | 6.162 / 37.025 / 31.82% | 4.679 / 13.426 / 47.73% | 5.091 / 14.282 / 38.89% | 4.256 / 19.595 / 49.21% | 4.445 / 37.868 / 43.10% |
| RTMPose | **3.353 / 17.327 / 65.38%** | 2.913 / 14.698 / 63.64% | **2.382 / 11.583 / 74.24%** | 3.213 / **7.288** / 65.08% | 2.314 / **6.690 / 81.75%** | 2.590 / 7.768 / 76.72% |
| RTMW | 3.366 / 19.924 / 64.62% | **2.830 / 11.846 / 66.67%** | 2.731 / 13.024 / 70.45% | **3.183** / 7.775 / **65.87%** | **2.305** / 12.269 / 71.43% | **2.403 / 6.832 / 77.59%** |

For wrist+elbow together, RTMPose has the lower median pixel error (3.094 versus 3.212) and better P95 (16.682 versus 17.999); RTMW has slightly higher PCK@0.10 (65.65% versus 64.50%) and coverage (96.95% versus 96.56%). The product decision favors RTMPose because fast-arm tail failures matter more than that small threshold-count advantage.

### Dataset and difficult-subset results

Cells are median pixel error / P95 pixel error / end-to-end PCK@0.10.

| Group | RAW YOLO | FILTERED YOLO | RTMPose | RTMW |
|---|---|---|---|---|
| JHMDB | 5.556 / 30.415 / 35.32% | 5.219 / 29.263 / 36.57% | **2.953 / 9.971 / 66.92%** | 3.275 / 10.758 / 63.43% |
| LSP | 4.622 / 45.560 / 44.17% | same | 2.449 / **12.417** / 75.56% | **2.199** / 20.206 / **75.83%** |
| Basketball | 3.454 / 9.981 / 52.78% | 3.630 / 9.267 / 51.85% | **2.551 / 6.113 / 78.70%** | 2.731 / 6.169 / 70.37% |
| Fast arm | 5.993 / 36.807 / 36.39% | 5.781 / 36.834 / 36.05% | **3.338 / 11.222 / 67.69%** | 3.594 / 11.371 / 63.27% |
| JHMDB difficult pose | 5.766 / 27.047 / 30.29% | 5.202 / 19.110 / 33.71% | **2.460 / 7.071 / 64.57%** | 2.783 / 7.513 / 63.43% |
| LSP upper difficult | 4.995 / 39.702 / 39.17% | same | 2.453 / **11.706** / 75.83% | **2.303** / 13.649 / **76.25%** |
| LSP lower difficult | 5.016 / 30.567 / 42.59% | same | 2.316 / **11.553** / 76.85% | **1.931** / 11.614 / **79.63%** |
| LSP occluded samples | 4.847 / 32.651 / 43.06% | same | 2.484 / 10.658 / 78.47% | **2.127 / 10.420 / 80.56%** |

RTMW's specialist signal is real but narrow: it is best by median/PCK on static LSP upper/lower difficulty and occluded-sample subsets. RTMPose is stronger on temporal JHMDB, basketball, fast-arm and JHMDB difficult poses, and usually has safer tails.

## Detection, crop and pose-head closure

The current RTM chains use the exact YOLO-selected person bbox. `CROP_SUCCESS` requires at least 75% of labelable HUMAN_GT core joints inside that bbox. This avoids hiding wrong-person selection as a pose-head error.

| Pipeline | Person detection fail | Wrong/incomplete crop | Pose-head sample fail on correct crop | Final sample coverage |
|---|---:|---:|---:|---:|
| All four pipelines | 0/66 | 2/66 | 0/64 | 64/66 (96.97%) |

The shared failures are:

1. `catch_Ballfangen_catch_u_cm_np1_fr_goo_0_0007`: YOLO selected a large foreground arm/hand instead of the GT child; GT core-joint coverage and bbox IoU are both zero. This is an identity/person-selection failure and invalidates all downstream models.
2. `lsp_im0483`: YOLO selected the correct gymnast but only 8/12 GT core joints lie inside its box (66.67% coverage, IoU 0.3335). This is an incomplete-crop pipeline failure, not a wrong-person case.

On the 64 correct-crop samples (738 labelable joints), RAW/FILTERED YOLO return 730 valid joints (98.92%), RTMPose returns 737 (99.86%), and RTMW returns 738 (100%). RTM pose heads are therefore not the source of the two end-to-end sample failures. The crop/identity stage is the shared ceiling.

## RAW versus FILTERED damage

The paired neutral band is ±2 pixels. All 762 labelable joints remain in the denominator; LSP's 360 comparisons are declared static pass-through and therefore neutral.

| Scope | Help | Harm | Neutral | Median delta px |
|---|---:|---:|---:|---:|
| All public GT | 2.49% | 0.52% | 96.98% | 0.000 |
| JHMDB only | 4.73% | 1.00% | 94.28% | -0.122 |
| Fast arm | 3.40% | 0.68% | 95.92% | -0.152 |
| Wrist | 3.08% | 0.00% | 96.92% | 0.000 |
| Elbow | 0.76% | **1.52%** | 97.73% | 0.000 |
| Knee | 4.76% | 1.59% | 93.65% | 0.000 |
| Ankle | 5.17% | 0.00% | 94.83% | 0.000 |

Filtering slightly improves aggregate YOLO median/P90/P95 and substantially improves its temporal second-difference proxy, but priority-arm localization does not consistently improve. The worst observed harm moves the right elbow in a throw from 22.25 px raw error to 45.61 px filtered error (+23.36 px). The strongest gains repair isolated jump/strike ankle failures by 18–23 px; one jumping wrist improves by 13.06 px.

The accuracy effect is therefore **MIXED**. Preserve raw pose; use filtered coordinates for downstream motion only with per-joint provenance and reliability. A future refinement should be selective and uncertainty-triggered, not stronger global smoothing.

## Temporal quality, kept separate

Across the 12 complete JHMDB clips, median sequence-level normalized second difference is 0.0511 RAW, 0.0197 FILTERED, 0.0254 RTMPose and 0.0263 RTMW. Median bone-length CV is 0.1920, 0.1806, 0.1592 and 0.1479 respectively. RTMPose has the fewest median large-jump outliers (4 versus RTMW 6 and FILTERED 6.5), while RTMW has the lowest bone variation, angle derivative noise and displacement.

RTMW receives the narrow `BEST_TEMPORAL_QUALITY` label for trajectory regularity, but this is not proof that it localizes better. RTMPose remains the HUMAN_GT accuracy winner.

## Runtime

Environment: Windows, Python 3.10.11, OpenCV 5.0.0, PyTorch 2.12.0 CPU build, ONNX Runtime 1.23.2 CPU/Azure providers, RTX 4060 present but unavailable to the installed runtimes.

| Pipeline/stage | Mean ms | Median ms | P95 ms |
|---|---:|---:|---:|
| YOLO11n-pose integrated person+pose | 756.2 | 758.0 | 839.9 |
| FILTERED incremental filter | 1.9 | 1.8 | 4.6 |
| FILTERED end-to-end | 758.1 | 760.2 | 840.6 |
| RTMPose pose head | 43.3 | 35.6 | 76.7 |
| RTMPose end-to-end with current YOLO provider | 799.5 | 800.9 | 870.4 |
| RTMW pose head | 303.9 | 301.3 | 441.7 |
| RTMW end-to-end with current YOLO provider | 1060.1 | 1044.5 | 1254.7 |

`PERSON_DETECT_RUNTIME` is conservatively represented by the integrated YOLO11n-pose call because the current adapter does not expose a detector-only stage. It therefore overstates the cost of a future bbox-only detector. The old 12.5/35.5 ms pose-head observations did not include this provider and are not substituted into this table. On this environment RTMW's head is about 7.0× RTMPose's head, while RTMPose adds only about 5.7% to the current provider and halves median localization error.

## Targeted research interpretation

The research was chosen after measurement and addresses observed mechanisms only.

- RTMPose uses the SimCC family of coordinate classification. SimCC separates x and y into one-dimensional classification targets and was designed to reduce low-resolution heatmap quantization without an expensive 2D upsampling head. The paper reports particular gains at low input resolution. This is consistent with, but does not by itself prove, the measured RTM advantage on low-resolution JHMDB/LSP and small wrist/elbow targets. Sources: [RTMPose](https://arxiv.org/abs/2303.07399), [SimCC](https://arxiv.org/abs/2107.03332), [official MMPose RTMPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose).
- RTMW adds feature pyramids and a Hierarchical Encoding Module for body parts at different scales, trains on the `cocktail14` mixture, and uses two-stage distillation. This provides a plausible mechanism for its static bent-body/occlusion wins. Its whole-body capacity is not free: this exact 256×192 model is much slower, and its body-joint P90/P95 are worse than RTMPose here. Source: [RTMW paper](https://arxiv.org/abs/2407.08634).
- PoseWarper propagates and aggregates temporally warped pose evidence; DCPose uses previous/current/next frames and refines residual heatmaps; FAMI-Pose explicitly aligns support-frame features coarse-to-fine and supervises extraction of complementary information; TDMI separates task-relevant temporal differences from motion/background noise. Their shared lesson is aligned, selective neighboring-frame evidence—not coordinate averaging. Sources: [PoseWarper official repository](https://github.com/facebookresearch/PoseWarper), [DCPose paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Deep_Dual_Consecutive_Network_for_Human_Pose_Estimation_CVPR_2021_paper.pdf), [FAMI-Pose paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Temporal_Feature_Alignment_and_Mutual_Information_Maximization_for_Video-Based_Human_CVPR_2022_paper.pdf), [TDMI paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_Mutual_Information-Based_Temporal_Difference_Learning_for_Human_Pose_Estimation_in_CVPR_2023_paper.pdf).
- PoseFix shows that an image-conditioned, model-agnostic pose refiner can correct swaps/misses rather than merely smooth them. It is relevant to selective refinement, but should be deferred until basketball-specific error distributions and GT exist; otherwise a refiner can reproduce the same “correct point moved away” failure seen in the elbow damage case. Source: [PoseFix paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Moon_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network_CVPR_2019_paper.html).

No external temporal repository is adopted in this task. PoseWarper is archived; FAMI-Pose publishes no pretrained model; and all candidate code/weights/datasets require a separate license and provenance review before product use. The MMPose code repository is Apache-2.0, but the exact `body7` and `cocktail14` training mixtures must be audited dataset by dataset before any commercial deployment or derivative training.

## Architecture options

| Option | Decision | Evidence |
|---|---|---|
| A. Single best backbone | Reject as final architecture | RTMPose wins body localization, but shared crop failure and absent hand/ball GT remain. |
| B. Fast primary + specialist windows | Adopt | RTMPose is the primary; expensive refinement is justified only around uncertain/high-value release windows. |
| C. Primary + temporal video refinement | Adopt as next research challenger, not installed now | Filtering helps some failures but harms correct fast elbows; aligned feature/heatmap evidence is the appropriate mechanism. |
| D. Body + arm/hand specialist + ball | Adopt after domain GT | RTMW's extra points were not scored; a hand/arm branch needs hand/ball GT before it can earn product status. |
| E. Hybrid evidence-driven perception | **Selected** | It directly addresses the measured crop ceiling, body localization, selective refinement, temporal evidence and future ball evidence. |

Target architecture:

```text
RGB video
  -> shooter/person detection + identity continuity
  -> RTMPose-class fast body pose (raw evidence retained)
  -> uncertainty and event-window gate
       -> aligned temporal pose refinement when evidence is weak
       -> fine arm/hand specialist only when GT supports it
  -> independent ball detector/trajectory
  -> anatomy + provenance-aware evidence fusion
  -> trusted human-ball trajectory
```

Do not feed smoothed output back as ground truth. Every refined joint should retain source model, confidence/uncertainty, crop identity, raw coordinate, correction type and availability.

## Product-domain limitation

JHMDB/LSP establish public athletic/general pose localization evidence. They do not establish accuracy on modern smartphone basketball, ball-hand occlusion at release, strict release timing, preferred side/45-degree capture, high-resolution shooting or fine hand/ball geometry. The nine JHMDB `shoot_ball` frames are low-resolution historical action footage, not product validation. `BASKETBALL_PRODUCT_DOMAIN_VALIDATION` remains `NOT_ESTABLISHED`.

## Artifacts and gates

Standalone review: `E:\BasketballShotAI\analysis_runs\public_pose_benchmark\review\index.html`

Machine-readable outputs stay on E: and are not committed: `summary.v1.json`, `joint_records.v1.json`, `filter_damage.v1.json`, `inference.v1.json`, review media.

```text
PUBLIC_GT_BENCHMARK = COMPLETE
BEST_PUBLIC_GT_LOCALIZATION = RTMPOSE
BEST_FAST_BACKBONE = RTMPOSE
BEST_WRIST_ELBOW = RTMPOSE
BEST_LOWER_BODY = RTMPOSE
BEST_DIFFICULT_POSE = RTMPOSE_OVERALL_RTMW_ON_STATIC_LSP_SUBSETS
BEST_TEMPORAL_QUALITY = RTMW
BEST_END_TO_END_COVERAGE = RTMW
FILTERING_ACCURACY_EFFECT = MIXED
POSE_BACKBONE_DECISION = HYBRID
BASKETBALL_PRODUCT_DOMAIN_VALIDATION = NOT_ESTABLISHED
POSE_READY_FOR_MOTION_UNDERSTANDING = PARTIAL
```

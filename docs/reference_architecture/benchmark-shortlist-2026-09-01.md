# Benchmark Shortlist

Date: 2026-09-01
Rule: no module has more than four candidates; every benchmark must use the same source-video-disjoint test.

| Module | Current baseline | Immediate challenger | Research challenger | Recommendation |
|---|---|---|---|---|
| Video quality/cuts | metadata + current heuristics | PySceneDetect | TransNetV2; DOVER-Mobile auxiliary | Benchmark PySceneDetect first; learned VQA only if explicit gates miss failures. |
| Body pose | YOLO11 pose | RTMPose | RTMW | Benchmark RTMPose and RTMW on stability and usable release-window joints, not COCO AP alone. |
| Hand evidence | wrist/body keypoints | MediaPipe Hands on crop | HaMeR on high-res crop | Test whether real project pixels support hands before adding a heavy model. |
| Ball detection | YOLO11n V1 detector | YOLO11 + SAHI | RF-DETR Nano/Small/Core | Run SAHI first; RF-DETR only after trusted independent test is ready. |
| Ball continuity | detector centers | CoTracker3 | TAPNext++ | Initialize from trusted detector anchors; report drift and visibility through 10-30 release frames. |
| Ball mask | box/center only | SAM2.1 prompted mask | Grounded SAM2 research-only | Benchmark only on frames where the ball has enough pixels; do not assume masks beat points. |
| Contact/release | pose heuristic + diagnostic detector fusion | explicit hand-ball state machine | learned basketball HOI classifier | Build labels/evaluation before T-DEED; release is contact-to-separation, not one pose. |
| Precise event | current release heuristic | PTS evaluation + lightweight temporal classifier | T-DEED; AdaSpot later | Adopt exact-frame metrics now; train T-DEED only after trusted release labels. |
| Five phases | current heuristic phase logic | MS-TCN++ | ASFormer | Cache identical evidence features; compare source-video-disjoint F1/edit/order violations. |
| 2D kinematics | current normalized pose metrics | Sports2D-style filtering/validity | controlled two-view OpenCap/Pose2Sim validation | Keep V1 planar and uncertainty-aware. |
| Monocular 3D | none/product unavailable | MotionBERT research baseline | OpenCap Monocular; GVHMR | Not benchmark-now unless 2D metrics show a specific unresolved need. |
| Quality/feedback | heuristic suggestions | rule graph + matched exemplar | HP-MCoRe/NS-AQA architecture | No learned global score now; evidence first, constrained wording second. |

## Benchmark-now set

1. YOLO11 pose vs RTMPose vs RTMW.
2. Native ball detector vs SAHI-sliced inference.
3. Detector centers vs detector-initialized CoTracker3.
4. Current release heuristic vs contact-transition state machine under exact-frame tolerance.
5. Current phase heuristics vs MS-TCN++ only after phase ground truth is frozen.

## Explicitly deferred

- RF-DETR training, T-DEED, AdaSpot, TrackNetV4-like training, HaMeR, 3D body mesh and learned AQA.
- Reason: each depends on labels, reference measurements or a license/data gate that does not yet exist.

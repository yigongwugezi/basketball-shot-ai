# Reference Stack Selection 2026-09-01

This is the first winner selection after executable local benchmarks. `KEEP` means current default; `ADD` means a gated experimental component; it does not imply product readiness.

## Selection table

| Module | Action | Selected component | Rejected/deferred challenger | Why |
|---|---|---|---|---|
| Full-video person/pose | KEEP | YOLO11 pose + current largest-person baseline | ByteTrack DEFER; BoT-SORT default REJECT | trackers added no stability gain and switched on IMG_7216; shooter association remains unsolved |
| Release-window whole body | ADD | RTMW on short offline ROI | RTMPose DEFER | RTMW adds 133 whole-body points; too slow for full-video CPU use |
| Ball detector | KEEP + ADD | keep release-ball v1 prototype; add RF-DETR Nano as V2 challenger | COCO sports-ball and smoke multi-class model REJECT | RF-DETR recall won but FP/runtime lost; v1 remains current precision/runtime baseline |
| Ball anchors | KEEP | detector observations and explicit gaps | none | trackers cannot correct bad initialization reliably |
| Ball continuity | ADD | CoTracker3 with detector initialization, periodic re-anchor and drift/visibility gate | LK control only; TAPNext++ DEFER | CoTracker3 won strongly on clean IMG_7216 but failed catastrophically on BILI_002 |
| Ball mask fallback | DEFER | Grounding DINO + SAM 2.1 research tool only | product integration REJECT now | pipeline ran, but mask coverage was 33.3% and CPU latency 2.67 s/frame |
| Strict release | ADD | contact-transition geometry proxy with abstention | formal API replacement REJECT now | interface is useful; only one of three historical clips was near agreement |
| Five phases | KEEP experimental | phase-v2 evidence and risk flags, HEAD unchanged | MS-TCN++ DEFER until GT | 10/10 ordering but only 1/10 status ok; no complete phase GT |
| Product action quality | DEFER | evidence-only and constrained rules | learned AQA | upstream events and independent evaluation are not ready |

## Winner, loser and blocked

### Winners

1. **Current YOLO11 pose baseline** for full-video CPU operation.
2. **RTMW** for new whole-body/hand evidence in a short offline ROI.
3. **RF-DETR Nano** as the next detector challenger, because it had the highest measured recall.
4. **CoTracker3 conditionally**, only after a trusted anchor and with re-anchoring/abstention.
5. **Release-ball v1** remains the current prototype detector because its FP/runtime tradeoff is much better than zero-shot RF-DETR.

### Losers

- Generic YOLO11 COCO sports-ball: 3.7% recall.
- Existing ball/rim/player smoke model for release-ball use: 0% recall.
- BoT-SORT as the default person selector: slower with no stability gain.
- OpenCV LK as a reliable ball tracker: fast but severe drift.
- Grounded SAM 2 as a primary video tracker: insufficient mask persistence and CPU speed.

### Blocked

No attempted candidate was blocked after one reasonable setup pass. ViTPose, TAPNext++, learned HOI and trainable temporal models were deliberately not attempted; their prerequisites are missing and they remain `DEFER`, not `BLOCKED`.

## Most surprising results

1. RF-DETR Nano's un-fine-tuned COCO model slightly exceeded the custom release v1 recall, but at the cost of 90 false positives versus 7.
2. CoTracker3 changed from excellent (0.18-ball-diameter drift proxy) to catastrophic (17.16) between two short clips.
3. ByteTrack and BoT-SORT did not solve shooter selection; the hard problem is association evidence, not tracker branding.
4. Grounded SAM 2 ran successfully on CPU but propagated a nonempty mask through only one third of the 15-frame clip.
5. RTMW's 133 points are available locally, but whole-video CPU use would be impractical without ROI restriction or GPU runtime.

## Components most likely to be replaced

- **First:** release-ball v1, after RF-DETR/YOLO V2 is trained and judged on a frozen independent test.
- **Second:** pose-only strict release, after contact-transition GT and a detector+tracker+hand pipeline are validated.
- **Third:** heuristic five phases, after phase GT supports an MS-TCN++ baseline.
- **Not yet:** YOLO11 full-video pose and the formal API. This sprint does not justify replacing either.

## ONE RECOMMENDED REFERENCE STACK

```text
OpenCV ingest + existing quality gates
 -> existing shot range / shooter candidates
 -> YOLO11 pose for full video
 -> RTMW only on gated release ROI
 -> release-ball v1 anchors now
 -> RF-DETR Nano/Core as V2 detector challenger
 -> detector-reanchored CoTracker3 with drift/visibility abstention
 -> explicit hand-ball contact/separation state machine
 -> phase-v2 evidence now; MS-TCN++ after frozen GT
 -> confidence-aware normalized 2D metrics
 -> evidence-only constrained feedback
```

Grounded SAM 2 is outside the main stack and remains a small-volume research/annotation fallback.

## Immediate benchmark follow-ups

1. Freeze 20-40 strict-release clips with human frame labels and 0/1/2/3/5-frame metrics.
2. Add detector re-anchor and drift rejection to CoTracker3, then rerun the same two clips plus hard occlusion slices.
3. Fine-tune RF-DETR Nano or Core and YOLO11 on one identical trusted V2 train split; compare on one frozen test.
4. Run RTMW only around release and quantify usable hand-point coverage against manual visibility labels.
5. Add shooter-association GT before further ByteTrack/BoT-SORT tuning.
6. Freeze phase GT before training MS-TCN++.

Do not expand manual labeling to thousands of frames before these small tests identify the actual bottleneck.

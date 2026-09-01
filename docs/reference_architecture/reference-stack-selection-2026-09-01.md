# Reference Stack Selection 2026-09-01

This is the first winner selection after executable local benchmarks. `KEEP` means current default; `ADD` means a gated experimental component; it does not imply product readiness.

## Selection table

| Module | Action | Selected component | Rejected/deferred challenger | Why |
|---|---|---|---|---|
| Full-video person/pose | KEEP | YOLO11 pose + current largest-person baseline | ByteTrack DEFER; BoT-SORT default REJECT | trackers added no stability gain and switched on IMG_7216; shooter association remains unsolved |
| Release-window whole body | ADD | RTMW body/wrist on short offline ROI | RTMW fingers REMOVE; RTMPose DEFER | only 3/17 shooter-credible cached hand frames passed the automatic usability proxy; body/wrist remains useful |
| Ball detector | KEEP + ADD | keep release-ball v1 prototype; add RF-DETR Nano as V2 challenger | COCO sports-ball and smoke multi-class model REJECT | RF-DETR recall won but FP/runtime lost; v1 remains current precision/runtime baseline |
| Ball anchors | KEEP | detector observations and explicit gaps | none | trackers cannot correct bad initialization reliably |
| Small-ball slicing | REMOVE | native release-ball inference | SAHI DEFER for one future executable rerun | SAHI could not run in the closure environment and has no measured gain to justify latency/duplicates |
| Ball continuity | DEFER | detector observations remain default; retain gated CoTracker3 adapter for research | LK control only; TAPNext++ DEFER | re-anchor restored 100% coverage on 24 IMG_7216 GT frames but used 16 anchors and did not beat detector-only center error |
| Ball mask fallback | DEFER | Grounding DINO + SAM 2.1 research tool only | product integration REJECT now | pipeline ran, but mask coverage was 33.3% and CPU latency 2.67 s/frame |
| Strict release | ADD | contact-transition decoder v1 with pose prior, detector evidence, persistence and abstention | formal API replacement REJECT now | exact on the one overlapping human-GT cached clip and correctly abstained on two unsupported historical-reference clips; frozen evaluation remains pending |
| Five phases | KEEP experimental | phase-v2 evidence and risk flags, HEAD unchanged | MS-TCN++ DEFER until GT | 10/10 ordering but only 1/10 status ok; no complete phase GT |
| Product action quality | DEFER | evidence-only and constrained rules | learned AQA | upstream events and independent evaluation are not ready |

## Winner, loser and blocked

### Winners

1. **Current YOLO11 pose baseline** for full-video CPU operation.
2. **RTMW body/wrist only** for new offline release-ROI evidence; finger/palm landmarks are unavailable unless they pass size and stability gates.
3. **RF-DETR Nano** as the next detector challenger, because it had the highest measured recall.
4. **Contact-transition decoder v1**, with explicit abstention and pose/ball evidence kept separate.
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
- **Second:** pose-only strict release, after source-separated contact-transition GT is frozen and evaluated.
- **Third:** heuristic five phases, after phase GT supports an MS-TCN++ baseline.
- **Not yet:** YOLO11 full-video pose and the formal API. This sprint does not justify replacing either.

## ONE RECOMMENDED REFERENCE STACK

```text
OpenCV ingest + existing quality gates
 -> existing shot range / shooter candidates
 -> YOLO11 pose for full video
 -> RTMW body/wrist only on gated release ROI; hand-too-small means unavailable
 -> release-ball v1 anchors now
 -> RF-DETR Nano/Core as V2 detector challenger
 -> detector-only continuity now; gated CoTracker3 remains research-only
 -> explicit wrist-ball contact/separation state machine with abstention
 -> phase-v2 evidence now; MS-TCN++ after frozen GT
 -> confidence-aware normalized 2D metrics
 -> evidence-only constrained feedback
```

Grounded SAM 2 is outside the main stack and remains a small-volume research/annotation fallback.

SAHI is outside the selected stack until a single controlled native-versus-sliced rerun demonstrates a worthwhile gain. CoTracker3 is retained as a research adapter, not a Reference V1 default.

## Validation closure decision

`REFERENCE_V1_BUILD_READY = YES`

The selected build has pose evidence, release-ball anchors, contact-transition decoding, risk flags and `insufficient_data` behavior. Source-separated reviewer work is still required before accuracy or product claims.

## Immediate benchmark follow-ups

1. Complete reviewer 1 for the 10 pending strict-release shots, then obtain a real independent reviewer 2.
2. Freeze the six `IMG_7222` shots before threshold changes and run exact/1/2/3/5-frame metrics.
3. Run one controlled SAHI comparison when the existing CV runtime is executable; reject permanently if recall gain is not worth latency and duplicates.
4. Evaluate re-anchored CoTracker3 on the new hard-occlusion human centers; promote only if it beats detector-only usable coverage without excessive anchors.
5. Fine-tune RF-DETR Nano or Core and YOLO11 on one identical trusted V2 train split; compare on one frozen test.
6. Add shooter-association GT before further ByteTrack/BoT-SORT tuning.
7. Freeze phase GT before training MS-TCN++.

Do not expand manual labeling to thousands of frames before these small tests identify the actual bottleneck.

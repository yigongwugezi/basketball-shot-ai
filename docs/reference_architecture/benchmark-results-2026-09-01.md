# Reference V1 Benchmark Results

Date: 2026-09-01
Status: first executable sprint, not a product certification.

## Scope and environment

- CPU: Intel Core i7-14650HX, 16 cores / 24 logical processors.
- GPU present: RTX 4060 Laptop 8 GB, but this sprint used `torch 2.12.0+cpu` and ONNX Runtime CPU.
- Python 3.10.11, OpenCV 5.0.0, ONNX Runtime 1.23.2.
- Generated evidence: `E:\BasketballShotAI\analysis_runs\reference_benchmark\`.
- Harness: `benchmarks/reference_v1/` with adapters, cache, CSV/JSON summaries, runtime, failure log, trajectories and contact sheets.
- The ball benchmark uses 216 usable rows from the 230-frame manually reviewed set. It is `research_only_not_independent_product_test` and must not support product claims.
- Pose and tracking tests have no identity/pose ground truth. Coverage, temporal step and event-frame values are proxies, not accuracy.

## Executive result

The current stack should not be broadly replaced. The sprint supports three targeted additions: RF-DETR Nano as the next detector challenger, RTMW as an offline whole-body release-window challenger, and CoTracker3 only behind clean detector initialization plus re-anchoring. Grounded SAM 2 remains a research fallback. The new strict-release state machine is useful evidence plumbing but is not accurate enough to replace formal release logic.

## Candidate table

| Module | Candidate | Ran? | Input | Result | Runtime | Strength | Failure / limitation | Decision |
|---|---|---:|---|---|---:|---|---|---|
| Person tracking | largest-person baseline | yes | 3 clips, 183 frames | coverage 100%; event deltas +1/-27/-4 | 37.1-53.4 ms/frame | simplest, no ID dependency | broadcast clip likely selected a static/wrong person | KEEP |
| Person tracking | ByteTrack | yes | same 3 clips | coverage 100%; 0/0/1 reported switches | 35.2-50.2 ms/frame | persistent IDs | one switch on IMG_7216; no event stability gain | DEFER |
| Person tracking | BoT-SORT | yes | same 3 clips | coverage 100%; 0/0/1 reported switches | 46.4-77.6 ms/frame | appearance/motion tracker | slower; one IMG_7216 switch; no gain | REJECT for default, retain adapter |
| Pose | YOLO11 pose via largest-person baseline | yes | same 3 clips | 100% clip coverage; 10.5-15.8 visible points | 37.1-53.4 ms/frame including selection | current real-time body baseline | only 17 body points; wrong-person risk | KEEP |
| Pose | RTMPose | yes | every third frame, 63 frames | 100% coverage; 15.4-17 visible points | 665.7-875.1 ms/frame | independent pose challenger | 13-24x slower than YOLO; no clear proxy gain | DEFER |
| Pose | RTMW | yes | every third frame, 63 frames | 129.4-133 visible points; lower temporal step than RTMPose on all clips | 919.1-1176.9 ms/frame | whole-body/hand/foot evidence | CPU slow; single clip event proxy was +21 frames | ADD offline ROI only |
| Ball detection | YOLO11 COCO sports ball | yes | 216 reviewed frames | recall@IoU50 3.7%; small-ball recall 0%; 71 unmatched detections | 36.6 ms/frame | no custom training | unusable recall on this set | REJECT |
| Ball detection | release-ball v1 | yes | same 216 frames | recall 43.5%; small-ball recall 37.5%; 7 unmatched detections | 36.5 ms/frame | best current recall/runtime/unmatched-detection tradeoff | misses over half; training-related research set | KEEP prototype baseline |
| Ball detection | ball/rim/player smoke model | yes | same 216 frames | recall 0%; 0 unmatched detections | 34.1 ms/frame | multi-class interface | threshold/model does not transfer to this set | REJECT for release ball |
| Ball detection | RF-DETR Nano COCO | yes | same 216 frames | recall 49.1%; small-ball recall 37.5%; 90 unmatched detections | 114.7 ms/frame | highest measured recall | 3.1x slower than v1 and many unmatched detections; not fine-tuned | ADD challenger, do not replace yet |
| Ball tracking | detector center sequence | yes | two strict-release clips | detector coverage 75.6% / 93.5% | detector about 31-32 ms/frame | observable anchor evidence | gaps and wrong detections remain | KEEP anchor/fallback |
| Ball tracking | detector + OpenCV LK | yes | same clips | coverage 100% / 96.7%; drift proxy 16.97 / 5.51 ball diameters | 0.66 / 5.97 ms/frame tracker-only | very cheap control | severe drift; visibility is over-optimistic | KEEP control, not winner |
| Ball tracking | detector + CoTracker3 | yes | same clips | visible coverage 34.1% / 54.8%; drift proxy 17.16 / 0.18 | 515.2 / 481.6 ms/frame | excellent alignment on clean IMG_7216 initialization | catastrophic BILI_002 drift; CPU slow | ADD experimental with re-anchor gate |
| Video object propagation | Grounding DINO Tiny + SAM 2.1 Hiera Tiny | yes | IMG_7216 frames 128-142 | first box score 0.674; nonempty masks 5/15 | 2673.2 ms/frame | text-to-box-to-mask pipeline works locally | only 33.3% mask coverage; no mask GT | DEFER as research fallback |
| Strict release | geometry contact-transition v0 | yes | 3 historical strict-release clips | predictions 51/139/none; delta to historical ball +7/+1/none | 65.8-75.3 ms/frame | emits contact/separation evidence and risks | one large error and one abstention; wrist-only proxy | ADD experimental, no formal replacement |
| Five phases | phase-v2 dense ordered events | yes, existing evidence | historical 10-video diagnostic set | 10/10 complete ordered sequences; status 1 ok, 8 needs_review, 1 insufficient_data | historical runtime not recorded | explicit ordering, statuses and risk flags | no complete phase GT; completeness is not correctness | KEEP experimental |

## Tracking interpretation

All person methods reported full pose coverage, but this did not establish shooter correctness. On broadcast BILI_004 all three selected an almost static trajectory and produced an event proxy 27 frames early. ByteTrack and BoT-SORT did not solve shooter selection and each switched once on IMG_7216. The immediate problem is shooter association, not choosing a more sophisticated tracker.

CoTracker3 was highly sample-dependent. On IMG_7216 its detector-relative drift was 0.18 ball diameters versus LK's 5.51, but on BILI_002 both trackers drifted about 17 ball diameters. Learned tracking therefore needs clean initialization, detector re-anchors, ROI/size gates and an abstention rule.

## Pose interpretation

RTMW is the only challenger that adds a genuinely new evidence type: 133 whole-body points instead of 17. Its lower temporal-step proxy than RTMPose is encouraging, but no pose GT exists and CPU latency is roughly one second per sampled frame. Use it on a short release ROI for hand/foot evidence; keep YOLO11 pose for the full video.

## Detector interpretation

RF-DETR Nano achieved the highest research-set recall, 49.1%, but generated 90 unmatched detections compared with release-ball v1's 7. These are not a formal false-positive rate because the research set is not a complete independent negative test. This makes RF-DETR the next fine-tuning/independent-test candidate, not an immediate winner. The current v1 model remains prototype evidence only and the 216-frame result is not independent because it comes from the existing reviewed workflow.

## Strict release interpretation

The experimental path is now executable:

```text
coarse release window
 -> release-ball detections
 -> shooting-wrist candidate
 -> normalized ball-hand distance
 -> contact-to-separation transition
 -> stable post-release movement
 -> predicted frame or abstention + risk flags
```

It matched historical detector evidence within one frame on IMG_7216, was seven frames late on BILI_002, and abstained on BILI_010_B. This validates the interface and abstention semantics, not the algorithm. RTMW hand points and tracker re-anchoring are the next evidence additions after a small strict-release GT set exists.

## Phase interpretation

Phase-v2 produced ordered five-event sequences for all ten historical diagnostics, but only one was `ok`; eight require review and one is `insufficient_data`. Ordered completeness must not be presented as phase accuracy. HEAD heuristics remain the product path, phase-v2 remains read-only experimental evidence, and the future temporal adapter should compare cached features under exact event-frame and phase metrics once GT is frozen. MS-TCN++ remains the first trainable candidate; no temporal model was trained in this sprint.

## Failures and deferred candidates

- No candidate that was actually attempted remained blocked: RTMLib, CoTracker3, RF-DETR Nano, Grounding DINO and SAM 2.1 all ran locally.
- ViTPose, TAPIR/TAPNext++, learned hand-contact models, SAHI and MS-TCN++ were not run. They are deferred, not falsely reported as blocked.
- Grounded SAM 2 ran but failed the practical continuation gate on this smoke clip: only 33.3% nonempty-mask coverage at 2.67 seconds/frame CPU.

## Evidence locations

- Aggregate CSV/JSON: `E:\BasketballShotAI\analysis_runs\reference_benchmark\benchmark_results.*`
- Failure log: `E:\BasketballShotAI\analysis_runs\reference_benchmark\failures.jsonl`
- Per-candidate detail/cache: subdirectories under that root.
- Point trajectories/contact sheets: `ball_tracking\opencv_lk\` and `ball_tracking\cotracker3\`.
- Grounded SAM 2: `grounded_sam2\img_7216_near_agreement\`.
- Strict-release evidence JSON: `strict_release\`.

## Validity boundary

These measurements select engineering challengers only. They do not establish product accuracy because identity, pose, point tracking, mask and complete phase GT are absent, and the ball labels are research-only rather than an independent source-video-disjoint test.

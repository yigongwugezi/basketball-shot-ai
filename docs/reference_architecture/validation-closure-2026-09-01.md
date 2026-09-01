# Reference V1 Validation Closure

Date: 2026-09-01
Scope: Wave 2.5 only. This is an experimental evidence-pipeline decision, not product certification.

## Execution boundary

- Reused the Wave 2 cache under `E:\BasketballShotAI\analysis_runs\reference_benchmark\`; no identical model inference was repeated.
- Built one 22-shot micro-GT seed: 12 existing human-labeled release windows plus 10 pending shots from `IMG_7221` and `IMG_7222`.
- The 12 legacy windows overlap v1 detector training data and are development evidence only.
- `IMG_7221` contributes four pending development shots; `IMG_7222` contributes six pending source-separated frozen-evaluation candidates.
- One legacy row, `BILI_001_A`, has no recoverable source path/FPS in the current formal files; those fields remain blank instead of being invented.
- Reviewer 2 remains pending for every row. No algorithm prediction was represented as a second reviewer.
- The current managed execution environment could not start the repository Python 3.10 CV runtime because its base interpreter under `AppData` was denied. The available bundled Python has no CV packages, `sahi` is not installed in the existing venv, and network installation is unavailable. This blocked new model inference but not cached-evidence evaluation or review-tool generation.

Generated, uncommitted review evidence is under `tmp/reference_validation_closure/`. The preferred long-lived generated-output location remains `E:\BasketballShotAI\analysis_runs\reference_validation_closure\` when that drive is writable.

## Micro-GT status

| Group | Source IDs | Shots | Reviewer 1 | Reviewer 2 | Role |
|---|---|---:|---|---|---|
| Existing batches 001/003 | 12 clips across 12 IDs | 12 | imported legacy human annotation; identity unavailable | pending | development only; detector-training overlap |
| IMG_7221 | one source video | 4 | pending | pending | development |
| IMG_7222 | one source video | 6 | pending | pending | frozen evaluation candidate |

The review page captures `contact_last_supported_frame`, `separation_candidate_frame`, the first persistent supported no-contact frame, uncertainty, visibility, status, notes, and per-frame ball centers. `null`, `ambiguous`, and `insufficient_data` are valid outcomes.
Its CSV schema stores reviewer-1 and reviewer-2 event fields separately, and the evaluator supports exact or frame-tolerance disagreement reporting.

## SAHI

### Native versus sliced inference

| Candidate | Same weights/confidence/GT/IoU? | Ran this sprint? | Recall@IoU50 | Small-ball recall | Matched center error | Unmatched detections | Duplicate detections | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| release-ball v1 native 640 | yes for the existing 216-frame research result | prior evidence | 43.5% | 37.5% | not retained in prior cache | 7, historically named FP | not retained | 36.5 ms/frame |
| SAHI 320, overlap 0.20 | adapter preserves the same settings | no, runtime blocked | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| SAHI 480, overlap 0.20 | adapter preserves the same settings | no, runtime blocked | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

The harness now uses the correct term **unmatched detections**. The positive research set is neither a complete negative set nor an independent product test, so these detections cannot be presented as formal precision or false-positive rate.

### SAHI verdict

**REJECT from Reference V1 now.** There is no measured sliced-recall gain to justify latency and duplicate-merging cost. This rejects immediate inclusion, not the SAHI technique in general. One future rerun is sufficient when the existing CV runtime is executable; no parameter sweep is warranted.

## CoTracker3 with detector re-anchor

The validation module implements the requested minimal chain:

```text
trusted detector observation
 -> CoTracker3 point
 -> periodic detector re-anchor
 -> displacement and expected-size plausibility gate
 -> visibility/reinitialize gate
 -> explicit abstention
```

No Kalman filter or additional learned model was added.

### Human ball-center result

Input: `IMG_7216`, `NEW_003`, frames 127-150, 24 human-labeled visible-ball frames. These annotations overlap v1 training and are development evidence, not frozen product test. The legacy trajectory cache did not retain detector box size, so the re-anchor cache evaluation used a fixed 60 px development diameter hint; this limitation is recorded in the output.

| Method | Median center error | Mean center error | Median error / ball diameter | Visibility F1 | Track survival | Bad drift frames | Re-anchors | Usable coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Detector only | 2.72 px | 2.58 px | 0.042 | 1.000 | 100.0% | 0 | 0 | 100.0% |
| Original CoTracker3 | 8.71 px | 7.85 px | 0.140 | 0.588 | 41.7% | 0 | 0 | 41.7% |
| Re-anchored CoTracker3 | 3.40 px | 4.51 px | 0.055 | 1.000 | 100.0% | 0 | 16 | 100.0% |

The prior CoTracker runtime remains about 481.6 ms/frame on CPU. The re-anchor layer restored coverage but required 16 anchors in 24 frames and did not beat detector-only center error. It therefore acted mostly as a gated detector fallback rather than adding independent continuity.

`BILI_002` still has no human ball-center GT. Applying the re-anchor rules to its cached trajectory produced 95.1% proxy coverage, 26 anchors over 41 frames, and two abstentions. The detector-agreement distance was zero at the median because the result repeatedly re-anchored to the detector. This is **not tracking accuracy** and confirms that the missing GT cannot be bypassed with a detector proxy.

### CoTracker verdict

**RESEARCH ONLY.** Re-anchoring is valuable as a safety mechanism, but CoTracker3 has not shown gain over detector-only on human GT and is CPU-expensive. It does not enter the Reference V1 default execution path. Retain the adapter for a later hard-occlusion evaluation after reviewer-1 ball centers are complete.

## RTMW hand usability

The question was narrowed from “does RTMW return 133 points?” to “are the release-hand points physically usable in a real release ROI?” Cached RTMW frames within release +/-12 were checked with a conservative automatic plausibility proxy: median point confidence, at least 18 px hand span, hand-root/body-wrist proximity, and temporal jump gate. Human visual review remains pending, so this is a usability proxy, not landmark accuracy.

| Sample | Assumed hand | Frames | Proxy usable | Known limitation |
|---|---|---:|---:|---|
| BILI_010_A | right | 9 | 0/9 (0.0%) | hand mostly 7-17 px; one larger result jumped implausibly |
| BILI_004_A broadcast | left | 9 | 7/9 (77.8%) | prior benchmark likely followed a static/wrong prominent person; not trusted shooter evidence |
| IMG_7216 | right | 8 | 3/8 (37.5%) | large release/post-release temporal jumps |
| All cached samples | mixed | 26 | 10/26 (38.5%) | includes untrusted broadcast association |
| Shooter-credible samples only | right | 17 | 3/17 (17.6%) | no human hand-landmark GT |

### RTMW hand verdict

**WRIST/BODY ONLY.** RTMW remains useful for offline body/wrist ROI evidence. Finger/palm evidence does not enter Reference V1. Apply `hand_too_small -> unavailable`; do not infer contact from a nominal 133-point output.

## Strict release contact-transition decoder v1

Decoder v1 uses the pose release only as a coarse prior, then requires supported contact followed by persistent normalized separation and post-release ball movement. It emits `insufficient_data` instead of forcing a frame. RTMW hand and CoTracker are not default inputs because their experiments did not pass the inclusion gate.

### Four-method comparison on confirmed human GT

Only `IMG_7216 / NEW_003` currently has both cached predictions for all four methods and a human strict frame (`138`). Reporting a 12- or 22-shot accuracy table before the pending review is completed would be fabricated.

| Method | Predicted frame | Absolute error | Exact | Within +/-1 | Within +/-2 | Within +/-3 | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Current pose release | 136 | 2 | no | no | yes | yes | pose evidence |
| Existing release fusion diagnostic | 136 | 2 | no | no | yes | yes | final source remains pose; diagnostic only |
| Geometry contact-transition v0 | 139 | 1 | no | yes | yes | yes | experimental |
| Contact-transition decoder v1 | 138 | 0 | yes | yes | yes | yes | experimental `ok` |

On the two additional cached clips without human strict GT, v1 abstained:

- `BILI_002`: the detector evidence became spatially static after apparent separation, so persistence/motion support failed.
- `BILI_010_B`: the only early transition was outside the pose-prior window, so the decoder rejected it rather than accepting a large early delta.

This is better failure semantics than forcing a plausible-looking number. It is not a frozen evaluation result. Clear/occluded/blurred slice metrics remain pending because the source-separated reviewer-1 set is not yet labeled.

### Strict release verdict

**Adopt contact-transition as the main experimental strict-release route.** Keep `release_pose_frame` separate, preserve detector/geometry evidence, and abstain on visibility or persistence failure. Do not replace the formal API in this sprint.

## FINAL FOUR DECISIONS

1. **SAHI enters Reference V1? NO.** No runnable evidence of gain; do not add latency speculatively.
2. **CoTracker3 enters Reference V1? CONDITIONAL.** Keep as research-only behind trusted anchors; it is not the default until it beats detector-only on source-separated human GT.
3. **RTMW hand evidence enters Reference V1? NO.** Use RTMW body/wrist only; hand points are gated unavailable when too small or unstable.
4. **Contact-transition becomes the strict-release main route? YES.** It is the experimental main route with explicit abstention, not a product-certified frame detector.

# Reference V1 Build Gate

`REFERENCE_V1_BUILD_READY = YES`

Reference V1 already has executable pose evidence, ball detector evidence, a contact-transition decoder, explicit risk flags, and `insufficient_data` semantics. The build does not depend on SAHI, CoTracker3, or RTMW fingers. Pending reviewer work limits accuracy claims but does not block an experimental evidence pipeline.

## Evidence locations

- Committed harness code: `benchmarks/reference_v1/`
- Generated micro-GT seed/page: `tmp/reference_validation_closure/micro_gt_seed.csv` and `strict_release_review.html`
- Generated cached evaluations: JSON files under `tmp/reference_validation_closure/`
- Prior model outputs: `E:\BasketballShotAI\analysis_runs\reference_benchmark\`

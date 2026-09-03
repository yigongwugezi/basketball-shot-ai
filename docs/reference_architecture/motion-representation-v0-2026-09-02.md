# Basketball Shot Motion Representation V0 — 2026-09-02

> Historical design record. Superseded by the canonical Motion Representation V1 frozen on 2026-09-03; see `reference-v1-final-acceptance-2026-09-03.md`.

## Status

`MOTION_REPRESENTATION_V0 = READY`

This status means the schema, builder, validation, uncertainty propagation, and developer debugger are operational. It does not mean pose accuracy or event GT is closed. Accordingly, `POSE_READY_FOR_MOTION_UNDERSTANDING = PARTIAL`.

## Model

`reference_v1/motion.py` introduces `ShotMotionRepresentation` with the following layers:

- `input_quality`, `view`, `pose_reliability`, and `ball_reliability`;
- sparse `events`;
- interval `phases`;
- factual `primitives`;
- human–ball and temporal `relations`;
- `kinematics`, including normalized shot time;
- `uncertainty` and `provenance`.

Every fact carries `value`, `unit`, `status`, `confidence`, `frame_range`, `evidence`, and `source`. A missing detector result produces `insufficient_data`; the builder does not invent a ball path, landing, or coordinate. Slow-motion inputs explicitly add `no_real_time_coordination_claim`, and contaminated sources add `no_generalization_claim`.

The V0 primitive vocabulary includes body lowering/rising, knee flexion/extension, hip rise, ball lowering/rising, shooting-wrist rise, elbow extension, shoulder elevation, trunk-lean change, takeoff, airborne interval, release, follow-through, landing, and forward drift. Unsupported primitives remain present with an unavailable status rather than a fabricated number.

## Event, phase, and relation separation

- **Event:** one sparse frame such as bottom, takeoff, pose release, strict ball release, body apex, or landing.
- **Phase:** an interval such as dip, upward drive, or follow-through.
- **Relation:** a fact connecting events, primitives, body, or ball, such as elbow-extension onset relative to release or release relative to apex.

Normalized shot time uses `bottom = 0` and strict release (or pose release only when strict release is unavailable) `= 1`. It compares sequencing across FPS without converting a slow-motion source into a real-time claim.

## Generated evidence examples

These are current heuristic/proxy facts, not human event GT:

| Sample | Elbow extension onset | Ball rise onset (normalized) | Strict release vs body apex | Ball rise continuity | Post-release ball evidence |
|---|---|---:|---|---|---:|
| IMG_7215 | 11 frames before strict release | 0.4545 | 5 frames after estimated apex | no pause candidate in observed steps | 16 frames |
| IMG_7216 | 24 frames before strict release | 0.6957 | 3 frames after estimated apex | no pause candidate in observed steps | 13 frames |

The large difference in elbow-onset timing is a diagnostic signal that the event heuristic needs GT, not a coaching conclusion. Forward drift remains a 2D pixel displacement proxy, not real-world distance.

## Event quality audit

No independent human event GT currently exists for these clips.

| Event | Current detector/evidence | Known weakness | Confidence semantics | Recommended next step |
|---|---|---|---|---|
| dip start | first knee-angle crossing at 20% of pre-bottom bend range | projection, filter, and baseline-window sensitive | signal coverage capped at 0.8 | label a small event set before product interpretation |
| bottom | argmax of 65% knee flexion + 35% hip lowering | camera motion and smoothing can move the maximum | composite heuristic peak | retain as provisional fact |
| takeoff | ankle rises ≥0.015 image-height from baseline | shoe visibility and camera motion; not force-plate takeoff | ankle coverage capped at 0.8 | downgrade when feet are obscured |
| pose release | peak of wrist height, elbow extension, and wrist velocity | IMG_7215 moved 122→125 under filtering; agreement with strict release is not validation | normalized composite score | human event label required; keep distinct from strict release |
| strict release | persistent ball–wrist separation after contact proxy | prototype detector, no learned hand-contact model | observation-count cap at 0.85 plus risk flags | keep experimental semantics unchanged |
| body apex | minimum hip-y after bottom | camera motion and pose identity affect the estimate | pose signal coverage cap | require stable-camera evidence or mark low confidence |
| landing | ankle return near pre-shot baseline | clip often ends first; occlusion common | ankle coverage cap at 0.75 | unavailable is preferable to guessed landing |
| elbow extension onset | ≥5° increase over a four-frame window | sensitive to wrist/elbow localization and smoothing | currently inherits timing metric confidence | label onset or validate against joint GT before downstream use |

The audit does not change strict-release semantics or force unavailable events to have values.

## Developer debugger

`benchmarks/reference_v1/motion_debugger.py` writes:

- `motion_representation_v0.json`;
- `motion_debugger.html`.

The standalone page synchronizes an annotated video with clickable phases/events and plots shooting-wrist height, elbow angle, trunk angle, and detected ball height. It is a research artifact, not product UI.

Generated examples:

- `E:\BasketballShotAI\analysis_runs\pose_reliability_pass\accepted_img_7215\motion_debugger.html`
- `E:\BasketballShotAI\analysis_runs\pose_reliability_pass\accepted_img_7216\motion_debugger.html`

## Boundary

V0 may state: “elbow extension began 11 frames before strict release.” It may not state that this is good, bad, early, late, or something the player should change. Those claims require pose/event GT plus basketball expertise and are intentionally outside this task.

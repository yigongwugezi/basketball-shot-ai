# Human-Object Spatiotemporal Motion Intelligence Problem Tree

Date: 2026-09-01
Product application: Basketball Shot AI
Search rule: **search by problem, not by sport**.

## 0. System objective

The system must turn ordinary video into traceable evidence about a person, an object, the environment, and their evolution over time. Basketball shooting is the first application, not the boundary of the technology.

```text
Video
  -> validity and quality gate
  -> candidate action intervals
  -> person/object/environment perception
  -> temporally continuous tracks
  -> human-object interaction states
  -> precise events and ordered phases
  -> view-aware kinematics with uncertainty
  -> comparative skill evidence
  -> actionable, bounded feedback
```

Each leaf below is a separately measurable engineering problem. A downstream result must not silently hide failure in an upstream leaf.

## 1. Video Understanding

### 1.1 Input validity

- Decode validity: can all required frames be decoded with stable timestamps?
- Metadata integrity: FPS, variable-frame-rate status, frame count, duration, resolution and orientation.
- Subject visibility: is the shooter large enough and present for the complete action?
- Object visibility: is the ball visible often enough around the release interval?
- View suitability: side, front, diagonal, rear, broadcast or unknown.

### 1.2 Quality

- Blur detection: global blur versus local ball/hand motion blur.
- Low light and noise: confidence degradation rather than cosmetic scoring.
- Resolution and scale: pixels occupied by person, hand and ball.
- Frame rate: temporal quantization error for release and phase boundaries.
- Compression artifacts, dropped/duplicate frames and rolling shutter.
- Output: quality flags, uncertainty multipliers and an explicit `insufficient_data` decision.

### 1.3 Long-video structure

- Camera-cut detection and replay separation.
- Camera motion and zoom detection.
- Shot/action proposal: find candidate shooting intervals without classifying every frame deeply.
- Long-video segmentation: split repeated attempts while preserving one `source_video_id`.
- Basketball mapping: setup-to-landing windows, multiple attempts per source video.

### 1.4 Evaluation

- Decode failure rate; cut F1; proposal recall at temporal IoU; false proposals/minute.
- Quality-gate calibration: probability that a downstream metric is trustworthy conditional on the gate.

## 2. Human Perception

### 2.1 Person and identity

- Person detection.
- Shooter/person-of-interest selection using ball proximity, motion and temporal persistence.
- Identity tracking through short occlusion and crossings.
- Failure mode: selecting a defender or bystander with better pose confidence.

### 2.2 Pose hierarchy

- 2D body pose: coarse limbs and torso.
- Whole-body pose: body, feet, hands and face in one coordinate system.
- Foot pose: stance, take-off and landing support.
- Hand/finger pose: wrist-chain and ball-contact evidence.
- Monocular 3D body/hand mesh: research evidence, not automatically metric biomechanics.
- Occlusion robustness: uncertainty per joint, temporal recovery and no hallucinated joints.

### 2.3 Evaluation

- AP/PCK by joint group, not one aggregate score.
- Temporal jitter, missing-run length, left/right swaps and bone-length consistency.
- Shooter identity consistency and usable-frame recall around strict release.

## 3. Object Perception

### 3.1 Detection

- Closed-set object detection for basketball, rim and backboard.
- Open-vocabulary detection for bootstrapping and research-only prelabels.
- Small-object detection at native/tiled resolution.
- Fast-object detection under blur and partial hand occlusion.
- Generic sports equipment transfer: tennis ball, shuttlecock, baseball and table-tennis ball.

### 3.2 Representation

- Bounding box for training and coarse geometry.
- Center point or heatmap for motion continuity.
- Mask when exact visible extent matters.
- Visibility and occlusion state separate from location.

### 3.3 Evaluation

- AP is insufficient: report release-window recall, small-object recall, center error, visibility precision and false positives per frame.
- Test by unseen `source_video_id`, person, view and scene.

## 4. Object Motion

### 4.1 Temporal continuity

- Object tracking from sparse detector anchors.
- Point tracking after detector initialization.
- Mask propagation after a reliable mask prompt.
- Detector-to-tracker fusion with confidence and reinitialization.

### 4.2 Occlusion and blur

- Short occlusion recovery: hand covers part of the ball for several frames.
- Long occlusion/re-identification: ball disappears behind body or leaves frame.
- Motion-blur-aware localization: heatmap/blur orientation rather than box only.
- Temporal interpolation constrained by visibility and physical plausibility.

### 4.3 Trajectory

- Image-coordinate trajectory and velocity.
- Camera-motion-compensated trajectory.
- Optional calibrated/world trajectory; never infer metric speed without scale/calibration.

### 4.4 Evaluation

- Track survival, average point accuracy, occlusion accuracy, reinitialization count and center error.
- Release-window continuity and maximum unsupported interpolation gap.

## 5. Human-Object Interaction

### 5.1 Contact evidence

- Possession: ball associated with the selected shooter.
- Object-near-hand: geometric candidate, not proof of contact.
- Hand-object contact: visible hand/ball relationship and contact confidence.
- Contact transition: `contact -> separating -> no_contact`.
- Object release: first persistent no-contact frame after verified contact.
- Reacquisition: rebound/catch must not be confused with a second release.

### 5.2 Interaction state machine

```text
unknown
  -> candidate_possession
  -> contact_confirmed
  -> separation_candidate
  -> released_confirmed
  -> flight / out_of_view / reacquired
```

- Transitions require temporal persistence and evidence from ball, hand and motion.
- Missing evidence yields `insufficient_data`, not a fabricated frame.
- Basketball-specific fine-tuning is required because general hand-object datasets emphasize grasps and egocentric household objects.

### 5.3 Evaluation

- Contact-state F1, transition frame error, release precision/recall and failure reason distribution.

## 6. Precise Temporal Understanding

### 6.1 Tasks

- Action segmentation: label every frame/interval.
- Temporal action localization: detect action segments in long video.
- Event spotting: predict sparse event timestamps.
- Precise event spotting: frame-level localization with strict tolerance.
- Ordered event decoding: enforce plausible phase order without forcing every event to exist.

### 6.2 Basketball mapping

- Setup: stable possession before downward preparation.
- Dip: ball/body descend and load.
- Jump/drive: upward kinetic-chain initiation or set-shot drive.
- Strict release: persistent hand-ball separation, not merely wrist extension.
- Follow-through: post-release arm/wrist continuation.
- Landing/recovery: feet return and body stabilizes when visible.

### 6.3 Cross-domain search targets

- Golf swing sequencing, tennis serve/hit events, badminton hit detection.
- Diving/gymnastics phase-aware assessment.
- Weightlifting and rehabilitation repetitions.
- Surgical and industrial procedural segmentation.

### 6.4 Evaluation

- Per-event frame error and mAP at 0/1/2/3/5-frame tolerances.
- Segment F1@IoU, edit score, frame accuracy and phase-order violations.
- Report temporal quantization from FPS as part of uncertainty.

## 7. Motion Reconstruction

### 7.1 Signal processing

- Confidence-aware keypoint smoothing.
- Missing-point interpolation with maximum gap limits.
- Normalized image coordinates using person scale and stable reference axes.
- Camera-motion compensation before differentiating coordinates.

### 7.2 Geometry

- Camera intrinsics/extrinsics and court calibration where visible.
- Homography for planar court quantities only.
- Monocular 3D pose/body mesh with scale/depth ambiguity.
- Multi-view 3D triangulation and synchronization for research-grade validation.
- Hand mesh around release where pixel scale supports it.

### 7.3 Evaluation

- Reprojection error, calibration error, 2D/3D joint error and drift.
- Compare against marker-based or validated markerless references by task and view.

## 8. Kinematics / Biomechanics

### 8.1 Measurable features

- Joint angles and phase-relative ranges.
- Angular velocity only after FPS validation and smoothing.
- Segmental timing and proximal-to-distal coordination proxies.
- Body alignment, stance symmetry, take-off/landing balance and within-session consistency.
- Ball release height/angle/velocity only with sufficient geometry; otherwise image-plane proxies.

### 8.2 Scientific limits

- Single-camera 2D is strongest for motion near the image plane.
- Monocular 3D is not automatically force, torque, power or clinical-grade joint kinetics.
- Population, shot type, distance, defense, fatigue and camera view condition every recommendation.
- Output must include uncertainty, evidence frames and allowed wording.

### 8.3 Evaluation

- Angle/velocity error against reference, repeatability, ICC, Bland-Altman limits and view dependency.
- Reliability must be measured per metric, not declared for the whole model.

## 9. Skill Understanding

- Skill recognition: which procedure/shot is performed.
- Fine-grained skill estimation: proficiency without pretending to know causality.
- Novice/expert comparison and movement style.
- Procedure understanding: which expected keysteps occurred and in what order.
- Exemplar retrieval: find comparable attempts under matched view and task.
- Evaluation: subject-disjoint ranking correlation, balanced accuracy, pairwise preference and calibration.

## 10. Action Quality Assessment

- Global AQA score.
- Phase-aware AQA using localized sub-actions.
- Pose-guided AQA combining appearance and skeleton evidence.
- Explainable AQA with evidence clips and explicit rules.
- Causal AQA that tests whether foreground motion, not background, drives predictions.
- Comparative AQA against the user's own history or a matched exemplar.
- Consistency assessment across repeated attempts.
- Evaluation: Spearman/Pearson, relative score error, pairwise ranking, uncertainty calibration and explanation faithfulness.

## 11. Feedback / Coaching

- Mistake localization: phase, body part, frame and confidence.
- Evidence-based feedback: every statement links to observable measurements.
- Actionable feedback: one or two prioritized changes, not a generic report dump.
- Rule-based + learned hybrid: learned perception, explicit thresholds/conditions, retrieved evidence.
- Neuro-symbolic reasoning: rule programs over validated phase/kinematic evidence.
- Expert demonstration retrieval/generation: future, never present generated motion as measured fact.
- Allowed language must reflect measurement strength; safety/medical claims are out of scope.

## 12. Evaluation Architecture

### 12.1 Layered metrics

- Detection: AP, small-object and release-window recall.
- Tracking: point/visibility accuracy, continuity and drift.
- Interaction: contact-state F1 and release frame error.
- Temporal: event tolerance, phase F1/edit/order.
- Pose: spatial accuracy plus temporal stability.
- Kinematics: reference error and uncertainty calibration.
- AQA: subject-disjoint ranking/regression and explanation faithfulness.
- Feedback: expert agreement, usefulness, specificity, evidence correctness and harmful-claim rate.

### 12.2 Data separation

- Split by `source_video_id`; never random frame split.
- Independent test is frozen and excluded from model selection.
- Separate in-domain, new-person, new-view and hard-visibility test slices.
- Product claims use trusted independent test only.

## 13. Deployment

- GPU baseline for research and batch analysis.
- CPU fallback for quality gates, classic geometry and lightweight models.
- Export candidates: ONNX, TensorRT, CoreML and TFLite only after numerical parity tests.
- Track latency, peak memory, model/weights license and cloud inference cost per module.
- Graceful degradation: omit unsupported metrics rather than inventing them.

## 14. Dependency and evidence graph

```text
quality gate
  -> person/object visibility
    -> body/hand/ball observations
      -> continuous tracks
        -> contact transition and strict release
          -> ordered phases
            -> view-aware kinematics
              -> comparative quality evidence
                -> bounded coaching feedback
```

The graph defines the product rule: a downstream module may lower confidence or return unavailable, but may not erase upstream uncertainty.

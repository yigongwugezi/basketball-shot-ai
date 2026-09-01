# Global Cross-Domain Technology Landscape

Date: 2026-09-01
Scope: Human-Object Spatiotemporal Motion Intelligence, with basketball shooting as the first application.

## Method and evidence policy

- Search order was general computer vision, transferable sport/skill domains, then basketball-specific work.
- Existing internal research was deduplicated against `tmp/codex_handoff/HANDOFF_RESEARCH.md` and `docs/basketball-shot-analysis-external-project-study-2026-08-29.md`.
- `Tested` means tested by this project on project material. Reading a paper, opening a demo, or finding a checkpoint is not a project test.
- `License` separates code, weights and dataset where known. `UNKNOWN` is intentional and blocks production adoption until resolved.
- Metrics are the authors' benchmark claims, not results on our basketball videos.
- Resource count in this document: **130 meaningful resources**: approximately 66 papers/surveys with substantive method or evaluation, 51 code repositories/toolkits, 29 datasets/benchmarks, and 12 basketball-biomechanics sources. Hybrid paper+repo resources count once in the resource total and may count in multiple type totals.

## Record schema

Every record includes: name/year/domain/original problem; paper/repo/dataset/site; module; input/output; spatial and temporal method; training/weights; benchmark/metrics; strengths/limits; maturity; code/weights/dataset license and commercial risk; prior study/test/runnability; transfer, basketball fine-tuning, decision and exact project value.

# A. Skill, procedure and expert feedback

## R001 Ego-Exo4D

- **Year/domain/problem:** 2024, egocentric/exocentric skilled activity; multi-view procedure, proficiency and expert commentary understanding.
- **Links/type:** paper https://arxiv.org/abs/2311.18259 ; site/dataset https://ego-exo4d-data.org/ ; docs https://docs.ego-exo4d-data.org/benchmarks/proficiency_estimation/ . Paper + dataset/benchmark.
- **Module/I/O:** synchronized ego/exo video and optional pose/audio -> keysteps, proficiency and timestamped `good execution`/`needs improvement` commentary.
- **Architecture:** benchmark-dependent video encoders; temporal keysteps and commentary timestamps; spatially privileged multi-view observations. Training required; pretrained benchmark baselines exist. Benchmark: 1,286 h, 740 participants, 123 contexts; mAP at temporal tolerances for demonstrations.
- **Strength/limit:** uniquely grounds expert feedback in time and view; enormous, access-controlled and not basketball-form ground truth. Mature benchmark. Dataset terms must be checked; code/weights vary; commercial risk HIGH/UNKNOWN.
- **Project state:** studied yes; tested no; full download intentionally not attempted; runnable only after access. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** copy the timestamped evidence schema and separate proficiency, good execution and improvement labels instead of asking an LLM for an ungrounded verdict.

## R002 ExpertAF

- **Year/domain/problem:** 2025, expert action feedback; generate actionable commentary and corrected demonstrations.
- **Links/type:** paper https://arxiv.org/abs/2408.00672 ; project https://vision.cs.utexas.edu/projects/ExpertAF/ . Paper/project.
- **Module/I/O:** video + 3D body pose -> free-form feedback plus retrieved/generated corrected expert motion.
- **Architecture:** visual/pose representation aligned to weak expert commentary, then correction generation; temporal comparison and spatial pose evidence. Training required; public assets status varies. Evaluated by human preference and Ego-Exo4D-style tasks.
- **Strength/limit:** demonstrates observation-to-correction-to-demonstration chain; generated correction can hallucinate and depends on 3D pose quality. Research maturity. License UNKNOWN; commercial risk HIGH until code/model/data terms are verified.
- **Project state:** studied yes; tested no; runnable uncertain. Basketball fine-tune yes. **Decision: research_only. EXACT VALUE:** informs a future feedback layer that first cites phase/body evidence, then retrieves a real matched exemplar; generation stays out of V1 claims.

## R003 ExAct

- **Year/domain/problem:** 2025, skilled action analysis; evaluate expert-level video-language reasoning rather than generic captions.
- **Links/type:** paper https://arxiv.org/abs/2506.06277 . Paper/benchmark.
- **Module/I/O:** skilled-action video and question -> structured multiple-choice expert analysis.
- **Architecture:** benchmark, not a production architecture; expert commentary is refined into discriminative alternatives. Metrics: answer accuracy across skill-analysis dimensions.
- **Strength/limit:** exposes that plausible language is not expert correctness; no basketball release localization. Code/data/license UNKNOWN; commercial risk UNKNOWN.
- **Project state:** studied yes; tested no; runnable uncertain; basketball fine-tune not applicable for benchmark use. **Decision: idea_reference. EXACT VALUE:** use adversarial alternative answers to evaluate whether coaching text identifies the right phase/error instead of merely sounding professional.

## R004 SkillSpotter

- **Year/domain/problem:** 2026, cross-view skilled action; jointly detect action intervals and grade correct versus needs-improvement.
- **Links/type:** paper https://arxiv.org/abs/2606.31127 ; repo https://github.com/eth-siplab/SkillSpotter . Paper + repository.
- **Module/I/O:** ego/exo video + optional 3D pose -> temporal skilled-action detections and binary quality labels.
- **Architecture:** adaptive temporal suppression, gated pose fusion, bidirectional cross-view attention. Training required; public code, weights status to verify. Reported class mAP 21.82 and balanced accuracy 60.40 on its benchmark.
- **Strength/limit:** directly joins localization and quality; grading remains difficult and near 60% balanced accuracy. License UNKNOWN; new research code; commercial risk UNKNOWN.
- **Project state:** newly studied; tested no; runnable potentially. Basketball fine-tune yes. **Decision: benchmark_later. EXACT VALUE:** after phase labels exist, test gated pose fusion for localized `good/needs_improvement` evidence rather than one whole-shot score.

## R005 BASKET

- **Year/domain/problem:** 2025, basketball skill estimation; large-scale player skill recognition in broadcast video.
- **Links/type:** site https://sites.google.com/cs.unc.edu/basket ; CVPR paper linked there. Dataset/benchmark.
- **Module/I/O:** long basketball highlight video -> 20 fine-grained skills and five-level player proficiency.
- **Architecture:** benchmark baselines over long video; temporal aggregation of broadcast observations. Training required; dataset access/weights per site. 4,477 h, 32,232 players; published models remain below 30% while experts lead by 31 points.
- **Strength/limit:** direct basketball scale and skill labels; broadcast identity/style is not frame-level shooting biomechanics. License/terms require confirmation; commercial risk HIGH.
- **Project state:** studied yes; tested no; downloading prohibited. Fine-tune yes. **Decision: benchmark_later. EXACT VALUE:** future player/style retrieval and coarse skill context, not strict release or body-part correction.

## R006 PROSKILL

- **Year/domain/problem:** 2026, industrial/procedural skill; efficient action-level absolute and pairwise skill annotation.
- **Links/type:** repo/dataset https://github.com/fpv-iplab/ProSKILL_WACV . Paper + repository/benchmark.
- **Module/I/O:** procedural clips + comparative judgments -> Elo/Swiss rankings, absolute skill labels and agreement.
- **Architecture:** annotation/ranking framework rather than perception model; temporal subgoal clips, no special spatial model. No training required for ranking tools; benchmarks aggregate Assembly101, Ego-Exo4D, IKEA and others.
- **Strength/limit:** pairwise judgments reduce burden and expose agreement; rankings are task/population dependent. License UNKNOWN in current verification; source datasets have separate terms.
- **Project state:** studied yes; tested no; scripts likely runnable. Basketball fine-tune/data yes. **Decision: adopt_candidate. EXACT VALUE:** rank two matched attempts or a user attempt versus an exemplar before attempting unreliable absolute 0-100 scoring.

## R007 Assembly101

- **Year/domain/problem:** 2022, industrial assembly; free-order procedures, mistakes, corrections and multi-view hand-object actions.
- **Links/type:** paper https://assembly-101.github.io/assets/Assembly101.pdf ; site https://assembly-101.github.io/ . Dataset/benchmark.
- **Module/I/O:** 8 static + 4 ego synchronized views -> 1M fine action segments, 18M 3D hand poses, skill/mistake labels.
- **Architecture:** benchmark for recognition, anticipation, temporal segmentation and mistake detection. Metrics include F1/edit/MoF depending task.
- **Strength/limit:** excellent sequence variation and error annotation; toy assembly contact differs from ballistic release. Dataset license must be checked; commercial risk UNKNOWN/HIGH.
- **Project state:** studied yes; tested no; full data not downloaded. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** model ordered but non-rigid procedures and explicit corrections without forcing every shot into one canonical timing.

## R008 SkillSight

- **Year/domain/problem:** 2026, egocentric skill; efficient skill assessment using gaze as privileged supervision.
- **Links/type:** repo https://github.com/JasonWUCHI/SkillSight . Paper + repository.
- **Module/I/O:** first-person video, teacher gaze during training -> skill prediction from video at inference.
- **Architecture:** gaze teacher/student distillation; temporal video representation, gaze-weighted spatial evidence. Training required; code available, weights/license UNKNOWN.
- **Strength/limit:** teaches attention without requiring gaze at inference; phone exocentric basketball lacks aligned gaze supervision. New research maturity, commercial risk UNKNOWN.
- **Project state:** studied yes; tested no. Basketball fine-tune and new gaze data would be required. **Decision: reject for V1 / idea_reference. EXACT VALUE:** only the privileged-attention idea may transfer; do not add gaze collection to the current product.

## R009 ExpertEdit

- **Year/domain/problem:** 2026, motion editing; transform a novice performance toward expert motion.
- **Links/type:** paper https://arxiv.org/abs/2604.10466 . Paper.
- **Module/I/O:** novice motion + expert reference -> skill-aware corrected motion.
- **Architecture:** learned motion editing conditioned on expert examples; temporal motion generation and spatial body representation. Training/weights/license UNKNOWN.
- **Strength/limit:** compelling demonstration concept; generated motion is not measured evidence and can be unsafe. Early research, commercial risk HIGH.
- **Project state:** studied yes; tested no; runnable uncertain; basketball fine-tune yes. **Decision: research_only. EXACT VALUE:** future visualization after factual feedback is validated; never use generated pose as proof of the user's error.

## R010 Fitness-AQA

- **Year/domain/problem:** 2022, fitness; detect subtle workout form errors under camera/clothing/occlusion variation.
- **Links/type:** repo/dataset https://github.com/ParitoshParmar/Fitness-AQA . Paper + repository/dataset.
- **Module/I/O:** in-the-wild exercise video -> fine-grained form class/quality.
- **Architecture:** domain-informed self-supervised pose contrastive learning and pose/appearance disentanglement; temporal synchronization. Training required; dataset access by form.
- **Strength/limit:** direct form-feedback analogy and scene-invariance; only three exercises and no ball interaction. Dataset explicitly non-commercial; code license not confirmed; commercial risk HIGH.
- **Project state:** studied yes; tested no; runnable after data access. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** separate motion quality from appearance/background and evaluate subject-disjoint, rather than learning court aesthetics.

# B. Precise temporal events and action phases

## R011 Spotting Temporally Precise, Fine-Grained Events / PTS

- **Year/domain/problem:** 2022, multi-sport; localize instantaneous fine events to exact frames.
- **Links/type:** paper https://arxiv.org/abs/2207.10213 ; project https://jhong93.github.io/projects/spot.html ; repo https://github.com/SoccerNet/PTS-baseline . Paper + repository/benchmark.
- **Module/I/O:** trimmed/untrimmed video -> event class and frame timestamp.
- **Architecture:** frame-feature extraction plus high-resolution temporal spotting; spatial person crop and temporal context. Training required; baseline checkpoints/code available. Metric: mAP at tight frame tolerances.
- **Strength/limit:** correct task formulation for strict release; visual event classes still need labels. Repo BSD-3; dataset terms vary; low code risk.
- **Project state:** studied yes; tested no; runnable likely. Basketball fine-tune yes. **Decision: benchmark_now after labels. EXACT VALUE:** adopt exact-frame event protocol and 0/1/2/3/5-frame error before changing the model.

## R012 T-DEED

- **Year/domain/problem:** 2024, tennis/diving/gymnastics/skating/soccer; temporally discriminative precise event spotting.
- **Links/type:** repo https://github.com/arturxe2/t-deed ; paper linked in repo. Paper + repository.
- **Module/I/O:** video -> fine-grained event timestamps.
- **Architecture:** encoder-decoder over multiple temporal scales with temporal discriminability enhancement; local spatial features and long/short context. Training required; checkpoints and single-video inference available.
- **Strength/limit:** strong direct event architecture and broad sport transfer; needs basketball event labels and GPU training. GPL-3 code creates integration/commercial risk MEDIUM/HIGH.
- **Project state:** newly studied; tested no; likely runnable in isolated benchmark. Basketball fine-tune yes. **Decision: benchmark_later. EXACT VALUE:** first learned challenger for strict release and ordered phase events once a source-video-disjoint label set exists.

## R013 AdaSpot

- **Year/domain/problem:** 2026, precise sport events; preserve global context while using high-resolution local evidence.
- **Links/type:** repo https://github.com/arturxe2/AdaSpot ; CVPR 2026 paper linked there. Paper + repository.
- **Module/I/O:** video -> precise event timestamps plus task-aware local crops.
- **Architecture:** low-resolution global stream, unsupervised adaptive high-resolution ROI and temporal consistency; training required. Reported gains +3.96/+2.26 mAP@0 on Tennis/FineDiving.
- **Strength/limit:** directly addresses tiny hand/ball detail without full-frame high-res cost; very new and still requires labels. License UNKNOWN in verification; commercial risk UNKNOWN.
- **Project state:** newly studied; tested no; runnable likely after release. Basketball fine-tune yes. **Decision: research_challenger. EXACT VALUE:** learn when/where to crop the release hand/ball region while retaining whole-body phase context.

## R014 ActionFormer

- **Year/domain/problem:** 2022, generic temporal action localization; detect action segments in long untrimmed video.
- **Links/type:** paper https://arxiv.org/abs/2202.07925 ; repo https://github.com/happyharrycn/actionformer_release . Paper + repository.
- **Module/I/O:** precomputed video features -> action segments/classes.
- **Architecture:** anchor-free multiscale feature pyramid with local self-attention and boundary regression. Training required; public configs/weights.
- **Strength/limit:** mature long-video proposals; segment boundaries are not instantaneous release and precomputed features add complexity. License in repo must be checked; commercial risk UNKNOWN.
- **Project state:** studied yes; tested no; runnable. Basketball fine-tune yes. **Decision: benchmark_later for shot proposals. EXACT VALUE:** locate candidate shot windows in long videos, not determine strict release.

## R015 OpenTAD

- **Year/domain/problem:** 2025, generic TAL; unified reproducible temporal localization framework.
- **Links/type:** paper https://arxiv.org/abs/2502.20361 ; repo https://github.com/sming256/OpenTAD . Framework/repository.
- **Module/I/O:** video features -> segments from ActionFormer, TriDet, DyFADet and others.
- **Architecture:** common data/evaluation/model APIs; spatial features external, temporal heads modular. Training required; model zoo available.
- **Strength/limit:** fair controlled benchmarks; overkill before labels and not an exact-frame framework. Apache-2 code; model/data licenses separate; low-medium risk.
- **Project state:** newly studied; tested no; runnable likely. Basketball fine-tune yes. **Decision: benchmark_later. EXACT VALUE:** use only if comparing more than one shot-proposal/localization head on identical splits.

## R016 GolfDB / SwingNet

- **Year/domain/problem:** 2019, golf; sequence eight exact swing events in a trimmed action.
- **Links/type:** paper https://arxiv.org/abs/1903.06528 ; repo/dataset https://github.com/wmcnally/golfdb . Paper + repository/dataset.
- **Module/I/O:** cropped single-swing video -> eight ordered event frames.
- **Architecture:** MobileNetV2 frame encoder + bidirectional LSTM; spatial crop, temporal sequence decoder. Training required; pretrained SwingNet weight exists. Dataset: 1,400 videos; metric PCE.
- **Strength/limit:** closest simple ordered-event baseline; requires already trimmed/cropped single action and is old. Code/dataset CC BY-NC 4.0, commercial risk HIGH.
- **Project state:** already studied and locally explored in earlier work; not validated as current basketball model. Basketball fine-tune yes. **Decision: baseline_reference, superseded as final model. EXACT VALUE:** retain ordered-event decoding and PCE-style tolerance, not its weights/license-bound pipeline.

## R017 MS-TCN++

- **Year/domain/problem:** 2020, cooking/procedure; dense temporal action segmentation with refinement.
- **Links/type:** paper https://arxiv.org/abs/2006.09220 ; repo https://github.com/sj-li/MS-TCN2 . Paper + repository.
- **Module/I/O:** frame features -> per-frame phase labels.
- **Architecture:** multi-stage dilated temporal convolutions and dual dilation refinement. Training required; public code, dataset-specific weights vary. Metrics F1@IoU/edit/frame accuracy.
- **Strength/limit:** simple strong low-data baseline and smooth segments; can over-smooth short release events. License UNKNOWN in current check.
- **Project state:** already studied; tested only conceptually/experimental phase path, not accepted benchmark. Basketball fine-tune yes. **Decision: benchmark_now for phases. EXACT VALUE:** use as controlled phase baseline on fixed features before transformers.

## R018 ASFormer

- **Year/domain/problem:** 2021, procedure; reduce over-segmentation with local temporal attention.
- **Links/type:** paper https://arxiv.org/abs/2110.08568 ; repo https://github.com/ChinaYi/ASFormer . Paper + repository.
- **Module/I/O:** frame features -> dense action phases.
- **Architecture:** encoder-decoder transformer with local/windowed attention and refinement losses. Training required.
- **Strength/limit:** longer context than TCN and good standard metrics; quadratic/implementation cost and still needs dense labels. License UNKNOWN.
- **Project state:** already studied; tested no accepted result. Basketball fine-tune yes. **Decision: benchmark_later, immediate challenger to MS-TCN++. EXACT VALUE:** test whether attention improves variable-duration dip/jump while maintaining phase order.

## R019 TeCNO

- **Year/domain/problem:** 2020, surgery; online causal phase recognition in long procedures.
- **Links/type:** paper https://arxiv.org/abs/2003.10751 ; repo https://github.com/tobiascz/TeCNO . Paper + repository.
- **Module/I/O:** frame CNN features -> causal phase labels.
- **Architecture:** multi-stage causal dilated temporal convolutions; hierarchical prediction refinement. Training required; Cholec80 benchmark.
- **Strength/limit:** explicit online causality and stable workflow phases; surgical phases are minutes long and ordered more rigidly. License UNKNOWN; dataset restricted.
- **Project state:** newly studied; tested no. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** use prefix-invariance tests if future live coaching is added; offline V1 need not pay the causal accuracy cost.

## R020 Trans-SVNet

- **Year/domain/problem:** 2021/2022, surgery; combine local spatial evidence with long temporal workflow context.
- **Links/type:** paper https://arxiv.org/abs/2103.09712 ; repo https://github.com/YuemingJin/Trans-SVNet_Journal . Paper + repository.
- **Module/I/O:** surgical frames/features -> current phase and anticipation.
- **Architecture:** ResNet embeddings, TCN and transformer hybrid aggregation. Training required; Dropbox weights reported; Cholec80/M2CAI metrics.
- **Strength/limit:** clear spatial-temporal separation; old dependencies and domain phases are slow. License UNKNOWN.
- **Project state:** newly studied; tested no; runnable with effort. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** cache spatial evidence once and compare temporal heads fairly instead of rerunning pose/detector for every model.

## R021 EAST

- **Year/domain/problem:** 2025, generic/industrial action segmentation; end-to-end segmentation without frozen features.
- **Links/type:** paper https://openaccess.thecvf.com/content/ICCV2025W/SVU/papers/Wang_End-to-End_Action_Segmentation_Transformer_ICCVW_2025_paper.pdf . Paper.
- **Module/I/O:** video -> per-frame action segments.
- **Architecture:** fine-tuned ViT-G plus segmentation transformer; end-to-end spatial-temporal learning. Training is heavy; weights/code status UNKNOWN. Reported Assembly101 F1@10/25/50 42.3/39.4/32.8.
- **Strength/limit:** newer performance lineage beyond ASFormer; too costly and data-hungry for current labels. License/runnability UNKNOWN.
- **Project state:** newly studied; tested no. Basketball fine-tune yes. **Decision: research_challenger. EXACT VALUE:** evidence that frozen generic features may cap phase quality; revisit only after a trusted phase dataset exists.

## R022 SoccerNet Ball Action Spotting

- **Year/domain/problem:** 2023-2026, soccer; densely localize ball-state-changing events in long broadcasts.
- **Links/type:** repo/benchmark https://github.com/SoccerNet/sn-spotting ; site https://www.soccer-net.org/tasks . Dataset/benchmark.
- **Module/I/O:** full matches -> event timestamps/classes, now including player-centric variants.
- **Architecture:** multiple benchmark baselines; temporal spotting over video features. Training required. Metrics average mAP over temporal tolerances; only seven annotated games for 2024 ball task.
- **Strength/limit:** strong low-label/dense-event analogy; tolerance is seconds and broadcast scale differs. Code terms visible per repo, video needs NDA, dataset commercial terms uncertain.
- **Project state:** newly studied; tested no; full download prohibited. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** copy low-label transfer-learning and event-evaluation protocols, not soccer weights.

## R023 SoccerNet 2023 Ball Action Spotting winner

- **Year/domain/problem:** 2023, soccer; improve dense ball-event spotting with long context.
- **Links/type:** repo https://github.com/lRomul/ball-action-spotting . Repository/competition solution.
- **Module/I/O:** long match video -> pass/drive timestamps.
- **Architecture:** early 2D to later 3D slow fusion, transfer learning, then long-sequence fine-tuning. Training heavy; pretrained results available. Reported mAP@1 improvement roughly 65% to 80% from fusion choice.
- **Strength/limit:** concrete multi-stage low-label recipe; 1-second events, RTX3090/Docker and huge source data are mismatched. MIT code; dataset NDA/terms separate.
- **Project state:** newly studied; tested no. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** pretrain temporal representation on broader shot clips, then fine-tune strict release with long local windows once labels exist.

## R024 Timestamp-supervised temporal segmentation

- **Year/domain/problem:** 2021-2023, generic procedures; train dense phases from sparse timestamps.
- **Links/type:** internal prior research records the relevant timestamp-supervised TCN literature; exact authoritative URL is **unable_to_verify from current repository history**.
- **Module/I/O:** sparse phase timestamps -> dense per-frame phases.
- **Architecture:** pseudo-label propagation and temporal consistency over frame features. Training required; benchmarked on Breakfast/50Salads/GTEA in the literature.
- **Strength/limit:** reduces annotation burden; pseudo-boundaries can amplify annotation/model bias. License UNKNOWN.
- **Project state:** already studied; no accepted project experiment confirmed. Basketball fine-tune yes. **Decision: benchmark_later. EXACT VALUE:** use sparse key events only if 5-phase dense labeling cost proves prohibitive, with a small dense validation set.

## R025 Fine-grained shuttle hitting event detection

- **Year/domain/problem:** 2023, badminton; spot racket-shuttle contact around a fast, blurred object.
- **Links/type:** paper https://arxiv.org/abs/2306.10293 . Paper.
- **Module/I/O:** badminton video -> hitting event frames.
- **Architecture:** SwingNet-style temporal sequencing over visual motion; temporal context disambiguates tiny contact. Training required; public code/weights not confirmed.
- **Strength/limit:** close physical transition to ball release; racket contact differs from hand separation. License/runnability UNKNOWN.
- **Project state:** newly studied; tested no. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** treat release as a local temporal event around object/effector interaction, not a pose snapshot.

# C. Body, whole-body and hand perception

## R026 MMPose

- **Year/domain/problem:** active, general pose; unified 2D/3D body, whole-body, hand, face and mesh toolkit.
- **Links/type:** repo https://github.com/open-mmlab/mmpose . Repository/model zoo.
- **Module/I/O:** image/video person crops -> keypoints/pose structures.
- **Architecture:** framework hosting top-down/bottom-up models including RTMPose/RTMW/ViTPose; temporal processing external. Training optional; many pretrained weights and standard COCO/WholeBody benchmarks.
- **Strength/limit:** mature common interface and whole-body coverage; dependency stack and model/data licenses vary. Apache-2 code; weights/datasets separate.
- **Project state:** studied yes; tested in broader project research, not selected in production. Runnable yes. Basketball fine-tune usually no for body, possibly yes for occluded hands. **Decision: adopt_candidate benchmark harness. EXACT VALUE:** benchmark whole-body pose without building a custom inference framework.

## R027 RTMPose

- **Year/domain/problem:** 2023, general pose; real-time accurate 2D pose.
- **Links/type:** paper https://arxiv.org/abs/2303.07399 ; implementation in MMPose.
- **Module/I/O:** person crop -> 2D keypoints/confidence.
- **Architecture:** CSPNeXt-style backbone and SimCC coordinate classification; framewise spatial model. Training optional; pretrained weights available.
- **Strength/limit:** strong speed/accuracy and deployment; standard body joints do not resolve fingers. Apache-2 framework, checkpoint terms follow source datasets.
- **Project state:** newly compared; tested no project benchmark. Runnable yes. Basketball fine-tune likely no initially. **Decision: benchmark_now. EXACT VALUE:** immediate challenger to YOLO11 pose for more stable body timing on the same trusted clips.

## R028 RTMW

- **Year/domain/problem:** 2024, whole-body pose; handle multi-scale body, foot, face and hand landmarks.
- **Links/type:** paper https://arxiv.org/abs/2407.08634 ; implementation in MMPose.
- **Module/I/O:** person crop -> whole-body 2D keypoints.
- **Architecture:** feature pyramid and hierarchical coordinate encoding for differently scaled parts. Framewise; temporal smoothing external. Pretrained weights available.
- **Strength/limit:** one model aligns wrist/hand/body evidence; hands may still be too small/occluded at 480p. Framework Apache-2; weights/data terms separate.
- **Project state:** newly studied; tested no. Runnable likely. Basketball fine-tune optional. **Decision: benchmark_now. EXACT VALUE:** selected pose challenger for release-hand ROI and feet/body phases before adding a separate hand model.

## R029 RTMW3D

- **Year/domain/problem:** 2024+, general whole-body 3D; predict body and hands in camera-relative 3D.
- **Links/type:** MMPose model family https://github.com/open-mmlab/mmpose ; paper lineage RTMW https://arxiv.org/abs/2407.08634 . Model.
- **Module/I/O:** monocular person crop -> whole-body 3D keypoints.
- **Architecture:** RTMW-style hierarchical representation with depth prediction. Training optional with public weights; benchmark details in model cards.
- **Strength/limit:** convenient 3D hypothesis; scale/depth ambiguity and sports validation are unresolved. License/data terms separate.
- **Project state:** newly studied; tested no. Runnable likely. Basketball fine-tune maybe. **Decision: research_only benchmark. EXACT VALUE:** compare qualitative joint coordination, never emit metric biomechanics until validated against a reference.

## R030 ViTPose / ViTPose++

- **Year/domain/problem:** 2022/2023, general pose; high-capacity transformer pose estimation and multi-dataset generalization.
- **Links/type:** repo https://github.com/ViTAE-Transformer/ViTPose . Paper + repository/model zoo.
- **Module/I/O:** person crop -> body/whole-body keypoints.
- **Architecture:** plain ViT backbone with pose head; ViTPose++ extends multi-dataset knowledge. Framewise, heavy. Training optional; weights available.
- **Strength/limit:** accuracy challenger; latency and whole-body hand scale can be costly. Apache-2 code shown by repo; checkpoint data terms separate.
- **Project state:** studied yes; tested no current benchmark. Runnable with compatible environment. Basketball fine-tune not first step. **Decision: benchmark_later. EXACT VALUE:** research accuracy ceiling if RTMW misses joints; not the default V1 runtime.

## R031 DWPose

- **Year/domain/problem:** 2023, general whole-body; distill two-stage whole-body pose.
- **Links/type:** repo https://github.com/IDEA-Research/DWPose . Paper + repository.
- **Module/I/O:** person image -> body/face/hand keypoints.
- **Architecture:** teacher-student distillation combining body and hand/face stages. Framewise. Weights available.
- **Strength/limit:** efficient whole-body; old OpenMMLab dependencies and generative-control ecosystem focus. License must be checked; model/data terms separate.
- **Project state:** newly studied; tested no. Runnable with environment work. Basketball fine-tune optional. **Decision: benchmark_later. EXACT VALUE:** alternative if RTMW hand landmarks are insufficient, but no custom integration before benchmark evidence.

## R032 MediaPipe Hands

- **Year/domain/problem:** active, hand tracking; real-time 21-landmark hand pose from RGB.
- **Links/type:** official docs https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker . SDK/model.
- **Module/I/O:** image/video -> 21 normalized and approximate world hand landmarks, handedness and confidence.
- **Architecture:** palm detector plus hand landmark model with temporal tracking in video mode. Pretrained; no training required.
- **Strength/limit:** easy CPU/mobile deployment; fails when hand is tiny, ball-occluded or outside hand assumptions; handedness assumes input convention. Apache-style MediaPipe code, model terms should be checked.
- **Project state:** studied yes; not confirmed tested on release crops. Runnable yes. Basketball fine-tune unavailable. **Decision: benchmark_now on hand ROIs. EXACT VALUE:** low-cost test of whether finger/wrist evidence is recoverable at actual project resolution before adopting HaMeR.

## R033 HaMeR

- **Year/domain/problem:** 2024, monocular hand; reconstruct detailed 3D hand mesh under varied poses.
- **Links/type:** repo https://github.com/geopavlakos/hamer . Paper + repository/model.
- **Module/I/O:** hand crop -> MANO mesh, camera and 3D hand pose.
- **Architecture:** transformer-based hand reconstruction trained at scale; temporal smoothing external. Pretrained weights available; MANO model access required.
- **Strength/limit:** richer contact geometry; tiny/occluded hand crop and MANO licensing make it heavy. Code license and MANO/model terms differ; commercial risk MEDIUM/HIGH.
- **Project state:** newly studied; tested no. Runnable after gated assets. Basketball fine-tune potentially needed. **Decision: research_challenger. EXACT VALUE:** only benchmark on high-resolution release ROIs if 2D hand landmarks cannot separate contact states.

## R034 4DHumans / HMR 2.0

- **Year/domain/problem:** 2023, monocular humans; detect, reconstruct and track 3D human meshes in video.
- **Links/type:** repo https://github.com/shubham-goel/4D-Humans . Paper + repository/model.
- **Module/I/O:** monocular video -> per-person SMPL mesh tracks.
- **Architecture:** ViT-based HMR2.0 plus tracking; temporal association over spatial mesh predictions. Pretrained weights available.
- **Strength/limit:** robust whole-body shape/motion hypothesis; metric depth, hands and fast sports validity are limited. MIT code, SMPL assets separate/nontrivial; commercial risk MEDIUM.
- **Project state:** studied yes; tested no. Runnable after SMPL access. Basketball fine-tune maybe. **Decision: benchmark_later. EXACT VALUE:** research comparison for body orientation/occlusion, not V1 joint-angle truth.

## R035 PoseFlow

- **Year/domain/problem:** 2018, general multi-person pose; temporally associate pose detections.
- **Links/type:** repo https://github.com/YuliangXiu/PoseFlow . Paper + repository.
- **Module/I/O:** per-frame poses -> pose tracks.
- **Architecture:** pose-flow graph association using pose similarity; no end-to-end training required for association.
- **Strength/limit:** explicit identity stabilization; old and can be replaced by detector/tracker integration. License UNKNOWN in current verification.
- **Project state:** already studied; project experiment status not confirmed. Runnable with older stack. Basketball fine-tune no. **Decision: superseded. EXACT VALUE:** retain identity-consistency metric, use current tracker/person selection rather than adopting old code.

# D. Object detection, segmentation and motion

## R036 YOLO11

- **Year/domain/problem:** 2024, general detection/pose; efficient closed-set perception. **Links/type:** docs https://docs.ultralytics.com/models/yolo11/ ; repository https://github.com/ultralytics/ultralytics . Model/repository.
- **I/O/method:** frames -> boxes/keypoints; one-stage spatial detector, temporal tracking external. Training optional; pretrained and project-fine-tuned weights exist. Benchmarks use COCO AP and latency.
- **Strength/limit/governance:** mature, fast and already integrated; tiny occluded ball recall and pose hand detail are weak. AGPL-3 or enterprise terms; commercial risk HIGH without licensing review.
- **Project:** already studied/tested yes; runnable yes; ball model needs basketball data. **Decision: current_baseline. EXACT VALUE:** preserve as baseline so all challengers prove release-window gain on identical trusted test clips.

## R037 RF-DETR

- **Year/domain/problem:** 2025/2026, general detection; real-time transformer detector with strong accuracy. **Links/type:** repo https://github.com/roboflow/rf-detr ; paper https://arxiv.org/abs/2511.09554 . Paper/model/repository.
- **I/O/method:** frames -> boxes/classes; DINOv2 features, DETR-style set prediction and NAS; no temporal model. Training optional; core checkpoints/export available. COCO AP/latency benchmarks.
- **Strength/limit/governance:** strong modern detector and ONNX/TensorRT path; small-ball benefit is unproven. Nano/Small/L core code/weights Apache-2 per current repo; XL/2XL/Plus have separate PML terms. Risk LOW for core, HIGH for restricted tiers.
- **Project:** already researched, not tested on trusted split; basketball fine-tune yes. **Decision: benchmark_now core only. EXACT VALUE:** best immediate detector challenger to YOLO11n once the independent test exists.

## R038 Grounding DINO

- **Year/domain/problem:** 2023, open-vocabulary detection; locate text-described objects. **Links/type:** repo https://github.com/IDEA-Research/GroundingDINO . Paper/repository/model.
- **I/O/method:** image + text -> open-set boxes/scores; language-image transformer, framewise. Pretrained weights available; no basketball training needed for probing.
- **Strength/limit/governance:** useful for candidate discovery/prelabels, not stable tiny-ball production detection. Open core and newer hosted/1.5 offerings have distinct terms; exact weight license must be checked, risk MEDIUM.
- **Project:** studied; no confirmed project run. **Decision: research_only. EXACT VALUE:** generate candidate basketball/rim boxes for human review, never auto-promote to trusted labels.

## R039 SAM 2 / SAM 2.1

- **Year/domain/problem:** 2024, generic segmentation; promptable image/video masks. **Links/type:** repo https://github.com/facebookresearch/sam2 . Paper/repository/model.
- **I/O/method:** points/boxes/masks + video -> temporally propagated masks; memory-attention video segmentation. Pretrained checkpoints; SA-V benchmark and segmentation metrics.
- **Strength/limit/governance:** continuous visible extent and temporal memory; not semantic detection and can drift through occlusion. Apache-2 code/checkpoints per repo; dataset terms separate.
- **Project:** studied, not tested. Fine-tune usually no for benchmark. **Decision: benchmark_later. EXACT VALUE:** compare ball-mask propagation after a trusted frame against point tracking only when the ball occupies enough pixels.

## R040 Grounded SAM 2

- **Year/domain/problem:** 2024, open-world video segmentation; combine text detection and mask propagation. **Links/type:** repo https://github.com/IDEA-Research/Grounded-SAM-2 . Repository/pipeline.
- **I/O/method:** text + video -> detected object masks/tracks; Grounding DINO anchors SAM2. Training not required for demo; component weights available.
- **Strength/limit/governance:** efficient research prelabel pipeline; compounds detector/mask errors and component licenses. Repo license/components must each be checked; risk MEDIUM/HIGH.
- **Project:** studied, not tested. **Decision: research_only. EXACT VALUE:** candidate mask/contact visualization, not runtime evidence or trusted annotation without review.

## R041 SAHI

- **Year/domain/problem:** 2022, general small objects; improve detection through sliced inference. **Links/type:** paper https://arxiv.org/abs/2202.06934 ; repo https://github.com/obss/sahi . Paper/repository.
- **I/O/method:** high-resolution frame + detector -> merged boxes; overlapping spatial tiles, no temporal model. No retraining required; evaluates AP gains with supported detectors.
- **Strength/limit/governance:** cheapest small-ball experiment; tile boundaries, lost context and duplicate boxes can hurt. MIT license; model licenses separate, low code risk.
- **Project:** newly studied, not tested. Fine-tune no for inference comparison. **Decision: benchmark_now. EXACT VALUE:** compare native versus sliced YOLO/RF-DETR on release-window recall before collecting more data.

## R042 LocateAnything

- **Year/domain/problem:** 2025, open-vocabulary localization; text-prompt object boxes. **Links/type:** project/license recorded in existing project docs; local deployment under `E:/BasketballShotAI/tools/locateanything_local`; upstream exact paper URL not reverified in this sprint.
- **I/O/method:** image + prompt -> boxes; visual-language localization, framewise. Local Q8 model tested; generic benchmark per upstream.
- **Strength/limit/governance:** found approximate basketball position but systematic right/down box bias and hand occlusion required review. Research-only/unknown commercial chain; project policy `commercial_use=no`.
- **Project:** actually tested on selected video frames and 2,570-frame workflow; basketball fine-tune not performed. **Decision: research_only. EXACT VALUE:** keep human-reviewed prelabel acceleration; never use its labels directly in trusted V2.

## R043 Basketball51

- **Year/domain/problem:** basketball object detection dataset. **Links/type:** Roboflow Universe URL recorded in existing handoff; exact current URL unable_to_verify in this sprint. Dataset.
- **I/O/method:** basketball images -> boxes/classes; used to train closed-set detectors. Metrics AP/recall depend on split.
- **Strength/limit/governance:** basketball-specific scale; provenance/license/split leakage unresolved. Commercial risk HIGH/UNKNOWN.
- **Project:** studied and used in V1 data lineage; not independent product evidence. **Decision: quarantine/candidate_external. EXACT VALUE:** source audit only; do not mix into V2 trusted test.

## R044 Roboflow Sports datasets

- **Year/domain/problem:** multi-sport candidate data discovery. **Links/type:** site https://universe.roboflow.com/ . Dataset portal.
- **I/O/method:** hosted images/labels -> downloadable dataset variants; no single architecture. Training depends on selected dataset.
- **Strength/limit/governance:** broad candidate pool; every project has separate provenance, license, classes and quality. Commercial suitability cannot be assumed.
- **Project:** studied; individual candidates partially inspected, not trusted. **Decision: research_only candidate pool. EXACT VALUE:** populate manifest candidates only after per-dataset license check, preview and source-level review.

## R045 ByteTrack

- **Year/domain/problem:** 2021, MOT; retain low-score detections in association. **Links/type:** repo https://github.com/ifzhang/ByteTrack ; paper https://arxiv.org/abs/2110.06864 . Paper/repository.
- **I/O/method:** frame detections -> object tracks; two-stage score association, temporal Kalman/matching. Pretrained detector external; MOT metrics HOTA/IDF1/MOTA.
- **Strength/limit/governance:** mature for people; a tiny ball is a single fast object outside standard motion assumptions. MIT code, detector terms separate.
- **Project:** already studied; tracking path discussed/partially integrated for people, no trusted ball result. **Decision: adopted for person candidates, not ball winner. EXACT VALUE:** stabilize shooter identity while point tracking handles the ball.

## R046 BoT-SORT

- **Year/domain/problem:** 2022, MOT; combine motion, appearance and camera-motion compensation. **Links/type:** repo https://github.com/NirAharon/BoT-SORT ; paper https://arxiv.org/abs/2206.14651 . Paper/repository.
- **I/O/method:** detections + frames -> tracks; Kalman, ReID and global motion compensation. Training optional for ReID.
- **Strength/limit/governance:** robust moving-camera person identity; unnecessary for a single tiny ball and fixed-camera clips. MIT license; ReID weights/data separate.
- **Project:** already studied; no accepted project benchmark. **Decision: benchmark_later for shooter tracking. EXACT VALUE:** use only when multiple people/camera motion break ByteTrack, not as a universal tracker.

## R047 SportsMOT

- **Year/domain/problem:** 2023, basketball/football/volleyball; multi-person tracking under sports motion. **Links/type:** repo/dataset https://github.com/MCG-NJU/SportsMOT . Paper/dataset/benchmark.
- **I/O/method:** broadcast sports video -> player tracks; benchmark is tracker-agnostic. 240 clips; metrics HOTA/IDF1/MOTA.
- **Strength/limit/governance:** sports acceleration/crowding; does not track balls or close-form poses. Dataset/code licenses must be checked.
- **Project:** newly studied; not tested/downloaded. **Decision: benchmark_later. EXACT VALUE:** source of hard shooter-selection scenarios, not strict-release training.

## R048 CoTracker3

- **Year/domain/problem:** 2024/2025, generic point tracking; long-range joint point trajectories with visibility. **Links/type:** repo https://github.com/facebookresearch/co-tracker ; paper https://arxiv.org/abs/2410.11831 . Paper/repository/model.
- **I/O/method:** video + query point/grid -> tracks, visibility and confidence; jointly updates points, offline/online temporal windows. Pretrained weights available; TAP-Vid metrics.
- **Strength/limit/governance:** strong occlusion-aware continuity and practical API; point can drift from rotating blurred ball. Apache-style Meta research repo terms should be rechecked for weights.
- **Project:** newly studied, not tested. No basketball fine-tune for first probe. **Decision: benchmark_now. EXACT VALUE:** initialize at a high-confidence ball center and bridge 10-30 release frames through detector gaps.

## R049 TAP-Vid

- **Year/domain/problem:** 2022, generic point tracking; standardized any-point tracking evaluation. **Links/type:** site https://tapvid.github.io/ ; repo https://github.com/google-deepmind/tapnet . Dataset/benchmark.
- **I/O/method:** video/query points -> position and occlusion; metrics average Jaccard, position accuracy and occlusion accuracy.
- **Strength/limit/governance:** correct tracking metrics and diverse motion; benchmark points are not sports balls. Dataset/model licenses in repository must be checked.
- **Project:** studied, not tested. **Decision: adopted metric reference. EXACT VALUE:** evaluate ball-center tracker accuracy and visibility separately instead of only track existence.

## R050 TAPIR

- **Year/domain/problem:** 2023, generic point tracking; accurate long-range tracking and occlusion. **Links/type:** repo https://github.com/google-deepmind/tapnet ; project https://deepmind-tapir.github.io/ . Paper/model.
- **I/O/method:** query point + video -> coordinates/occlusion; global matching then local iterative refinement. Pretrained weights; TAP-Vid benchmark.
- **Strength/limit/governance:** robust baseline and mature lineage; slower/older than TAPNext family. Code/model license to verify.
- **Project:** newly studied, not tested. **Decision: superseded benchmark reference. EXACT VALUE:** fallback comparator only if CoTracker3/TAPNext behavior is unclear.

## R051 BootsTAPIR

- **Year/domain/problem:** 2024, point tracking; improve TAPIR using pseudo-labelled real video. **Links/type:** implementation in https://github.com/google-deepmind/tapnet . Paper/model.
- **I/O/method:** video/query -> tracks; bootstrapped pseudo-label training improves domain robustness. Pretrained checkpoints; TAP-Vid metrics.
- **Strength/limit/governance:** closes synthetic-to-real gap; superseded by newer architecture for selection. License per tapnet repo, not reverified.
- **Project:** studied, not tested. **Decision: superseded/idea_reference. EXACT VALUE:** motivates using high-confidence detector tracks as pseudo-labels, but only inside quarantine data.

## R052 TAPNext

- **Year/domain/problem:** 2025, point tracking; simplify tracking as next-token prediction. **Links/type:** repo https://github.com/google-deepmind/tapnet ; project/paper linked there. Paper/model.
- **I/O/method:** video/query -> sequential track tokens and visibility; autoregressive temporal propagation. Pretrained weights expected in official lineage.
- **Strength/limit/governance:** faster/simpler successor to TAPIR; autoregressive errors can accumulate. License UNKNOWN in current verification.
- **Project:** newly studied, not tested. **Decision: benchmark_later. EXACT VALUE:** research comparison when CoTracker3 fails on long clips, not first integration.

## R053 TAPNext++

- **Year/domain/problem:** 2026, point tracking; stable very-long tracking with occlusion/re-detection. **Links/type:** paper https://arxiv.org/abs/2604.10582 ; project https://tap-next-plus-plus.github.io/ ; official lineage repo https://github.com/google-deepmind/tapnet . Paper/model.
- **I/O/method:** long video/query -> coordinates/visibility; next-token tracking trained on sequences up to 1,024 frames, long-memory/re-detection. Authors report up to 40x longer stability.
- **Strength/limit/governance:** best research challenger for long occlusion; very new, heavier and exact release assets/license must be verified.
- **Project:** newly studied, not tested. **Decision: research_challenger. EXACT VALUE:** future long-video ball continuity after CoTracker3 establishes whether point tracking is valuable at all.

## R054 TrackNet

- **Year/domain/problem:** 2019, tennis/badminton; detect tiny blurred sports balls from consecutive frames. **Links/type:** paper https://arxiv.org/abs/1907.03698 . Paper/model family.
- **I/O/method:** stacked recent frames -> ball heatmap; convolutional spatial heatmap with short temporal input. Training required; sport-specific datasets.
- **Strength/limit/governance:** seminal motion-aware tiny-object formulation; old and superseded. Code/data license varies/UNKNOWN.
- **Project:** newly studied, not tested. Basketball fine-tune yes. **Decision: idea_reference/superseded. EXACT VALUE:** predict a center heatmap from multiple frames rather than independent boxes when blur dominates.

## R055 TrackNetV3

- **Year/domain/problem:** 2023, badminton; shuttle tracking plus trajectory repair. **Links/type:** repo https://github.com/qaz812345/TrackNetV3 . Paper/repository/model.
- **I/O/method:** frame sequence + estimated background -> shuttle heatmaps/trajectory; temporal mixup, inpainting and rectification. Training required; dataset/checkpoint assets available per repo.
- **Strength/limit/governance:** explicit occlusion recovery and reported 87.72 to 97.51 accuracy improvement; shuttle physics/scale differ. License file exists but exact terms not reverified; dataset risk UNKNOWN.
- **Project:** newly studied, not tested. Basketball fine-tune yes. **Decision: benchmark_later/idea_reference. EXACT VALUE:** borrow visibility-aware gap repair, capped by maximum unsupported gap.

## R056 TrackNetV4

- **Year/domain/problem:** 2024, tennis/badminton; successor fast-ball tracking with motion attention. **Links/type:** repo https://github.com/TrackNetV4/TrackNetV4 ; paper https://arxiv.org/abs/2409.14543 . Paper/repository/model.
- **I/O/method:** multiple frames -> ball/shuttle location heatmaps; motion-attention maps emphasize moving tiny objects. Training required; sport checkpoints may exist.
- **Strength/limit/governance:** current TrackNet-family candidate; still needs basketball-specific labels and may confuse moving hands. License/data terms UNKNOWN.
- **Project:** newly studied, not tested. **Decision: research_challenger. EXACT VALUE:** if detector+point tracking fails, train a basketball multi-frame heatmap model modeled on V4 rather than another per-frame box detector.

## R057 TTNet

- **Year/domain/problem:** 2020, table tennis; joint real-time ball localization, table segmentation and hit/bounce event spotting. **Links/type:** paper https://arxiv.org/abs/2004.09927 ; repo https://github.com/net5/ttnet-realtime-for-table-tennis-pytorch . Paper/repository/dataset link.
- **I/O/method:** high-FPS video -> ball coordinates, segmentation and events; shared spatiotemporal multi-task network. Training required; OpenTTGames benchmark.
- **Strength/limit/governance:** demonstrates perception+event joint learning at 120 fps; table geometry and hit events differ. License UNKNOWN.
- **Project:** newly studied, not tested. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** future joint ball/contact-event head can share features, but V1 keeps modules separable for diagnosis.

## R058 BlurBall

- **Year/domain/problem:** 2025, table tennis; model severe motion blur and blur orientation for tiny balls. **Links/type:** repo/dataset https://github.com/cogsys-tuebingen/blurball . Paper/repository/dataset.
- **I/O/method:** multi-frame crops -> ball center and blur attributes; HRNet-style heatmaps with temporal attention. Training required; 64k annotated frames and baselines.
- **Strength/limit/governance:** rare explicit blur labels; table-tennis speed/domain and dataset terms require care. License UNKNOWN in current verification.
- **Project:** newly studied, not downloaded/tested. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** add blur length/orientation and visibility to future ball manifest instead of treating every box equally.

## R059 TOTNet

- **Year/domain/problem:** 2024/2025, sports ball tracking; combine temporal and spatial cues. **Links/type:** repo https://github.com/AugustRushG/TOTNet . Paper/repository.
- **I/O/method:** frame sequences -> ball trajectory; dedicated temporal/spatial tracking network. Training/checkpoints per repo; benchmarks include sports-ball sets.
- **Strength/limit/governance:** potentially useful dedicated tracker; maturity, license and independent evidence are less clear than TrackNetV4. Risk UNKNOWN.
- **Project:** newly discovered; not tested. Basketball fine-tune yes. **Decision: benchmark_later only if maintained. EXACT VALUE:** no immediate integration; retain as alternative fast-object architecture.

## R060 Where Is The Ball

- **Year/domain/problem:** 2025, sports geometry; recover 3D ball trajectory from monocular 2D tracking. **Links/type:** CVSports paper; authoritative URL not recovered in current local records, metadata verified from workshop search. Paper.
- **I/O/method:** calibrated video + 2D ball track -> physically constrained 3D trajectory. Spatial camera geometry plus temporal physics; no generic pretrained output.
- **Strength/limit/governance:** useful after reliable tracks/calibration; impossible to trust from uncalibrated close phone video. License/code UNKNOWN.
- **Project:** newly studied; not tested. Basketball calibration/fitting needed. **Decision: research_only. EXACT VALUE:** define a future calibrated trajectory layer, not a V1 release-velocity claim.

## R061 Supervision

- **Year/domain/problem:** active, CV tooling; annotation QA, visualization and dataset utilities. **Links/type:** repo https://github.com/roboflow/supervision . Repository/library.
- **I/O/method:** images/video + detections/labels -> overlays, slices, contact sheets and dataset operations; no learned model.
- **Strength/limit/governance:** mature low-cost QA; cannot replace human review or validate provenance. MIT code; model/data terms external.
- **Project:** already adopted in decision docs; custom workflow built outside repo, no production dependency confirmed. **Decision: adopted. EXACT VALUE:** visualize every auto label and produce review contact sheets before manifest promotion.

## R062 TAP-Vid DAVIS/Kinetics/RobotAP subsets

- **Year/domain/problem:** 2022+, generic/robotics; diverse point tracking benchmark slices. **Links/type:** site https://tapvid.github.io/ . Dataset/benchmark.
- **I/O/method:** annotated query points -> tracks/occlusion; evaluation only. Metrics AJ, position and occlusion accuracy.
- **Strength/limit/governance:** separates generalization conditions; no tiny sports-ball slice. Dataset licenses differ by source.
- **Project:** not downloaded/tested. **Decision: benchmark_reference. EXACT VALUE:** mirror its visibility-aware annotation schema in a small basketball point-track test set.

## R063 SA-V

- **Year/domain/problem:** 2024, open-world video segmentation; train/evaluate promptable mask propagation. **Links/type:** distributed with SAM2 at https://github.com/facebookresearch/sam2 . Dataset/benchmark.
- **I/O/method:** videos + masks -> segmentation training/evaluation; diverse spatial objects over time.
- **Strength/limit/governance:** large mask benchmark; not sports contact and too large for current needs. Dataset terms per Meta release.
- **Project:** not downloaded/tested. **Decision: reject for current data acquisition. EXACT VALUE:** no direct project action; use pretrained SAM2 only if selected.

## R064 AI Basketball Shot Detection Tracker

- **Year/domain/problem:** basketball prototype; detect/track shots and outcomes. **Links/type:** exact repository URL is recorded in `HANDOFF_RESEARCH.md`; not re-listed because no new successor evidence was found.
- **I/O/method:** basketball video -> person/ball/rim tracks and shot logic; detector/tracker plus heuristics.
- **Strength/limit/governance:** useful end-to-end decomposition; prototype data/heuristics and license uncertainty prevent adoption.
- **Project:** already studied, not adopted as core. **Decision: idea_reference. EXACT VALUE:** retain state-machine decomposition, replace its sport-specific confidence claims with independently evaluated modules.

## R065 Basketball-Shot-Detection

- **Year/domain/problem:** basketball shot event prototype. **Links/type:** https://github.com/browlm13/Basketball-Shot-Detection . Repository.
- **I/O/method:** video -> ball trajectory/shot classification using detection and geometric logic; temporal heuristics over spatial detections.
- **Strength/limit/governance:** concrete trajectory/state ideas; older dependencies, narrow scenes and no trusted generalization proof. License must be checked.
- **Project:** actually cloned/read/ran partially in earlier work; no accepted product result. **Decision: idea_reference. EXACT VALUE:** use its geometric state transitions as testable hypotheses, not copy the stack or claim accuracy.

# E. Human-object contact and interaction

## R066 100 Days of Hands (100DOH)

- **Year/domain/problem:** 2020, internet hand-object interaction; detect hands, contact state and contacted object. **Links/type:** paper https://arxiv.org/abs/2006.06669 ; project https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/ . Paper/dataset/model.
- **I/O/method:** image -> hand box, side, contact state and object-in-contact box; contact-aware spatial detector, no transition model. Training required; 100k annotated frames.
- **Strength/limit/governance:** foundational contact labels at scale; mostly household interactions and static frames, not tiny ball separation. Dataset/model terms require confirmation; risk UNKNOWN.
- **Project:** newly studied; not tested/downloaded. Basketball fine-tune yes. **Decision: adopt_candidate schema / benchmark_later model. EXACT VALUE:** define release evidence as `contact -> separating -> persistent no_contact`, not wrist pose alone.

## R067 Hands23

- **Year/domain/problem:** 2023, hand-object interaction; extend hand, active object and second-order object relations. **Links/type:** official project from 100DOH lineage; exact URL/license not reliably recovered, marked UNKNOWN.
- **I/O/method:** image -> hand grasp/contact and first/second interacting object relations; spatial HOI annotations.
- **Strength/limit/governance:** richer successor annotations; static household domain and uncertain release assets. Commercial risk UNKNOWN.
- **Project:** newly studied; not tested. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** possession should associate shooter hand, basketball and possibly rim context as distinct relations.

## R068 ContactHands

- **Year/domain/problem:** hand contact classification; distinguish no/self/person/object contact. **Links/type:** official dataset page identified during research; exact stable URL/license UNKNOWN.
- **I/O/method:** hand image/box -> four contact states; framewise classifier.
- **Strength/limit/governance:** simple contact taxonomy; no identity of basketball and no temporal release transition. Dataset terms UNKNOWN.
- **Project:** not tested. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** include `unknown` and non-ball contact states to prevent any hand overlap being interpreted as ball possession.

## R069 EgoHOS

- **Year/domain/problem:** 2022, egocentric HOI segmentation; segment hands and first/second-order interacting objects. **Links/type:** repo https://github.com/owenzlz/EgoHOS . Paper/repository/model.
- **I/O/method:** egocentric frame -> left/right hand and interacting-object masks; multi-stage spatial segmentation. Pretrained models available.
- **Strength/limit/governance:** explicit contact boundary/masks; egocentric household bias and old MMCV stack. MIT code; weight/data terms separate.
- **Project:** newly studied, not tested. Basketball fine-tune yes. **Decision: research_only. EXACT VALUE:** mask-overlap features may help contact on high-res crops, but model weights are not direct basketball evidence.

## R070 ContactPose

- **Year/domain/problem:** 2020, grasp/contact; capture 3D hand-object contact maps. **Links/type:** repo/dataset https://github.com/facebookresearch/ContactPose . Paper/repository/dataset.
- **I/O/method:** RGB-D/multiview grasp sequences -> 3D hand pose and object surface contact. Spatial 3D reconstruction; limited temporal transitions.
- **Strength/limit/governance:** detailed physical contact supervision; only 25 rigid objects/static grasps, unlike ballistic release. MIT code; object models/data have separate terms.
- **Project:** not tested/downloaded. Basketball fine-tune/new data essential. **Decision: idea_reference. EXACT VALUE:** contact is a surface relation with uncertainty, not equivalent to box intersection.

## R071 HOT3D

- **Year/domain/problem:** 2025, egocentric 3D HOI; benchmark hand/object tracking with calibrated devices. **Links/type:** site https://facebookresearch.github.io/hot3d/ . Dataset/benchmark.
- **I/O/method:** synchronized egocentric RGB/depth/pose -> 3D hands and rigid objects; multi-view/calibrated temporal tracking. 833 minutes, 3.7M images.
- **Strength/limit/governance:** strong 3D tracking ground truth; headset domain, rigid known objects and custom dataset license. Commercial risk HIGH/needs review.
- **Project:** not downloaded/tested. Basketball fine-tune and new capture needed. **Decision: research_reference. EXACT VALUE:** informs how to create a small synchronized validation capture if strict-release truth becomes critical.

## R072 EPIC-KITCHENS-100

- **Year/domain/problem:** 2021, egocentric activities; recognize fine hand-object actions in long video. **Links/type:** site https://epic-kitchens.github.io/2021 ; annotations repo https://github.com/epic-kitchens/epic-kitchens-100-annotations . Dataset/benchmark.
- **I/O/method:** first-person video -> verb/noun/action segments and object/hand annotations; temporal action recognition/localization.
- **Strength/limit/governance:** large procedural HOI and long-video splits; kitchen domain and egocentric viewpoint. Dataset is research-access controlled; commercial risk HIGH.
- **Project:** studied, not tested/downloaded. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** represent interaction as verb+noun+time (`hold ball`, `release ball`, `reacquire ball`) rather than one shot class.

## R073 VISOR

- **Year/domain/problem:** 2022, egocentric video object segmentation; dense hand/object masks and relations. **Links/type:** project https://epic-kitchens.github.io/VISOR/ . Dataset/benchmark.
- **I/O/method:** video -> temporally associated masks for hands and active objects. Spatial masks with temporal identity.
- **Strength/limit/governance:** high-quality HOI masks; egocentric kitchen scale and access terms. Commercial use unresolved/high risk.
- **Project:** not tested/downloaded. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** evaluate release-hand/ball mask identity over time if box/contact ambiguity remains.

## R074 Video2Knowledge contact transitions

- **Year/domain/problem:** 2026, manipulation understanding; infer grasp/release from fingertip geometry and object motion. **Links/type:** publication identified in research; authoritative open URL/code unavailable, license UNKNOWN.
- **I/O/method:** video hand/object tracks -> grasp/release transition; fingertip-to-palm distance trends, object motion and two-frame persistence.
- **Strength/limit/governance:** interpretable transition logic; likely household/manipulation assumptions and unverified code.
- **Project:** not tested. Basketball data needed. **Decision: idea_reference. EXACT VALUE:** combine hand aperture trend, ball motion and persistence as explicit release features.

## R075 Contact-state transition recognition in manipulation

- **Year/domain/problem:** 2007+, robotics; recognize manipulation stages from changing contact state. **Links/type:** paper lineage identified, exact canonical URL unable_to_verify; concept reference.
- **I/O/method:** contact/sensor state sequence -> manipulation event; finite-state temporal model, not RGB spatial perception.
- **Strength/limit/governance:** robust conceptual decomposition; sensor contact is unavailable in ordinary video. License not applicable/UNKNOWN.
- **Project:** not tested. **Decision: adopted idea. EXACT VALUE:** strict release decoder should be a probabilistic state machine over evidence, with `insufficient_data` when transitions are unsupported.

# F. Markerless motion reconstruction and biomechanics tooling

## R076 Sports2D

- **Year/domain/problem:** 2023/2024, sports biomechanics; accessible 2D kinematics from ordinary video. **Links/type:** repo https://github.com/davidpagnon/Sports2D . JOSS paper/repository.
- **I/O/method:** single/multi-person video -> 2D keypoints, filtered coordinates, angles and optional OpenSim files; pose backend plus interpolation/filtering. No custom training required.
- **Strength/limit/governance:** practical and transparent; authors explicitly restrict validity to motion near the image plane and recommend multi-camera for research 3D. License per repo must be checked.
- **Project:** newly studied, not tested. Basketball fine-tune no initially. **Decision: benchmark_now as measurement reference. EXACT VALUE:** implement confidence-aware 2D metrics and copy its explicit planar-validity warnings.

## R077 Pose2Sim

- **Year/domain/problem:** active, sports biomechanics; markerless multi-camera 3D and OpenSim workflow. **Links/type:** repo https://github.com/perfanalytics/pose2sim . Paper/repository.
- **I/O/method:** synchronized calibrated multiview video -> triangulated/filter 3D keypoints and OpenSim kinematics; supports multiple pose backends.
- **Strength/limit/governance:** best open multi-view reference architecture; requires calibration/sync and multiple cameras, beyond normal user input. BSD-3 code; model/OpenSim/data terms separate.
- **Project:** studied, not tested. No basketball fine-tune for pipeline. **Decision: future_validation_stack. EXACT VALUE:** ground-truth-ish comparison route for a small lab validation, not V1 user runtime.

## R078 OpenCap

- **Year/domain/problem:** 2023, biomechanics; estimate kinematics/dynamics using synchronized smartphones. **Links/type:** paper https://doi.org/10.1371/journal.pcbi.1011462 ; repo https://github.com/opencap-org/opencap-core ; site https://www.opencap.ai/ . Paper/repository/service.
- **I/O/method:** two or more calibrated phone views -> 3D kinematics/dynamics through pose, triangulation and OpenSim. Pretrained components; no user training.
- **Strength/limit/governance:** validated accessible workflow with reported kinematic errors around 4.3-4.7 degrees on studied tasks; task/device validity does not transfer automatically to jump shots. Apache-2 core; cloud/service/model terms separate.
- **Project:** studied, not tested. **Decision: validation_reference. EXACT VALUE:** use a controlled two-phone subset to quantify error of V1 monocular metrics.

## R079 OpenCap Monocular

- **Year/domain/problem:** 2026, clinical movement; infer biomechanics from one handheld smartphone. **Links/type:** paper https://arxiv.org/abs/2603.24733 ; code lineage https://github.com/IntelligentSensingAndRehabilitation/MonocularBiomechanics . Paper/repository.
- **I/O/method:** monocular video -> whole-body kinematics/gait metrics; monocular fitting with learned body model and biomechanical constraints.
- **Strength/limit/governance:** clinically promising with reported ICC >0.9 in deployment metrics; validation tasks are walking/squat/sit-to-stand, not ballistic shooting. New code/license/model terms need review.
- **Project:** newly studied, not tested. **Decision: research_challenger. EXACT VALUE:** test only on controlled clips after 2D baseline; do not infer forces or universal jump-shot angles.

## R080 MotionBERT

- **Year/domain/problem:** 2023, human motion; learn temporal representations and lift 2D pose to 3D. **Links/type:** repo https://github.com/Walter0807/MotionBERT . Paper/repository/model.
- **I/O/method:** up to ~243-frame 17-joint 2D sequence -> 3D pose/action representation; dual-stream spatial-temporal transformer. Pretrained weights available.
- **Strength/limit/governance:** simple mature temporal lifting baseline; H36M skeleton/domain, no hands and camera-relative ambiguity. License/model/data terms require check.
- **Project:** studied, not tested. Basketball fine-tune maybe. **Decision: benchmark_later. EXACT VALUE:** establish whether 3D lifting adds repeatable coordination information beyond 2D normalized features.

## R081 WHAM

- **Year/domain/problem:** 2024, monocular moving-camera video; world-grounded human motion and trajectory. **Links/type:** project https://wham.is.tue.mpg.de/ . Paper/model.
- **I/O/method:** video + optional camera cues -> SMPL motion in world coordinates; combines image features, temporal motion and camera estimation.
- **Strength/limit/governance:** handles moving cameras; world scale/ground and fast sports remain uncertain. Research code/model/SMPL terms separate; risk MEDIUM.
- **Project:** newly studied, not tested. Basketball fine-tune maybe. **Decision: research_reference. EXACT VALUE:** only relevant when camera motion cannot be controlled; fixed-camera V1 should not pay this cost.

## R082 GVHMR

- **Year/domain/problem:** 2024, monocular video; globally consistent human motion in gravity-aligned coordinates. **Links/type:** repo https://github.com/zju3dv/GVHMR . Paper/repository/model.
- **I/O/method:** video -> global SMPL motion; gravity-view representation, temporal network and optional visual odometry.
- **Strength/limit/governance:** stronger successor candidate to WHAM/4DHumans for global motion; heavy, SMPL-gated and sports validity unproven. License/model terms need review.
- **Project:** newly studied, not tested. **Decision: research_challenger. EXACT VALUE:** future landing/body-translation research, not immediate V1 metric source.

## R083 HumanMM

- **Year/domain/problem:** 2025, multi-shot video; reconstruct human motion across camera cuts. **Links/type:** CVPR 2025 paper; official code URL not confirmed in this sprint.
- **I/O/method:** edited multi-shot video -> coherent human motion; temporal linking across cuts and body reconstruction.
- **Strength/limit/governance:** addresses discontinuous camera edits; project should reject edited clips for quantitative V1 instead. License/runnable UNKNOWN.
- **Project:** not tested. **Decision: reject for V1 / idea_reference. EXACT VALUE:** camera cuts trigger quality rejection rather than a complex reconstruction dependency.

## R084 BioPose

- **Year/domain/problem:** 2025, biomechanics; biomechanically constrained monocular 3D pose. **Links/type:** WACV 2025 paper; exact official repo/URL not confirmed, license UNKNOWN.
- **I/O/method:** monocular video/2D pose -> 3D body motion via neural inverse kinematics and optimization.
- **Strength/limit/governance:** explicitly targets biomechanical plausibility; runtime, sports validity and assets uncertain.
- **Project:** not tested. Basketball validation required. **Decision: research_challenger. EXACT VALUE:** candidate 3D ceiling after reference capture, not a current recommendation engine.

## R085 AthletePose3D

- **Year/domain/problem:** 2025, athletic motion; benchmark 3D pose on extreme sports actions. **Links/type:** paper https://arxiv.org/abs/2503.07499 . Dataset/benchmark.
- **I/O/method:** athletic images/video -> 3D poses; evaluates methods under fast/extreme motion.
- **Strength/limit/governance:** closer motion distribution than H36M; dataset availability/license and basketball coverage must be verified.
- **Project:** not downloaded/tested. **Decision: benchmark_reference. EXACT VALUE:** choose 3D models based on athletic, not indoor-lab, generalization evidence.

## R086 Markerless jump validation study

- **Year/domain/problem:** 2025, jump biomechanics; agreement of markerless versus marker-based joint kinematics. **Links/type:** primary validation study located during research; exact stable URL not preserved, metadata therefore `unable_to_verify`.
- **I/O/method:** jump trials -> repeatability and reference error; reports repeatability ICC about .95/RMSE 1.91 degrees and reference concordance about .51/RMSE 3.29 degrees after bias handling, with hip flexion weaker.
- **Strength/limit/governance:** directly warns repeatable is not equivalent to accurate; exact population/setup must be reverified before citation in product.
- **Project:** not tested. **Decision: evidence_boundary. EXACT VALUE:** report reliability and reference agreement separately for every metric.

## R087 Sports markerless motion-capture systematic review

- **Year/domain/problem:** 2026, sports biomechanics; synthesize validation of markerless systems. **Links/type:** preprint/review discovered in search; exact authoritative URL and peer-review status UNKNOWN.
- **I/O/method:** validation studies -> task/system error synthesis.
- **Strength/limit/governance:** broad capability boundary; heterogeneous protocols prevent one universal error number.
- **Project:** literature only. **Decision: idea_reference. EXACT VALUE:** require task-, view- and metric-specific validation before product recommendations.

# G. Action quality, causal interpretation and feedback

## R088 A Decade of Action Quality Assessment

- **Year/domain/problem:** 2025, AQA survey; systematic taxonomy of a decade of scoring/assessment. **Links/type:** paper https://arxiv.org/abs/2502.02817 ; repo https://github.com/HaoYin116/Survey_of_AQA . Survey/repository.
- **I/O/method:** >200 papers -> taxonomy of datasets, models, metrics and trends. No training.
- **Strength/limit/governance:** strongest map for lineage; survey rankings are not basketball validation. Repository license UNKNOWN.
- **Project:** newly studied, not tested. **Decision: adopted taxonomy. EXACT VALUE:** organize AQA as procedure, pose, uncertainty, causality and explanation rather than one regression head.

## R089 Comprehensive AQA Survey

- **Year/domain/problem:** 2024/2025, AQA; unified review and benchmark perspective. **Links/type:** paper https://arxiv.org/abs/2412.11149 . Survey.
- **I/O/method:** >150 studies -> task/dataset/method/metric synthesis.
- **Strength/limit/governance:** cross-checks the first survey; no direct code. Paper use low risk.
- **Project:** studied, not tested. **Decision: idea_reference. EXACT VALUE:** prevent cherry-picking one SOTA number across incompatible datasets.

## R090 FineDiving

- **Year/domain/problem:** 2022, diving; phase/procedure-aware quality scoring. **Links/type:** repo/dataset https://github.com/xujinglin/FineDiving . Paper/repository/dataset.
- **I/O/method:** 3,000 diving clips -> 29 sub-actions, step boundaries and judge scores; procedure segmentation plus score regression. Training required.
- **Strength/limit/governance:** canonical phase-aware AQA; judged diving scores and fixed taxonomy differ from shooting feedback. MIT code; dataset requires signed release agreement and separate terms.
- **Project:** already studied, not tested/downloaded. Basketball fine-tune yes. **Decision: adopt_candidate architecture. EXACT VALUE:** score/compare phase evidence only after boundaries are trusted; no direct weight transfer.

## R091 HP-MCoRe

- **Year/domain/problem:** 2025, diving AQA; fuse visual and skeleton procedure evidence. **Links/type:** repo https://github.com/Lumos0507/HP-MCoRe ; paper https://arxiv.org/abs/2501.03674 . Paper/repository.
- **I/O/method:** video + pose -> phase-aware quality score; dynamic visual-skeleton encoding, procedure segmentation, multimodal contrastive regression. Training required.
- **Strength/limit/governance:** strongest direct pose-guided candidate; pose errors and data demands are high. License UNKNOWN.
- **Project:** already studied, not tested. Basketball fine-tune yes. **Decision: research_challenger. EXACT VALUE:** model appearance and skeleton as complementary gated evidence rather than concatenating noisy pose blindly.

## R092 FineCausal

- **Year/domain/problem:** 2025, AQA; causal and interpretable fine-grained scoring. **Links/type:** repo https://github.com/Harrison21/FineCausal . Paper/repository.
- **I/O/method:** action video + expert causal graph -> quality score and causal evidence; graph attention/intervention and temporal causal attention. Training required.
- **Strength/limit/governance:** tests foreground cause versus background correlation; causal graph requires genuine expertise. MIT code; dataset terms separate.
- **Project:** studied, not tested. Basketball-specific graph essential. **Decision: benchmark_later. EXACT VALUE:** intervene on phase/body evidence and verify the score changes for the stated reason.

## R093 NS-AQA

- **Year/domain/problem:** 2024, fitness; neuro-symbolic comprehensive form reports. **Links/type:** repo https://github.com/laurenok24/NSAQA . Paper/repository/rule programs.
- **I/O/method:** exercise video/pose -> score, rule violations, evidence and report; learned parser plus expert-verified symbolic rules. Training required for parser; rules explicit.
- **Strength/limit/governance:** closest coaching architecture; fitness rules do not transfer and dataset is likely non-commercial. Code license/data terms need review.
- **Project:** studied, not tested. Basketball rules and validation required. **Decision: adopt_candidate architecture, not rules. EXACT VALUE:** V1 feedback should execute transparent basketball rules over measured evidence, then verbalize only passed rule outputs.

## R094 UIL-AQA

- **Year/domain/problem:** 2026, rhythmic gymnastics; uncertainty-aware clip-level interpretable long-form AQA. **Links/type:** project https://andrewjohngilbert.github.io/UILAQA/ . Paper/repository.
- **I/O/method:** long video -> difficulty/quality and clip-level uncertainty/evidence; temporal clip decomposition and distribution learning.
- **Strength/limit/governance:** explicit uncertainty and local explanations; judged long routines differ from short shots. License UNKNOWN.
- **Project:** newly studied, not tested. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** propagate per-phase uncertainty and abstain instead of averaging weak phases into a confident score.

## R095 CoFInAl

- **Year/domain/problem:** 2024, AQA; coarse-to-fine instruction alignment. **Links/type:** repo https://github.com/ZhouKanglei/CoFInAl_AQA ; paper https://arxiv.org/abs/2404.13999 . Paper/repository.
- **I/O/method:** video + action instruction -> quality score; coarse action and fine instruction alignment over temporal features. Training required.
- **Strength/limit/governance:** incorporates procedure semantics; text instructions can encode unsupported universal rules. License UNKNOWN.
- **Project:** newly studied, not tested. Basketball fine-tune yes. **Decision: benchmark_later. EXACT VALUE:** condition feedback on shot type/view/phase instructions, not a single universal shooting template.

## R096 PECoP

- **Year/domain/problem:** 2024, AQA; parameter-efficient continual pretraining for limited AQA data. **Links/type:** paper/code referenced by AQA survey; exact official repo URL not independently verified.
- **I/O/method:** pretrained video model + unlabeled target video -> adapted AQA representation; parameter-efficient temporal pretraining. Training required.
- **Strength/limit/governance:** useful for small basketball labels; can learn dataset artifacts from untrusted video. License UNKNOWN.
- **Project:** not tested. **Decision: research_later. EXACT VALUE:** adapt only on manifest-controlled trusted/quarantine-separated clips after baseline labels exist.

## R097 FineParser

- **Year/domain/problem:** 2024, AQA; parse fine action structure for quality. **Links/type:** CVPR 2024 paper identified via AQA surveys; authoritative repo/license not confirmed.
- **I/O/method:** action video -> parsed temporal components and score; fine-grained temporal parser. Training required.
- **Strength/limit/governance:** reinforces procedure-aware scoring; insufficient verified assets for immediate benchmark.
- **Project:** not tested. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** quality errors should attach to parsed phases rather than whole-video feature averages.

## R098 MTL-AQA

- **Year/domain/problem:** 2019, diving; jointly learn score and auxiliary action attributes. **Links/type:** foundational paper/dataset lineage in AQA surveys; exact official URL not reverified.
- **I/O/method:** video -> quality score plus auxiliary labels; multi-task spatiotemporal network. Training required.
- **Strength/limit/governance:** classic baseline; superseded by procedure/pose/causal models. License UNKNOWN.
- **Project:** not tested. **Decision: superseded. EXACT VALUE:** auxiliary shot type/phase labels can regularize scoring, but do not benchmark this old architecture.

## R099 MUSDL / USDL

- **Year/domain/problem:** 2020, AQA; model uncertain score distributions rather than point scores. **Links/type:** paper lineage documented in AQA surveys; exact official repo URL not reverified.
- **I/O/method:** video -> score distribution/uncertainty; distribution learning over temporal features. Training required.
- **Strength/limit/governance:** important uncertainty foundation; newer UIL-AQA is more interpretable. License UNKNOWN.
- **Project:** not tested. **Decision: idea_reference/superseded. EXACT VALUE:** predict intervals/distributions for subjective quality, not fake decimal precision.

## R100 CoRe AQA

- **Year/domain/problem:** 2021, AQA; comparative regression against exemplars. **Links/type:** paper lineage documented in AQA surveys; exact official repo URL not reverified.
- **I/O/method:** query and exemplar videos -> relative/absolute quality; contrastive regression over temporal features. Training required.
- **Strength/limit/governance:** comparison is more learnable than absolute score; view/difficulty mismatch biases results. License UNKNOWN.
- **Project:** not tested. Basketball fine-tune yes. **Decision: adopted idea. EXACT VALUE:** compare attempts matched by shot type/view/person scale and show evidence differences.

## R101 FLEX

- **Year/domain/problem:** 2025/2026, weightlifting/fitness; multimodal, multiview and rule-linked AQA. **Links/type:** paper https://arxiv.org/abs/2506.03198 ; project https://haoyin116.github.io/FLEX_Dataset . Dataset/benchmark.
- **I/O/method:** 5 RGB views + MoCap/3D pose/sEMG/physiology -> skill levels, keysteps, errors and feedback penalty graph. 7,512 samples, 20 actions, 38 subjects.
- **Strength/limit/governance:** rare knowledge-graph feedback supervision; weightlifting differs and data/license terms require review. Commercial risk UNKNOWN/HIGH.
- **Project:** newly studied; not downloaded/tested. **Decision: idea_reference. EXACT VALUE:** design basketball metric-to-error-to-feedback graph with explicit phase and evidence links.

## R102 DeepRehabPile

- **Year/domain/problem:** 2026, rehabilitation; standardized subject-aware skeleton quality benchmark. **Links/type:** repo https://github.com/MSD-IRIMAS/DeepRehabPile . Paper/repository/benchmark.
- **I/O/method:** skeleton sequences -> correct/incorrect classification or clinical score; nine deep architectures across 39 classification and 21 regression sets.
- **Strength/limit/governance:** stresses grouped splits and same-action subtle errors; clinical motions are slower and labels differ. License/source dataset terms vary.
- **Project:** newly studied, not tested. **Decision: benchmark_protocol_reference. EXACT VALUE:** subject-grouped nested evaluation and interpretable simple features before claiming deep AQA gains.

## R103 ExerciseLLM

- **Year/domain/problem:** 2025, rehabilitation; generate exercise feedback from structured movement descriptions. **Links/type:** repo https://github.com/jessicaxtang/exercisellm . Paper/repository/dataset derivation.
- **I/O/method:** categorized pose/movement evidence -> natural-language feedback through prompting; no visual perception itself.
- **Strength/limit/governance:** feedback formatting reference; LLM can amplify upstream errors and datasets inherit UI-PRMD/REHAB terms. License UNKNOWN.
- **Project:** newly studied, not tested. **Decision: idea_reference. EXACT VALUE:** LLM is the final verbalizer over a validated evidence JSON, never the measurement engine.

## R104 AIFit

- **Year/domain/problem:** 2021, fitness; automatic 3D human-interpretable exercise feedback. **Links/type:** CVPR 2021 paper referenced by rehabilitation/AQA benchmarks; exact official repo URL not confirmed.
- **I/O/method:** 3D pose sequence -> action quality and interpretable corrections; pose alignment/template features plus learned assessment.
- **Strength/limit/governance:** direct pose feedback precedent; controlled exercise assumptions and asset/license uncertainty.
- **Project:** not tested. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** retrieve a matched template and report localized deviations with uncertainty, not imitate its exercise rules.

# H. Camera, video quality and deployment

## R105 TransNetV2

- **Year/domain/problem:** 2020, video editing; detect hard/gradual shot boundaries. **Links/type:** repo https://github.com/soCzech/TransNetV2 ; paper https://arxiv.org/abs/2008.04838 . Paper/repository/model.
- **I/O/method:** video -> transition probabilities/frame ranges; dedicated temporal CNN. Pretrained inference available; ClipShots/BBC/RAI F1 benchmarks.
- **Strength/limit/governance:** mature and easy; camera cuts are rare in controlled phone clips but critical in internet videos. MIT license.
- **Project:** newly studied, not tested. Fine-tune no initially. **Decision: benchmark_now for quality gate. EXACT VALUE:** reject/split edited videos before interpreting pose continuity.

## R106 PySceneDetect

- **Year/domain/problem:** active, video tooling; practical content/adaptive scene detection and splitting. **Links/type:** docs https://www.scenedetect.com/docs/latest/ ; repo https://github.com/Breakthrough/PySceneDetect . Repository/tool.
- **I/O/method:** video -> cut list/scenes using classic content thresholds or detectors; no training.
- **Strength/limit/governance:** lightweight standard-library-style baseline; thresholds can fire on flashes/motion. BSD-style license should be verified in repo, low risk.
- **Project:** newly studied, not tested. **Decision: benchmark_now baseline. EXACT VALUE:** one-command cut metadata and comparison against TransNetV2; use the simpler winner.

## R107 DOVER / DOVER-Mobile

- **Year/domain/problem:** 2023, UGC video quality; separate technical and aesthetic quality. **Links/type:** repo https://github.com/VQAssessment/DOVER ; paper https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Exploring_Video_Quality_Assessment_on_User_Generated_Contents_from_Aesthetic_ICCV_2023_paper.pdf . Paper/repository/model.
- **I/O/method:** sampled video fragments -> technical, aesthetic and fused quality scores; two-branch 3D ConvNet, temporal/spatial sampling. Pretrained/ONNX/Mobile weights available. PLCC/SRCC on LSVQ/LIVE-VQC/KoNViD.
- **Strength/limit/governance:** efficient learned technical quality; global human-perception score may miss local ball blur. License/model terms need review.
- **Project:** newly studied, not tested. Fine-tune no for probe. **Decision: benchmark_later. EXACT VALUE:** auxiliary quality flag only; pair with explicit FPS/person/ball-scale/local-blur checks.

## R108 FAST-VQA / FasterVQA

- **Year/domain/problem:** 2022/2023, UGC quality; efficient end-to-end VQA. **Links/type:** repo https://github.com/VQAssessment/FAST-VQA-and-FasterVQA ; paper https://arxiv.org/abs/2207.02595 . Paper/repository/model.
- **I/O/method:** spatial-temporal fragments -> [0,1] quality; fragment sampling and 3D network. Pretrained weights; standard VQA correlation metrics.
- **Strength/limit/governance:** faster predecessor and useful CPU candidate; DOVER is its quality-dimension successor. License UNKNOWN.
- **Project:** not tested. **Decision: superseded by DOVER-Mobile for learned VQA. EXACT VALUE:** no parallel benchmark unless DOVER dependencies fail.

## R109 LightGlue

- **Year/domain/problem:** 2023, geometry; efficient adaptive local-feature matching. **Links/type:** paper https://arxiv.org/abs/2306.13643 ; repo https://github.com/cvg/LightGlue . Paper/repository/model.
- **I/O/method:** keypoints/descriptors from two frames -> matches/confidence; adaptive transformer matching, temporal use via frame pairs. Pretrained matchers for SuperPoint/DISK/ALIKED/SIFT.
- **Strength/limit/governance:** strong camera-motion homography input; moving people dominate if masks are absent. Apache-2 code reported by repo; feature weights terms separate.
- **Project:** newly studied, not tested. Fine-tune no. **Decision: benchmark_later. EXACT VALUE:** estimate background homography to flag/compensate camera movement before trajectory derivatives.

## R110 KaliCalib

- **Year/domain/problem:** 2022, basketball broadcast; court registration and camera calibration. **Links/type:** repo https://github.com/CEA-LIST/KaliCalib ; paper https://arxiv.org/abs/2209.07795 . Paper/repository.
- **I/O/method:** basketball court image -> court keypoints/registration/camera; learned spatial court model. Training/checkpoints per repo; court-calibration metrics.
- **Strength/limit/governance:** direct basketball geometry; close phone clips often omit court lines and moving broadcast assumptions differ. CeCILL-2.1 code; data/weights separate, commercial review needed.
- **Project:** newly studied, not tested. Fine-tune maybe. **Decision: benchmark_later for wide views. EXACT VALUE:** enable court-relative coordinates only when enough court landmarks are visible; otherwise explicitly unavailable.

## R111 DeepSportRadar Camera Calibration Challenge

- **Year/domain/problem:** 2022+, basketball court calibration; standard benchmark from FIBA broadcast frames. **Links/type:** repo https://github.com/DeepSportradar/camera-calibration-challenge . Dataset/benchmark.
- **I/O/method:** court images -> line/intersection/camera parameters; benchmark protocols and metrics. 728 frames across train/val/test.
- **Strength/limit/governance:** direct calibration test; tiny dataset/broadcast domain and dataset license require review.
- **Project:** not downloaded/tested. **Decision: benchmark_reference. EXACT VALUE:** evaluate any court calibration challenger before exposing metric spatial claims.

## R112 TVCalib / PnLCalib / BroadTrack family

- **Year/domain/problem:** 2023-2025, soccer/broadcast geometry; differentiable field calibration and temporal camera tracking. **Links/type:** TVCalib https://mm4spa.github.io/tvcalib/ ; PnLCalib https://github.com/mguti97/PnLCalib ; BroadTrack https://github.com/evs-broadcast/BroadTrack . Three related paper/repositories counted as one family record.
- **I/O/method:** field line/point observations over frames -> camera parameters/tracks; differentiable reprojection, point-line optimization and temporal tracking.
- **Strength/limit/governance:** successor lineage beyond single-frame templates; soccer-field/broadcast dependency and licenses differ. Commercial risk UNKNOWN.
- **Project:** newly studied, not tested. Fine-tune/calibration data yes. **Decision: research_reference. EXACT VALUE:** if product later supports wide-court videos, benchmark temporal calibration; V1 fixed close video uses no field calibration.

# I. Basketball sports-science evidence resources

These records are method evidence, not model-training resources. Product-safe interpretation is expanded in `biomechanics-evidence-base-2026-09-01.md`.

## R113 Okazaki, Rodacki and Satern jump-shot review

- **Year/domain/problem:** 2015, basketball biomechanics; synthesize trajectory, phases and influencing variables. **Link/type:** https://pubmed.ncbi.nlm.nih.gov/26102462/ . Systematic review.
- **I/O/method:** prior laboratory studies -> critical components; no code/training/weights/benchmark. Strength: broad foundation; limit: heterogeneous populations/tasks. Publisher copyright; low use risk.
- **Project:** literature studied, not experimentally tested. **Decision: evidence_reference. EXACT VALUE:** organize metrics by ball trajectory, movement phase and context, not universal ideal angles.

## R114 Youth jump-shot systematic review

- **Year/domain/problem:** 2021, youth basketball; effects of distance, fatigue, defender and visual information. **Link/type:** https://pmc.ncbi.nlm.nih.gov/articles/PMC8005190/ . Systematic review.
- **I/O/method:** youth studies -> contextual evidence; reports distance-related lower release angles/higher velocities and heterogeneous fatigue effects. Open article; no model assets.
- **Project:** studied, not tested. **Decision: evidence_reference. EXACT VALUE:** require age/shot distance/defense context before coaching language.

## R115 Miller and Bartlett distance/position study

- **Year/domain/problem:** 1996, basketball; shooting distance and playing position. **Link/type:** https://pubmed.ncbi.nlm.nih.gov/8809716/ . Controlled 3D cinematography study.
- **I/O/method:** 15 players, three distances at 100 Hz -> release speed/angle, joint and CoM timing. Strength: controlled 3D; limit: small adult sample.
- **Project:** literature only. **Decision: evidence_reference. EXACT VALUE:** longer shots legitimately use higher speed/lower angle/earlier release, so distance-normalized comparison is mandatory.

## R116 Rojas et al. defender study

- **Year/domain/problem:** 2000, professional basketball; technique adjustment against an opponent. **Link/type:** https://pubmed.ncbi.nlm.nih.gov/11083144/ . Controlled 3D video study.
- **I/O/method:** 10 professionals, with/without defender at 50 Hz -> release angle, flight time and joint posture; defender increased release angle and shortened flight time.
- **Project:** literature only. **Decision: evidence_reference. EXACT VALUE:** defended and open shots cannot share one reference template.

## R117 Increased shooting distance study

- **Year/domain/problem:** 2013, expert basketball; effect of 2.8/4.6/6.4 m distance. **Link/type:** https://pubmed.ncbi.nlm.nih.gov/24149195/ . Controlled kinematic study.
- **I/O/method:** 10 male experts -> accuracy, release height/angle/velocity; accuracy fell 59% to 37%, height/angle decreased and velocity increased.
- **Project:** literature only. **Decision: evidence_reference. EXACT VALUE:** report within-distance consistency, never penalize a long shot for differing from close-shot averages.

## R118 Female adolescent distance study

- **Year/domain/problem:** 2024, youth female basketball; kinematic adaptation from 4.75 to 5.75 m. **Link/type:** https://pubmed.ncbi.nlm.nih.gov/38314460/ . Controlled 2D marker study.
- **I/O/method:** 27 players, 10 shots each -> release/body joint variables; longer distance reduced angle, raised velocity and changed shoulder/knee motion.
- **Project:** literature only. **Decision: evidence_reference. EXACT VALUE:** population-specific evidence and release-before-jump-peak are descriptive, not universal pass/fail thresholds.

## R119 Near-minimum release speed strategy

- **Year/domain/problem:** 2020, free throw motor control; expert strategy under motor noise. **Link/type:** https://pubmed.ncbi.nlm.nih.gov/32217201/ . Motion-capture/simulation study.
- **I/O/method:** eight collegiate + one professional player -> release solution manifold; measured angles were 2.8 +/- 3.1 degrees from minimum-speed solution.
- **Project:** literature only. **Decision: evidence_reference. EXACT VALUE:** emphasize repeatable release parameter combinations, not isolated joint-angle optimization.

## R120 Hierarchical redundancy in free throws

- **Year/domain/problem:** 2022, basketball motor coordination; body and ball-level redundancy. **Link/type:** https://pubmed.ncbi.nlm.nih.gov/34968877/ . Motion-capture study.
- **I/O/method:** experienced players with/without feedback -> solution-manifold/UCM variability; inter-joint coordination improved release-position reproducibility.
- **Project:** literature only. **Decision: evidence_reference. EXACT VALUE:** quantify coordination/consistency across attempts rather than demanding one identical pose.

## R121 Misses versus swishes coordination variability

- **Year/domain/problem:** 2010, free throws; distinguish successful and missed coordination. **Link/type:** https://pubmed.ncbi.nlm.nih.gov/20552519/ . Controlled kinematic study.
- **I/O/method:** collegiate players, 20 attempts -> release parameters and elbow-wrist coordination; misses showed higher coordination variability in final 0.01 s.
- **Project:** literature only. **Decision: evidence_reference. EXACT VALUE:** release-window elbow-wrist timing variability is a candidate longitudinal metric, not a one-shot diagnosis.

## R122 Repeated-sprint fatigue and three-point shooting

- **Year/domain/problem:** 2018, experienced basketball; fatigue effects on three-point kinematics. **Link/type:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6006537/ . Controlled fatigue study.
- **I/O/method:** repeated-sprint protocol -> shot accuracy and kinematics; high-level players showed substantial kinematic stability, limiting universal fatigue claims.
- **Project:** literature only. **Decision: evidence_reference. EXACT VALUE:** do not infer fatigue from one changed angle; compare repeated attempts and context.

## R123 Context constraints and shooting accuracy

- **Year/domain/problem:** 2025, senior basketball; position, noise, opposition and release/jump parameters. **Link/type:** https://pubmed.ncbi.nlm.nih.gov/41283566/ . Force-platform/high-speed-camera study.
- **I/O/method:** 18 players, 90 shots each -> jump/release variables; isolated biomechanical associations with accuracy were weak (`R^2` 0.005-0.012).
- **Project:** literature only. **Decision: evidence_reference. EXACT VALUE:** forbid causal claims such as “this angle caused the miss” from monocular observation.

## R124 2026 athlete-versus-novice free throws

- **Year/domain/problem:** 2026, free throws; 2D athlete/novice and success/miss comparison. **Link/type:** https://pubmed.ncbi.nlm.nih.gov/42358510/ . Controlled sagittal 2D study.
- **I/O/method:** 50 adults, 20 trials, Dartfish -> joints, trunk, hand and release; reliability ICC .88-.95; greater elbow extension/release height correlated with success in this protocol.
- **Project:** literature only. **Decision: evidence_reference with caution. EXACT VALUE:** supports sagittal 2D feasibility under controlled view, but values remain free-throw/population-specific and cannot become universal rules.

# J. Additional transfer-domain controls

## R125 Interpretable pre-release baseball pitch anticipation

- **Year/domain/problem:** 2026, baseball; derive interpretable pre-release information from broadcast 3D kinematics. **Links/type:** paper https://arxiv.org/abs/2603.04874 . Paper.
- **I/O/method:** broadcast pitcher video -> automatic pitch event, monocular 3D pose, 229 biomechanical features and pitch type; diffusion-based pose plus gradient-boosted classifier. Training required; ground-truth-validated features reported.
- **Strength/limit/governance:** demonstrates event-aligned interpretable sports features at scale; pitch classification and broadcast 3D validity do not transfer to shooting advice. Code/license UNKNOWN.
- **Project:** newly studied, not tested. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** align all kinematic features to a validated event and audit feature influence instead of feeding arbitrary whole-video pose averages.

## R126 Volleyball group activity dataset

- **Year/domain/problem:** 2016, volleyball; jointly understand individuals and group play. **Links/type:** official dataset/project lineage http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/ ; paper benchmark. Dataset.
- **I/O/method:** 55 match videos/4,830 annotated frames -> player boxes/actions and team activity; hierarchical spatial-temporal modeling. Training required; group/action accuracy metrics.
- **Strength/limit/governance:** shooter-of-interest and teammate/defender context analogy; sparse broadcast labels and no ball/contact detail. Dataset license UNKNOWN.
- **Project:** newly studied, not downloaded/tested. **Decision: idea_reference. EXACT VALUE:** shooter selection may combine individual pose/ball proximity with group context rather than choosing the highest-confidence person.

## R127 MMFS figure-skating benchmark

- **Year/domain/problem:** 2023, figure skating; fine-grained multimodal action recognition and quality assessment. **Links/type:** paper https://arxiv.org/abs/2307.02730 . Paper/dataset.
- **I/O/method:** championship RGB/skeleton modalities -> fine actions and quality scores; RGB and skeleton baselines with temporal aggregation. Training required.
- **Strength/limit/governance:** tests pose contribution in complex skill; broadcast/judged routine and dataset terms differ. License UNKNOWN.
- **Project:** newly studied, not tested. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** benchmark pose-only against visual+pose under identical splits before assuming skeleton is sufficient for quality.

## R128 TaiChi-AQA

- **Year/domain/problem:** 2026, martial arts; quality assessment and visual explanation for technique sequences. **Links/type:** repo/dataset https://github.com/mlxger/TaiChi-AQA . Paper/repository/dataset.
- **I/O/method:** Tai Chi video -> multi-label quality and visual analysis; pose/temporal quality framework. Training required.
- **Strength/limit/governance:** whole-body technique and explainability analogy; slower continuous motion and self-released dataset require independent validation. License UNKNOWN.
- **Project:** newly studied, not tested. Basketball fine-tune yes. **Decision: idea_reference. EXACT VALUE:** evaluate phase/body-part explanations separately from overall score.

## R129 UMONS-TAICHI

- **Year/domain/problem:** 2017+, martial arts; multimodal motion-capture dataset of expertise. **Links/type:** repo/dataset https://github.com/numediart/UMONS-TAICHI . Dataset/repository.
- **I/O/method:** 2,200 3D motion samples, 13 gestures, 12 participants at varied skill -> morphology-independent expertise features. No pretrained production model assumed.
- **Strength/limit/governance:** controlled expert/novice 3D reference; small participants and different motion. Dataset/code terms require review.
- **Project:** newly studied, not downloaded/tested. **Decision: idea_reference. EXACT VALUE:** normalize morphology and compare timing/style, not raw joint positions across people.

## R130 MonocularBiomechanics clinical gait pipeline

- **Year/domain/problem:** 2026, rehabilitation/running-gait; clinically validate smartphone monocular whole-body kinematics. **Links/type:** repo https://github.com/IntelligentSensingAndRehabilitation/MonocularBiomechanics . Paper/repository/model.
- **I/O/method:** handheld phone video -> fitted body motion and gait metrics; biomechanical fitting around monocular human reconstruction. Authors report ICC >0.9 for selected metrics across 1,021 deployment videos.
- **Strength/limit/governance:** strongest recent ordinary-phone validation signal; clinical gait is slower/repetitive and does not validate jump-shot release. Code/model/body-model/data licenses must be checked.
- **Project:** newly studied, not tested. Basketball validation required. **Decision: research_challenger. EXACT VALUE:** copy prospective per-metric clinical validation discipline, not its gait-specific reliability claims.

# Domain coverage

| Domain | Representative resources | Transfer target |
|---|---|---|
| Basketball | BASKET, KaliCalib, 12 biomechanics sources, current V1 projects | Direct data, context and scientific rules |
| Tennis | TrackNet/TrackNetV4, PTS/T-DEED tennis splits | Fast tiny ball and exact events |
| Badminton | TrackNetV3/V4, shuttle hitting event detection | Blur, occlusion and contact events |
| Table tennis | TTNet, BlurBall | Multi-task ball/event and blur labels |
| Baseball | Pre-release pitch kinematics, baseball tracking lineage | Event-aligned interpretable features |
| Golf | GolfDB/SwingNet | Ordered sparse action events |
| Volleyball | Volleyball dataset, SportsMOT | Person-of-interest and group context |
| Soccer | SoccerNet spotting, ball-action winner, field calibration lineage | Long-video events and camera geometry |
| Diving | FineDiving, HP-MCoRe, AdaSpot | Phase-aware AQA and exact events |
| Gymnastics / figure skating | T-DEED, MMFS, UIL-AQA | Long skill quality and uncertainty |
| Weightlifting / fitness | FLEX, Fitness-AQA, NS-AQA | Error taxonomy and rule-based feedback |
| Sprint / running | MonocularBiomechanics, OpenCap | Smartphone validation and repeatability |
| Martial arts | TaiChi-AQA, UMONS-TAICHI | Whole-body expertise and morphology normalization |
| Rehabilitation / biomechanics | DeepRehabPile, Sports2D, Pose2Sim, OpenCap | Measurement validity and grouped evaluation |
| HOI / egocentric | 100DOH, Hands23, EgoHOS, EPIC/VISOR, HOT3D | Contact state and active object relations |
| Robotics manipulation | ContactPose and contact-state transition lineage | State-machine semantics |
| Industrial skill | Assembly101, PROSKILL | Mistakes, free-order procedures and pairwise ranking |
| Surgical workflow | TeCNO, Trans-SVNet | Stable long-context phases and causal online tests |

# Cross-domain transfer conclusions

1. **Strict release is primarily an HOI transition plus precise event problem**, not a pose-only event. 100DOH/Hands23 supply state semantics; CoTracker3 supplies continuous ball evidence; PTS/T-DEED supply exact-frame decoding.
2. **Fast-ball continuity is better addressed by tennis, badminton and table tennis research** than by current basketball repositories. TrackNetV4, BlurBall and TTNet show motion heatmaps, blur labels and joint event heads.
3. **Phase recognition is better developed in golf, surgery and industrial procedures.** GolfDB supplies ordered sparse events; MS-TCN++/ASFormer supply dense phases; Assembly101 warns that real procedures include variable ordering and correction.
4. **Explainable quality is better developed in diving, fitness and rehabilitation.** FineDiving/HP-MCoRe provide phase-aware scoring; NS-AQA/FLEX provide explicit rule/knowledge graphs; DeepRehabPile provides subject-grouped evaluation.
5. **Metric biomechanics is best bounded by rehabilitation/sports validation, not model demos.** Sports2D, Pose2Sim and OpenCap show the distinction between accessible measurement and validated measurement.

# What was actually tested by this project

- YOLO11 pose and the project V1 basketball detector/release-fusion path: yes, already integrated, with documented limitations.
- LocateAnything local Q8 prelabeling: yes; prompts detected the ball approximately, systematic box offset/occlusion issues required human review, and results remained research-only.
- Local batch extraction, prelabel and review workflow under `E:/BasketballShotAI/tools/locateanything_local`: yes; generated artifacts are outside Git and are not product evidence.
- Basketball-Shot-Detection external repository: partially cloned/read/run in earlier work; ideas retained, no product-quality result accepted.
- All other resources in this landscape: **not tested on project videos during this sprint**. This sprint performed no model download, training or new benchmark.

# Research gaps

- A trusted, source-video-disjoint basketball contact/release dataset with hand visibility and exact frame labels.
- A small ball-center/visibility tracking benchmark around release with motion blur and occlusion.
- Controlled comparison of YOLO11 pose, RTMPose and RTMW on the project's real camera views.
- Independent validation of selected 2D metrics against synchronized multi-view or marker-based reference.
- Basketball-specific phase annotation reliability and inter-reviewer frame tolerance.
- Expert-authored, population- and shot-type-conditioned feedback rules with forbidden claims.
- License review for every model weight and external dataset before any production/commercial path.
- Feedback evaluation by qualified basketball coaches, including evidence correctness and actionable usefulness.

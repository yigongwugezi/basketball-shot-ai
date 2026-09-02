# Pose Accuracy Closure Research — 2026-09-02

## Question and evidence boundary

The search was restricted to: **How should offline fast-motion human pose trajectories be made accurate, not merely smooth, especially under motion blur and occlusion?** It reviewed seven directly relevant method families. Benchmark claims below come from papers or official repositories, not from this project's clips. They are external evidence, not local accuracy measurements.

## Relevant methods

| Method | Accuracy mechanism | Relevance | Deployment finding |
|---|---|---|---|
| [PoseWarper](https://github.com/facebookresearch/PoseWarper) / [NeurIPS 2019 paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html) | Learned deformable warping and ±3-frame heatmap aggregation | Explicitly targets blur, defocus, occlusion, and large motion using neighboring image evidence | Strong conceptual fit, but a legacy HRNet/PoseTrack training stack rather than a drop-in postprocessor for current YOLO coordinates |
| [DCPose](https://github.com/Pose-Group/DCPose) / [CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Deep_Dual_Consecutive_Network_for_Human_Pose_Estimation_CVPR_2021_paper.html) | Bidirectional consecutive frames, pose residual fusion, heatmap correction | Directly reports gains on rapid motion, nearby people, occlusion, and defocus; difficult wrist/ankle joints are explicit evaluation targets | Best conceptual challenger after GT exists, but requires its own heatmap backbone and older custom training/runtime stack |
| [FAMI-Pose](https://github.com/Pose-Group/FAMI-Pose) / [CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Temporal_Feature_Alignment_and_Mutual_Information_Maximization_for_Video-Based_Human_CVPR_2022_paper.html) | Coarse-to-fine temporal feature alignment plus task-relevant mutual information | Designed for fast motion and occlusion without naïvely aggregating misaligned features | Official repository says pretrained models are unavailable and uses a PyTorch 1.8/DCN-era stack; not a proportionate local challenger |
| [TDMI](https://github.com/FRunyang/TDMI) / [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Feng_Mutual_Information-Based_Temporal_Difference_Learning_for_Human_Pose_Estimation_in_Video_CVPR_2023_paper.html) | Multi-stage temporal-difference encoding and motion/noise disentanglement | Avoids irrelevant nearby-person/background motion cues that can corrupt temporal aggregation | Research direction is relevant to the observed bystander failure, but integration requires a trained video pose model, not a coordinate-only patch |
| [SmoothNet](https://github.com/cure-lab/SmoothNet) / [ECCV 2022 paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/615_ECCV_2022_paper.php) | Long-window temporal-only learned refinement with velocity/acceleration branches | Useful warning and baseline: it measures both acceleration and position error and can improve difficult frames | Available pretrained weights center on 3D backbones/domains; accuracy is described as a side effect. It cannot be accepted here from lower jitter alone |
| [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) | Strong single-frame top-down pose head with SimCC-style coordinate classification | Practical maintained comparison with good runtime; already cached and packaged | Uses YOLO bbox in this benchmark; must be judged by human joint GT |
| [RTMW](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) | Whole-body multi-dataset RTMPose family | Potentially useful if hand/release localization later matters | More joints do not imply better 12-joint body accuracy; current comparison remains GT-gated |

## Conclusions for this project

1. Image evidence from neighboring frames can improve true pose accuracy under blur or occlusion. The strongest methods align or correct heatmaps/features; they do not merely low-pass filter coordinates.
2. Temporal refinement should be selective. Reliable high-confidence current-frame coordinates should remain unchanged; suspicious outliers and short gaps are the correction targets. This matches the proposed `reliable → keep raw`, `suspicious → correct`, `short gap → interpolate`, `long gap → unavailable` policy.
3. Temporal smoothness is a separate objective. SmoothNet is especially relevant because its paper evaluates both acceleration and localization error; importing only its smoothness premise would repeat the evidence mistake this task is closing.
4. Identity continuity must precede joint smoothing. On the generated comparison sheet, YOLO RAW switches to a bystander at IMG_7215 frame 133 and IMG_7216 frames 150–151; FILTERED marks those frames unavailable. RTMPose/RTMW crop comparisons also become unavailable because the trusted YOLO subject crop is intentionally withheld. This is measured visual/system behavior, not pose accuracy.
5. No additional challenger was run. Without human GT it could not change the backbone decision, while the most relevant video methods require non-trivial model training or legacy stacks. Adding one now would create runtime and integration work without resolving the decision gate.

## Challenger rule after human GT

If the 42-frame GT shows that all four current candidates fail materially on blur/occlusion while identities remain correct, DCPose is the first bounded challenger to consider. Run it on the same crops and frames only, record preprocessing and runtime, and accept it only if it improves human-GT wrist/elbow/release-window accuracy. Do not add both DCPose and another video model.

If GT instead shows that RAW is usually accurate and FILTERED causes positive `delta_error`, do not add a video model first. Redesign the current layer as selective correction and test at most these three policies:

1. outlier-only correction;
2. confidence-aware selective smoothing;
3. joint-specific correction, with shooting wrist/elbow preserved whenever reliable.

## Current decision

The previous `KEEP_YOLO_FILTERED` result remains a proxy-based provisional decision only. Human-GT accuracy is not available yet.

`POSE_BACKBONE_DECISION_FINAL = INSUFFICIENT_EVIDENCE`

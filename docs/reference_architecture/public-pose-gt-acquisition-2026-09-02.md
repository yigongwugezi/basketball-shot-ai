# Public Pose GT acquisition and benchmark integration — 2026-09-02

## Outcome

Eight serious sources were screened. JHMDB was acquired in full and Leeds Sports Pose (LSP) was acquired as full annotations plus a deterministic 30-image RGB review subset. Both were parsed into a provenance-preserving normalized layer and stopped at `REVIEW_READY / USER_DATASET_REVIEW=PENDING`.

MPII Human Pose annotations were acquired, but the 12.9 GB image archive was deliberately deferred. Penn Action and COCO official downloads were reachable but unusably slow in this run; their incomplete E-drive partial files are explicitly non-data. PoseTrack21, Ego-Exo4D, and SportsPose require approval or a signed license and were not bypassed.

The machine-readable source of record is [`public_pose_datasets.json`](../../benchmarks/reference_v1/public_pose_datasets.json). It records maintainer, version/year, source URLs, access method, expected/downloaded size, citation, RGB/video and annotation capabilities, camera setup, GT type, license class, mismatch, role, and acquisition/review status for every candidate.

## Candidate decision matrix

| Dataset | Problem relevance | GT / signals | Access result | License class | Current role/status |
|---|---|---|---|---|---|
| JHMDB | Basketball shooting, throwing, striking, jumping, fast articulation and short temporal clips | `HUMAN_GT`; 15 body joints on every frame; no per-joint visible/occluded split | Four official archives, 199,843,383 bytes, acquired in full | `RESEARCH_ONLY` | `POSE_BENCHMARK`, `MOTION_RESEARCH`; `REVIEW_READY` |
| Leeds Sports Pose | Difficult athletics, badminton, baseball, gymnastics, parkour, soccer, tennis and volleyball poses | `HUMAN_GT`; 14 joints with binary occlusion | Full `joints.mat` plus deterministic 30-image review subset from original-archive mirror | `UNCLEAR` | Static `POSE_BENCHMARK`; `REVIEW_READY` |
| MPII Human Pose | Broad difficult-pose/activity control | `HUMAN_GT`; 16 joints, scale/head box and visibility | Official 12,340,483-byte annotations acquired; 12.9 GB images deferred | `RESEARCH_ONLY` | `BLOCKED_MISSING_IMAGES` |
| COCO 2017 Keypoints | General 17-joint control and possible sports-object filtering | `HUMAN_GT`; 17 joints, COCO visibility; boxes/objects in separate file | Official and mirror annotation transfers timed out; 114,688-byte partial is invalid | `UNCLEAR` | `BLOCKED_INCOMPLETE_DOWNLOAD` |
| Penn Action | Per-frame fast sports, throwing/striking/jumping | `HUMAN_GT`; 13 joints, visibility and boxes | Official 3,235,203,923-byte archive projected at 15+ hours; 3,022,848-byte partial is invalid | `UNCLEAR` | Future temporal benchmark; `BLOCKED_INCOMPLETE_DOWNLOAD` |
| PoseTrack21 | Difficult video pose, occlusion, tracking identity and crop failures | `HUMAN_GT`; temporal multi-person pose | Signed agreement and token required | `RESTRICTED` | Future temporal control; `BLOCKED_ACCESS_APPROVAL` |
| Ego-Exo4D Body Pose | Highest basketball relevance; synchronized ego/exo body and hand views | `MULTIVIEW_GT`; 2D/3D body, optional hands | Signed license and approval required | `RESTRICTED` | Future pose/hand/motion/ball research; `BLOCKED_ACCESS_APPROVAL` |
| SportsPose | High-speed 3D sports motion | `MOCAP_GT`; 176K 3D poses, 24 subjects, 5 sports | Academic email request required | `RESEARCH_ONLY` | Future motion research; `BLOCKED_ACCESS_APPROVAL` |

No candidate with `AUTO_LABEL`, `PSEUDO_LABEL`, or unknown annotation provenance was admitted to the review-ready set.

## Acquired data and provenance

All downloads began and ended on E:. No dataset was downloaded to C: and moved afterward.

```text
E:\BasketballShotAI\public_data\downloads\
  JHMDB_video.zip              185,602,053 bytes
  joint_positions.zip          14,177,716 bytes
  splits.zip                       43,785 bytes
  sub_splits.zip                   19,829 bytes
  mpii_human_pose_v1_u12_2.zip 12,340,483 bytes
  Penn_Action.tar.gz            3,022,848 bytes (incomplete; not parsed)
  annotations_trainval2017.zip    114,688 bytes (incomplete; not parsed)

E:\BasketballShotAI\public_data\datasets\fast_sports\jhmdb\
E:\BasketballShotAI\public_data\datasets\fast_sports\leeds_sports_pose\
E:\BasketballShotAI\public_data\datasets\general_pose\mpii\
```

JHMDB normalized packages store SHA-256 values for all four official archives. LSP stores the original full annotation SHA-256 (`ddcfcbe904106bfbf26561c8d11720ca62a11c7a861548064b3bae84cb396b2a`) and a SHA-256 for every selected source image. Raw joint arrays and names remain alongside the derived 12-joint mapping.

## Review experience

Open:

```text
E:\BasketballShotAI\public_data\dataset_review\index.html
```

The index links to:

- JHMDB: 36 real-GT frames across `shoot_ball`, `throw`, `swing_baseball`, `golf`, `jump`, and `catch`.
- LSP: 30 deterministic samples spanning the full 2,000-item index range.

The JHMDB sample has 402 in-frame mapped joints and 30 source-annotated joints outside the RGB bounds; those 30 retain their source coordinates but map to explicit `OUT_OF_FRAME / not_labelable`. The LSP sample has 335 visible and 25 occluded mapped joints.

Each page displays source RGB with GT-only overlays, left/right color separation, filled versus occluded/visibility-unspecified points, dataset/domain, GT type, schema, license, downloaded scope, mismatch, provenance links, and recommended role. The decision file remains:

```json
{"datasets":{"jhmdb":"PENDING","lsp":"PENDING"}}
```

No dataset was auto-accepted.

## Adapter and benchmark boundary

`benchmarks/reference_v1/public_pose_gt.py` implements:

- exact JHMDB and LSP joint-name/order adapters;
- original-coordinate and original-visibility preservation;
- explicit `NOT_AVAILABLE` and `OUT_OF_FRAME` states rather than invented points;
- conservative JHMDB visibility mapping (`occluded_but_inferable`) because the source labels every joint but does not separate visible from occluded;
- review HTML and normalized candidate manifests;
- a persistent `PENDING / ACCEPTED / REJECTED` decision gate;
- export to the existing `pose_gt_manifest_v1` and `pose_gt_annotations_v1` schemas only after `ACCEPTED`;
- rejection of `AUTO_LABEL` and `PSEUDO_LABEL` as trusted export sources;
- explicit person/crop failure accounting separate from pose-head missing joints, while crop failures remain in end-to-end failure.

After acceptance, the existing pose evaluator supplies pixel error, body-scale normalized error, PCK, median/P90/P95, joint/body-region groupings, temporal metrics, and RAW-versus-FILTERED damage. RAW YOLO, FILTERED YOLO, RTMPose, and RTMW predictions remain separate inputs. RTMPose/RTMW still require explicit person/crop status because they depend on YOLO crops.

The public GT is therefore executable at the adapter/schema level, but product benchmark execution is intentionally blocked on user dataset review. No pose-backbone accuracy conclusion is published in this state.

## Limitations and next decision

- JHMDB is the strongest available temporal public GT, but is low resolution, research-only, and does not identify per-joint visibility.
- LSP is sports-relevant and has visibility, but is static, low resolution, and has unclear image-reuse terms.
- Neither source has hand or ball GT.
- The exact basketball/product-domain 42-frame package remains unlabelled and unchanged.

The next valid step is visual accept/reject of JHMDB and LSP. Only accepted sources may be exported under `E:\BasketballShotAI\public_data\benchmark_ready\` and used to run the four-model comparison.

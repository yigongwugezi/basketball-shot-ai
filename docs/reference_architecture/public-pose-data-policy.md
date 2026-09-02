# Public pose data policy

The project does not require the product owner to manually produce datasets as the default development path.

Public datasets, public GT, existing benchmarks, automated preprocessing, and machine-assisted annotation are preferred.

The product owner performs visual review and acceptance of candidate datasets before they become trusted benchmark/training sources.

Manual annotation by the product owner is reserved for exceptional, high-value validation cases and must not be assumed as routine work.

Large datasets, model weights, caches, extracted media, benchmark outputs, and generated review artifacts must not be stored on C:. Use `E:\BasketballShotAI\...` for large data.

## Trust and acceptance boundary

- Trusted ground-truth types are `HUMAN_GT`, `MOCAP_GT`, and `MULTIVIEW_GT`.
- `AUTO_LABEL` and `PSEUDO_LABEL` remain separate inputs and cannot be exported as trusted GT.
- A newly prepared public dataset stops at `REVIEW_READY`. It becomes a trusted benchmark source only after the project owner records `USER_ACCEPTED` for that exact source/version.
- `RESEARCH_ONLY`, `RESTRICTED`, and `UNCLEAR` data stays isolated from commercial training. An adapter or review acceptance does not change its license.
- Original annotations, coordinates, joint names, visibility values, source sample identity, and provenance are retained. Derived mappings never replace the originals.

## Storage layout

Large public-data work uses:

```text
E:\BasketballShotAI\public_data\
  downloads\
  datasets\basketball\
  datasets\fast_sports\
  datasets\general_pose\
  cache\
  dataset_review\
  benchmark_ready\
  tmp\
```

Experimental outputs use `E:\BasketballShotAI\analysis_runs\`. Only lightweight code, tests, documentation, manifests without media payloads, and metadata belong in the repository.

## Existing product-domain package

The 42-frame `pose_gt_v1` package and its annotation tool are retained unchanged. Its status remains `WAITING_FOR_HUMAN_LABELS`; public-data work must not populate it from model predictions or present pseudo-labels as human GT.

# Reference V1 Benchmark Harness

Small, cache-first benchmark harness for the Basketball Shot AI reference stack.

## Outputs

Default generated output is outside Git:

`E:\BasketballShotAI\analysis_runs\reference_benchmark\`

It contains cached JSON inference, detailed CSV rows, point trajectories, contact sheets, aggregate CSV/JSON results and `failures.jsonl`.

## Run

```powershell
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py tracking_pose
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py ball
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py sahi
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py point
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py cotracker
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py grounded_sam2
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py phase
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py strict
```

Validation-closure utilities reuse cached Wave 2 evidence and do not require model inference:

```powershell
python benchmarks\reference_v1\validation_closure.py self-test
python benchmarks\reference_v1\make_strict_release_review.py
python benchmarks\reference_v1\validation_closure.py strict-cache --input-dir "E:\BasketballShotAI\analysis_runs\reference_benchmark\strict_release" --output "tmp\reference_validation_closure\strict_decoder_v1.json"
```

The strict-release review output deliberately keeps reviewer 2 pending. Do not use an
algorithm prediction as a second human annotation. The cached track evaluator accepts
human ball-center labels; detector agreement is reported only as a proxy when GT is absent.

Add `--include-rtmlib` after installing RTMLib to benchmark RTMPose/RTMW. Use `--no-cache` only when intentionally rerunning inference.

The 230-frame ball benchmark is research-only and is not an independent product test.

Optional candidates require `rtmlib`, `onnxruntime`, `rfdetr`, `transformers`, and `sahi`.
Set `XDG_CACHE_HOME`, `TORCH_HOME`, and `HF_HOME` to an E-drive directory before
first use so model caches stay outside the repository and C drive.

## Human pose GT and motion representation

Prepare the fixed 42-frame human-review package from existing model caches:

```powershell
.\.venv310\Scripts\python.exe benchmarks\reference_v1\pose_gt_benchmark.py prepare
```

After replacing the starter `annotations\pose_gt.json` with the reviewed export:

```powershell
.\.venv310\Scripts\python.exe benchmarks\reference_v1\pose_gt_benchmark.py evaluate
```

The evaluator refuses unreviewed frames. Model predictions and annotations are separate files.

Public human GT candidates use the E-drive data boundary and a separate user-acceptance gate:

```powershell
.\.venv310\Scripts\python.exe benchmarks\reference_v1\public_pose_gt.py prepare-all
.\.venv310\Scripts\python.exe benchmarks\reference_v1\public_pose_gt.py verify
```

Review `E:\BasketballShotAI\public_data\dataset_review\index.html`. Do not record an
acceptance from an automated run. After the product owner explicitly accepts an exact
dataset/version, record and export it with:

```powershell
.\.venv310\Scripts\python.exe benchmarks\reference_v1\public_pose_gt.py decision --dataset jhmdb --value ACCEPTED
.\.venv310\Scripts\python.exe benchmarks\reference_v1\public_pose_gt.py export --dataset jhmdb
```

The exporter rejects pseudo/automatic labels and refuses `PENDING` or `REJECTED` data.
See `docs/reference_architecture/public-pose-data-policy.md` for the trust, license, and
storage boundary.

Run the accepted four-pipeline public-GT benchmark (large outputs remain on E:):

```powershell
.\.venv310\Scripts\python.exe benchmarks\reference_v1\public_pose_benchmark.py
```

Use `--reuse-inference` to rebuild metrics/review HTML from a valid saved inference artifact.

Build a standalone motion debugger from a completed Reference V1 run:

```powershell
.\.venv310\Scripts\python.exe benchmarks\reference_v1\motion_debugger.py "E:\BasketballShotAI\analysis_runs\pose_reliability_pass\accepted_img_7216"
```

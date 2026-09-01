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

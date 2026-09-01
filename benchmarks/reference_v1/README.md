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
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py point
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py cotracker
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py grounded_sam2
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py phase
.\.venv310\Scripts\python.exe benchmarks\reference_v1\benchmark.py strict
```

Add `--include-rtmlib` after installing RTMLib to benchmark RTMPose/RTMW. Use `--no-cache` only when intentionally rerunning inference.

The 230-frame ball benchmark is research-only and is not an independent product test.

Optional candidates require `rtmlib`, `onnxruntime`, `rfdetr`, and `transformers`.
Set `XDG_CACHE_HOME`, `TORCH_HOME`, and `HF_HOME` to an E-drive directory before
first use so model caches stay outside the repository and C drive.

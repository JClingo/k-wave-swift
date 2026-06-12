# Performance benchmarks: k-wave-swift vs k-wave-python

Paired benchmark harness. The same scenarios (grid, steps, physics) run on both
implementations and `compare.py` joins the results.

| component | what it does |
|---|---|
| `Sources/KWaveBench/main.swift` | Swift bench executable (`kwave-bench`), MLX GPU by default |
| `bench_python.py` | Python mirror, k-wave-python NumPy backend (float32), `--backend cpp` optional |
| `compare.py` | joins the two JSON outputs, prints speedup table, `--strict [TOL]` gates regressions |
| `run_bench.sh` | builds + runs everything end to end |

## Run

```bash
Scripts/bench/run_bench.sh                  # full suite, 3 repeats
Scripts/bench/run_bench.sh --filter kspace3d --repeats 5
Scripts/bench/run_bench.sh --strict 0.9     # exit 1 if Swift < 0.9x python anywhere
```

Requires the sibling `../k-wave-python` checkout with a synced venv
(`cd ../k-wave-python && uv sync --python 3.13` — 3.14 has no scipy wheels yet).

The Swift bench must be built with **xcodebuild**, not `swift run`: SwiftPM CLI cannot
bundle MLX's metallib, so a `swift run` binary dies with
"MLX error: Failed to load the default metallib". `run_bench.sh` handles this.

## Scenarios

`kspace2d_{64,128,256,512,1024}` IVP disc + PML 20, 256 steps; `_hetero` (heterogeneous
c/ρ maps), `_absorbing` (power-law), `_source` (tone-burst line source + line sensor
recording p) variants at 256²; `kspace3d_{32,64,128}` IVP ball + PML 10, 128 steps;
`angular_spectrum_128` (128×128×64 plane → 32 z-planes); `afp_64` (acousticFieldPropagator
64³ — k-wave-python has none, so the NumPy reference from `Scripts/parity` is timed instead).

Timing covers the solver call only (one warmup run excluded — first MLX call compiles
Metal kernels). Both sides use float32. Scenario definitions must stay in sync between
`main.swift` and `bench_python.py`; `compare.py` skips names present on one side only.

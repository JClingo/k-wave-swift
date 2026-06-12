# KWaveSwift

A Swift port of the [k-Wave](http://www.k-wave.org/) acoustic simulation toolbox, providing
time-domain simulation of acoustic wave propagation using the k-space pseudospectral method.

Supports 1D, 2D, and 3D simulations of linear and nonlinear propagation in heterogeneous media
with power-law absorption. It is a clean-room reimplementation of the
[MATLAB k-Wave toolbox](https://github.com/ucl-bug/k-wave) (v1.4.1), targeting Apple silicon and
built on [MLX-Swift](https://github.com/ml-explore/mlx-swift) for GPU-accelerated array compute.

## Why Swift + MLX

- **One code path, CPU or GPU.** The solver inner loop is written once over `MLXArray` and runs
  on the Metal GPU (float32) or the CPU stream (float32/float64) by selecting a device — no
  CUDA, no C++ kernels to author.
- **`MLXFFT`** provides the n-dimensional FFTs that are the spectral core of the k-space
  pseudospectral method.
- **Value semantics** for all configuration types (`KWaveGrid`, `KWaveMedium`, `KWaveSource`,
  `KWaveSensor` are `struct`s), with strong typing in place of MATLAB name-value pairs.
- **Native visualization**: SwiftUI live field monitoring and AVFoundation H.264 movie capture,
  no external plotting stack.

Licensed Apache-2.0 (the upstream MATLAB/Python implementations are LGPL-3.0; this is a
clean-room port, not a derivative).

## Requirements

| Requirement | Notes |
|---|---|
| Apple silicon Mac | MLX requires Metal / arm64 |
| macOS 14+ | Package platform floor |
| Swift 6.0+ / Xcode | Tests and executables must build via `xcodebuild` (see below) |
| HDF5 | `brew install hdf5` — linked as a system library (`CHDF5` target) |
| Metal Toolchain | One-time: `xcodebuild -downloadComponent MetalToolchain` |

> **Build note:** MLX's Metal shaders cannot be bundled by the SwiftPM CLI. `swift test` /
> `swift run` binaries fail at runtime with *"MLX error: Failed to load the default metallib"*.
> Always build, test, and run through `xcodebuild` (or Xcode itself).

## Quick start

Add the package as a dependency and import `KWave`:

```swift
import KWave
import MLX

// 128 × 128 grid, 0.1 mm spacing
var grid = KWaveGrid(nx: 128, dx: 1e-4, ny: 128, dy: 1e-4)
grid.makeTime(soundSpeedMax: 1500, cfl: 0.3, tEnd: 4e-6)

// Homogeneous water-like medium
let medium = KWaveMedium(soundSpeed: 1500, density: 1000)

// Initial pressure: a disc in the centre (photoacoustic-style IVP)
var source = KWaveSource()
source.p0 = makeDisc(nx: 128, ny: 128, radius: 5) * 1.0

// Record the pressure time series on a sensor mask
var sensor = KWaveSensor()
sensor.mask = makeCircle(nx: 128, ny: 128, radius: 40)
sensor.record = [.p, .pFinal, .pMax]

let output = kspaceFirstOrder(grid: grid, medium: medium, source: source, sensor: sensor)
// output.p      → [numSensorPoints, nt] time series
// output.pFinal → final whole-grid pressure field
```

`kspaceFirstOrder` is the single entry point for 1D/2D/3D — it branches on `grid.dim`, mirroring
the `kspaceFirstOrder1D/2D/3D` family in MATLAB.

### Options

`SimulationOptions` mirrors MATLAB's name-value pairs:

```swift
var opts = SimulationOptions()
opts.pmlSize = .uniform(20)         // PML thickness (default 20)
opts.smoothP0 = true                // smooth the initial pressure (default true)
opts.dtype = .float64               // float64 → CPU stream, tightest parity
opts.device = .cpu
opts.progress = { step, nt in print("\(step)/\(nt)") }
opts.recordMovie = "sim.mp4"        // H.264 capture of the live pressure field
opts.fieldMonitor = { step, p in /* live field hook (drives the SwiftUI monitor) */ }

let out = kspaceFirstOrder(grid: grid, medium: medium, source: source,
                           sensor: sensor, options: opts)
```

## Feature overview

### Solvers

| Solver | Function | Status |
|---|---|---|
| Fluid PSTD, 1D/2D/3D | `kspaceFirstOrder` | ✅ Linear + nonlinear (B/A), heterogeneous c₀/ρ₀, power-law & Stokes absorption, all source modes (dirichlet / additive / additive-no-correction), pressure + velocity sources |
| Axisymmetric | `kspaceFirstOrderAS` | ✅ WSWA-FFT variant: heterogeneous media, p0 + time-varying pressure & velocity sources, Stokes absorption, B/A nonlinearity |
| Time-reversal reconstruction | `timeReversal` | ✅ Photoacoustic reconstruction via Dirichlet re-injection |
| CW Green's-function propagator | `acousticFieldPropagator` | ✅ 1D/2D/3D, ramped + unramped |
| Angular spectrum | `angularSpectrum`, `angularSpectrumCW` | ✅ Broadband + CW, power-law absorption, angular restriction, grid expansion, forward + reverse projection |
| FFT reconstruction | `kspaceLineRecon` (2D), `kspacePlaneRecon` (3D) | ✅ Parity-tested |
| Elastic (`pstdElastic2D/3D`), thermal (`kWaveDiffusion`) | — | ❌ Not started (Phase 4) |

### Sensor recording

Driven by the `RecordField` option set: `p`, `pMax`, `pMin`, `pRms`, `pFinal`, collocated
`ux/uy/uz` time series, per-component `uMax`/`uRms`, `uFinal`, and time-averaged intensity
`iAvg`. 2D sensor directivity (pressure + gradient patterns) is supported. Cartesian
(off-grid) sensor masks record via multilinear interpolation.

### Transducers and arrays

- **`KWaveArray`** — off-grid source/sensor arrays via band-limited interpolant (`offGridPoints`,
  exact + truncated `tolStar`): arc/line/rect/disc elements in 2D; disc, focused-bowl, annulus,
  and annular-array elements in 3D; grid weights, binary masks, distributed source signals,
  `combineSensorData`, and affine array transforms (2D + 3D Euler).
- **`KWaveTransducer`** — MATLAB linear-array transducer model for 3D: element masks, azimuth
  (focus + steering) and elevation beamforming delays, apodization (Hanning/custom), input-signal
  padding, receive combination, and scan-line beamforming. Parity-tested against
  k-wave-python's `NotATransducer`.

### Utilities

| Area | Contents |
|---|---|
| Geometry (grid masks) | `makeDisc`, `makeCircle`, `makeBall`, `makeSphere`, `makeArc` |
| Geometry (Cartesian) | `makeCartCircle`, `makeCartSphere` (golden-section spiral), `makeCartArc`, `makeCartRect` |
| Grid utilities | `cart2grid`, `expandMatrix`, `resize`, `getOptimalPMLSize`, `interpCartData`, `fourierShift` |
| Filtering | `applyFilter`, `gaussianFilter`, `smooth` (1D/2D/3D), `getWin`, `filterTimeSeries` |
| Signals | `toneBurst`, `gaussian`, `spect`, `extractAmpPhase` |
| Materials | `waterSoundSpeed`, `waterDensity`, `waterAbsorption`, `waterNonlinearity`, `fitPowerLawParams`, `powerLawKramersKronig`, `db2neper`/`neper2db` |
| Spectral ops | Spectral + staggered derivatives, DTT foundation (`dct`/`dst` types I–IV) |
| Analytic references | `focusedBowlONeil` |
| I/O | HDF5 read/write compatible with the k-Wave C++/CUDA binary file format |
| Visualization | `getColorMap`, `FieldImage`, `SimulationMonitorView` (SwiftUI live monitor), `MovieRecorder` (H.264) |

For the authoritative per-feature status map (including known gaps), see
[REQUIREMENTS.md](REQUIREMENTS.md) §1.2.

## Precision policy

Metal GPUs have no `double` support, so:

- **Default: float32 on GPU** — fast, matches the precedent set by
  [jwave](https://github.com/ucl-bug/jwave).
- **Validation: float64 on the CPU stream** (`opts.dtype = .float64; opts.device = .cpu`) — used
  for the tightest numerical parity against the reference implementations.
- Parity tests document explicit tolerances per precision.

## Testing

```bash
./Scripts/test.sh            # wraps: xcodebuild test -scheme KWave-Package -destination 'platform=OS X'
```

The suite (~30 test files) covers units (FFT, grid, I/O, geometry, solver behaviour) and an
extensive **parity suite** comparing Swift output against reference data generated from:

- **k-wave-python's NumPy solver** — the formulation the Swift solver mirrors 1:1
  (heterogeneous, absorption, nonlinear, recording, transducer, array, …)
- **the k-Wave C++/OMP engine** — gold standard for homogeneous cases
- **independent NumPy ports of MATLAB formulas** where no Python equivalent exists
  (axisymmetric WSWA branch, `acousticFieldPropagator`, directivity)

Reference `.h5` files are generated by the scripts in `Scripts/parity/` (they require the
sibling `../k-wave-python` checkout with a synced venv) and are not tracked in git — parity
tests `XCTSkip` when references are absent, so the unit suite still runs clean on a fresh clone.

Note: the C++ engine and the NumPy solver diverge ~2% in *heterogeneous* media (a C++ internal
detail); heterogeneous references therefore come from the NumPy solver and homogeneous ones
from C++. See `Scripts/parity/diag_numpy_vs_cpp.py`.

## Benchmarks

A paired benchmark harness runs identical scenarios on Swift (MLX GPU) and k-wave-python
(NumPy, optional C++ backend) and joins the results:

```bash
Scripts/bench/run_bench.sh                       # full suite, 3 repeats
Scripts/bench/run_bench.sh --filter kspace3d --repeats 5
Scripts/bench/run_bench.sh --strict 0.9          # CI gate: fail if Swift < 0.9× python anywhere
```

Scenarios cover 2D IVP grids from 64² to 1024² (plus heterogeneous / absorbing / source
variants), 3D grids up to 128³, angular spectrum, and `acousticFieldPropagator`. See
[Scripts/bench/README.md](Scripts/bench/README.md).

## Continuous integration

GitHub Actions ([tests.yml](.github/workflows/tests.yml)) runs on every push/PR on an Apple
silicon `macos-15` runner: installs HDF5, runs the full test suite via `Scripts/test.sh`, and
smoke-runs the benchmark executable.

## Project layout

```
Sources/KWave/
  Grid/            KWaveGrid (1D/2D/3D, wavenumbers, makeTime), grid utilities
  Medium/          KWaveMedium (c0, rho0, absorption, B/A)
  Source/          KWaveSource (p0, time-varying p/u sources, source modes)
  Sensor/          KWaveSensor, RecordField, recording post-processing, directivity
  Solver/          kspaceFirstOrder (1D/2D/3D), kspaceFirstOrderAS, timeReversal,
                   angularSpectrum(CW), acousticFieldPropagator
  Array/           KWaveArray off-grid elements, offGridPoints (BLI)
  Transducer/      KWaveTransducer linear-array model
  Reconstruction/  kspaceLineRecon, kspacePlaneRecon
  PML/             PML profiles, getOptimalPMLSize
  FFT/             MLXFFT wrappers, spectral/staggered derivatives, DTT (dct/dst I–IV)
  Geometry/        Grid-mask and Cartesian shape generators
  Filter/ Signal/  Filtering, windows, tone bursts, spectra
  Material/        Water properties, power-law fitting, unit conversions
  Interp/          interpCartData, cart2grid
  IO/              HDF5 (k-Wave C++ file-format compatible)
  Viz/             Colormaps, field images, SwiftUI monitor, movie recorder
  Analytic/        O'Neil focused-bowl solution
Sources/KWaveBench/  kwave-bench executable (paired perf benchmarks)
Tests/KWaveTests/    Unit + parity suite
Scripts/parity/      Reference-data generators (k-wave-python / C++ / NumPy ports)
Scripts/bench/       Benchmark harness (Swift vs Python)
```

## Status

Phases 1–2 (foundation + full fluid solver) are complete; Phase 3 (extended features) is
largely complete — axisymmetric solver, CW propagation, KWaveArray, KWaveTransducer,
FFT reconstruction, and material properties are done, with some geometry helpers, signal
processing utilities, the CLI, and the dashboard still open. Phase 4 (elastic/thermal solvers,
3D visualization) has not started. [REQUIREMENTS.md](REQUIREMENTS.md) is the detailed,
up-to-date roadmap and status map.

## References

- [k-Wave (MATLAB)](https://github.com/ucl-bug/k-wave) — the original toolbox (v1.4.1)
- [k-wave-python](https://github.com/waltsims/k-wave-python) — primary porting reference
- [jwave](https://github.com/ucl-bug/jwave) — JAX port; design analog for the array-framework approach
- [MLX-Swift](https://github.com/ml-explore/mlx-swift) — Apple silicon array compute

## License

Apache-2.0.

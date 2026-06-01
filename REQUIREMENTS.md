# k-wave-swift: Swift Port of k-Wave — Requirements Document

## 1. Project Overview

**Goal:** Port the k-Wave acoustic simulation toolbox from MATLAB to Swift, providing both a 1:1
functional match with the MATLAB original and an improved UI/UX for production workflows.

**Source references:**
- MATLAB original: https://github.com/ucl-bug/k-wave (v1.4.1, LGPL-3.0)
- Python port: https://github.com/waltsims/k-wave-python (v0.6.1, LGPL-3.0)
- Functional reference: jwave (JAX) — the design analog for an array-framework/autodiff port

**License:** Apache-2.0 (clean-room port; see §14.7)
**Swift version:** ≥ 6.0
**Platform:** Apple silicon (macOS). See §14.6.

**Why Swift + MLX:**
- [MLX-Swift](https://github.com/ml-explore/mlx-swift) gives a JAX/jwave-style array framework
  native to Apple silicon: `MLXArray` n-d arrays, lazy evaluation, `compile`, and unified
  CPU/Metal-GPU execution from one code path.
- `MLXFFT` wraps n-dimensional FFTs (`fft`, `fft2`, `fftn`, `rfft`, `ifft`) — the spectral core
  of the k-space pseudospectral method.
- Metal GPU acceleration with no CUDA/C/C++ to author; the solver inner loop is written once over
  `MLXArray` and runs on CPU or GPU by selecting a `Device`/`StreamOrDevice`.
- Strong typing + value semantics for the configuration structs (grid/medium/source/sensor).
- Swift Package Manager for distribution; DocC for documentation; SwiftUI + Swift Charts +
  Metal for visualization.
- Accelerate/`vDSP` available as a fallback for any primitive MLX lacks.

**k-Wave** is a toolbox for time-domain simulation of acoustic wave propagation using the k-space
pseudospectral method. It supports 1D, 2D, and 3D simulations of linear/nonlinear propagation in
heterogeneous media with power-law absorption.

### 1.1 Floating-point precision (critical design constraint)

Metal GPUs have **no `double` support**, so MLX `float64` runs **only on the CPU stream**;
`float64` on the GPU raises an exception. MATLAB k-Wave is `float64` throughout, so exact
machine-precision parity is impossible on GPU. The port follows jwave's precedent (float32 by
default) with an explicit precision strategy:

- Solver precision is a configuration option (`dtype: DType`), **default `.float32`** for GPU speed.
- A **`.float64` + CPU-stream** mode is available for tightest numerical validation.
- Parity tests document explicit tolerances: tight for float64/CPU, looser for float32/GPU.

This precision policy is part of the acceptance criteria (§13).

---

## 1.2 Current Implementation Status (as of 2026-05-29)

This section is the ground-truth status map for agents continuing this project. It reflects the
actual code in `Sources/KWave/`, not the commit message ("Initial pass at 1-to-1 feature parity"
is aspirational; significant gaps remain).

### Phase 1 — Foundation ✅ (mostly complete)

**Implemented and validated:**

| Component | File | Notes |
|---|---|---|
| `KWaveGrid` (1D/2D/3D) | `Grid/KWaveGrid.swift` | `makeTime`, `setTime`, wavenumber grids |
| `KWaveMedium` | `Medium/KWaveMedium.swift` | HEAD version; see regression warning above |
| `KWaveSource` | `Source/KWaveSource.swift` | All fields including stress (elastic) |
| `KWaveSensor` + `RecordField` | `Sensor/KWaveSensor.swift` | All RecordField cases defined |
| `kspaceFirstOrder2D` | `Solver/KSpaceFirstOrder.swift` | See limitations below |
| `kspaceFirstOrder3D` | `Solver/KSpaceFirstOrder.swift` | See limitations below |
| PML (profile + vectorToColumn) | `PML/PML.swift` | |
| FFT utils + staggered derivative | `FFT/FFTUtils.swift` | `spectralDerivative`, `StaggeredDerivative` |
| `makeDisc`, `makeCircle` | `Geometry/Shapes.swift` | 2D grid binary masks |
| `makeBall`, `makeSphere` | `Geometry/Shapes.swift` | 3D grid binary masks |
| `expandMatrix`, `cart2grid2D`, `grid2cart2D` | `Grid/GridUtils.swift` | |
| `applyFilter`, `gaussianFilter`, `smooth`, `smooth3D`, `getWin`, `filterTimeSeries` | `Filter/Filters.swift` | getWin: Hann + Blackman only (not Tukey/Nuttall) |
| `toneBurst`, `gaussian` | `Signal/Generation.swift` | |
| `db2neper`, `neper2db` | `Material/Conversion.swift` | |
| HDF5 I/O | `IO/HDF5.swift`, `IO/KWaveInput.swift` | |
| `getColorMap` | `Viz/ColorMap.swift` | |

**Phase 1 gaps (items in §12 Phase 1 list not yet done):**

| Item | Status |
|---|---|
| `kspaceFirstOrder1D` | ✅ Implemented (linear, lossless; mirrors 2D/3D) |
| Heterogeneous media in solver | ✅ Implemented (spatially varying c0/rho0 in 1D/2D/3D; staggered density, `c_ref=max(c0)`) |
| `resize` (interpolated resizing) | ✅ Implemented (`Grid/GridUtils.swift`, separable linear interp 1D/2D/3D) |
| `getOptimalPMLSize` | ✅ Implemented (`PML/PML.swift`, prime-factor heuristic, returns `[Int]` per dim) |
| `interpCartData` (nearest), `fourierShift` | ✅ Implemented (`Interp/InterpCartData.swift`, `Signal/FourierShift.swift`); verified vs NumPy (`ParityCartesianTests`) |
| Cartesian (off-grid) sensor masks | ✅ Implemented (`[dim, N]` point masks, multilinear interp recording via `SensorSampler`; 1D/2D/3D) |
| `findClosest`, `offGridPoints`, `interpCartData` linear mode | Not implemented |
| `SimulationOptions.progress` callback | ✅ Added; fires once per step in all solvers |
| `SimulationOptions.smoothC0`, `smoothRho0` | ✅ Implemented and wired (smooth spatially varying c0/rho0 when set) |

**Solver limitations (2D and 3D):**
- Linear and lossless (homogeneous or spatially varying sound speed / density)
- `SimulationOutput` only returns `p` (time series) and `pFinal`; all other `RecordField` cases
  (pMax, pMin, pRms, ux, uy, uz, uMax, uRms, uFinal, iAvg, iMax) are **not recorded**
- ✅ Sensor directivity (2D pressure + gradient)
- Time-reversal mode not implemented
- Absorption (power-law) not implemented
- Nonlinear propagation (B/A) not implemented

### Phase 2 — Full Fluid Solver 🚧 (in progress)

- ✅ Heterogeneous media support in solver (spatially varying c0/rho0, 1D/2D/3D)
- ✅ Power-law absorption and dispersion (power-law + Stokes; 1D/2D/3D)
- ✅ Nonlinear propagation (B/A; 1D/2D/3D)
- ✅ Time-reversal reconstruction (`timeReversal` helper)
- ✅ Sensor recording fields beyond `p`/`pFinal` (pMax/pMin/pRms, collocated ux/uy/uz, uMax/uRms, uFinal, iAvg)
- ✅ Sensor directivity (2D, pressure + gradient patterns)
- Real-time monitoring UI (Metal)
- Movie recording (AVFoundation)
- Full parity test suite

### Phase 3 — Extended Features ❌ (not started)

All of Phase 3 is unimplemented. Missing files and their target locations:

| Missing | Target file | Exports |
|---|---|---|
| Cartesian geometry | `Geometry/Cartesian.swift` | `makeCartCircle`, `makeCartSphere`, `makeCartArc`, `makeCartBowl`, `makeCartDisc`, `makeCartRect` |
| Signal processing | `Signal/Processing.swift` | `addNoise`, `createCWSignals`, `extractAmpPhase`, `logCompression`, `envelopeDetection`, `gradientFD`, `gradientSpect`, `spect` |
| Material properties | `Material/Properties.swift` | `waterSoundSpeed`, `waterDensity`, `waterAbsorption`, `waterNonlinearity`, `fitPowerLawParams` |
| KWaveArray | `Array/KWaveArray.swift` | `KWaveArray`, `ArcElement`, `BowlElement`, `DiscElement`, `RectElement`, `SphereElement`, `addArcElement!`, `getElementBinaryMask`, `getDistributedSourceSignal`, `combineSensorData` |
| KWaveTransducer | `Transducer/KWaveTransducer.swift` | `KWaveTransducer`, `getTransducerBinaryMask`, `getTransducerSource`, `combineTransducerSensorData` |
| Geometry shapes (grid) | `Geometry/Shapes.swift` (add to existing) | `makeArc`, `makeLine`, `makeBowl`, `makeMultiArc`, `makeMultiBowl`, `makeSphericalSection` |
| FFT reconstruction | `Reconstruction/FFTRecon.swift` | `kspaceLineRecon`, `kspacePlaneRecon` |
| Axisymmetric solver | `Solver/Axisymmetric.swift` | `kspaceFirstOrderAS` |
| CW solver | `Solver/CW.swift` | `acousticFieldPropagator`, `angularSpectrumCW` |
| Visualization | `Viz/FieldDisplay.swift`, `Viz/Plots.swift` | `SimulationDisplay`, `beamPlot`, `flyThrough`, `overlayPlot`, `stackedPlot` |
| CLI | `kwave-cli/main.swift` | `kwave-cli run/validate/info` |

### Phase 4 — Advanced Solvers ❌ (not started)

| Missing | Target file | Exports |
|---|---|---|
| Elastic wave solver | `Solver/Elastic.swift` | `ElasticMedium`, `ElasticSource`, `pstdElastic2D`, `pstdElastic3D` |
| Thermal/bioheat solver | `Solver/Diffusion.swift` | `ThermalMedium`, `ThermalSource`, `kwaveDiffusion`, `bioheatExact` |
| Analytical reference solutions | `Reference/Analytical.swift` | `focusedAnnulusONeil`, `focusedBowlONeil`, `mendousse` |
| Beamforming reconstruction | `Reconstruction/Beamform.swift` | `beamformDelayAndSum`, `scanConversion` |
| 3D visualization | `Viz/Voxel.swift` | `voxelPlot`, `isosurfacePlot`, `maxIntensityProjection` |

### Reference implementation

**The k-wave-python source (`/Users/jingo/Work/Attune/k-wave-python/`) is the direct source
reference for all modules.** It is the most mature port (matches the MATLAB original closely),
the project ships a working install in `.venv-kwave/`, and its pure-NumPy solver
(`kwave/solvers/kspace_solver.py`) is the formulation the Swift solver mirrors 1:1.

When implementing or validating a module, map the Swift target to its k-wave-python counterpart:

| Swift area | k-wave-python source |
|---|---|
| Fluid solver internals (update equations, p0 init, staggered density, source scaling) | `kwave/solvers/kspace_solver.py` |
| Solver setup (staggered-grid medium, `c_ref`, kappa/PML operators) | `kwave/kspaceFirstOrder{2D,3D,AS}.py`, `kwave/kWaveSimulation*.py` |
| Grid / wavenumbers / `makeTime` | `kwave/kgrid.py` |
| Medium / source / sensor | `kwave/kmedium.py`, `kwave/ksource.py`, `kwave/ksensor.py` |
| Geometry, filters, signals, interp, PML, conversions | `kwave/utils/*.py` (e.g. `mapgen.py`, `filters.py`, `interp.py`, `pml.py`) |
| Reconstruction | `kwave/kspaceLineRecon.py`, `kwave/kspacePlaneRecon.py`, `kwave/reconstruction/` |

Porting steps:
1. Read the corresponding k-wave-python source above.
2. Port to Swift using `MLXArray` in place of NumPy arrays.
3. Follow the existing Swift patterns (value types, `MLXArray` for all numeric fields).

**Parity note (see §9.2):** the C++/OMP engine is the gold standard and homogeneous references use
it. But the C++ engine and k-wave-python's NumPy solver diverge ~2% in *heterogeneous* media (a C++
internal detail the NumPy solver doesn't reproduce). Since the Swift solver mirrors the NumPy
formulation, heterogeneous parity references are generated from the NumPy solver, homogeneous ones
from C++. See `Scripts/parity/generate_reference_hetero.py` / `diag_numpy_vs_cpp.py`.

---

## 2. Core Simulation Engine

### 2.1 Fluid Simulations (k-space pseudospectral method)

| Function (MATLAB) | Description | Priority |
|---|---|---|
| `kspaceFirstOrder1D` | 1D time-domain acoustic simulation | P0 |
| `kspaceFirstOrder2D` | 2D time-domain acoustic simulation | P0 |
| `kspaceFirstOrder3D` | 3D time-domain acoustic simulation | P0 |
| `kspaceFirstOrderAS` | Axisymmetric simulation | P1 |
| `kspaceSecondOrder` | Fast homogeneous-media simulation | P2 |

**Requirements:**
- k-space corrected pseudospectral time-domain (PSTD) solver
- FFT-based spatial gradient computation (via `MLXFFT`)
- Perfectly Matched Layer (PML) absorbing boundary conditions
- Support for heterogeneous sound speed and density fields
- Power-law acoustic absorption and dispersion
- Nonlinear propagation via B/A parameter
- Time-reversal mode for photoacoustic image reconstruction
- Staggered grid implementation for velocity/pressure fields
- CFL-based automatic time step calculation

**Swift design (dimension specialization):** Swift has no multiple dispatch, so the grid is a
single `KWaveGrid` value type carrying optional `nx/ny/nz` and a `dim` count (mirroring
`k-wave-python`'s `kWaveGrid`). The solver entry point branches on `grid.dim`; dimension-specific
internal steps (gradient, PML, source injection) are separate functions selected by dimension.

```swift
func kspaceFirstOrder(
    grid: KWaveGrid,
    medium: KWaveMedium,
    source: KWaveSource,
    sensor: KWaveSensor,
    options: SimulationOptions = .init()
) -> SimulationOutput
```

### 2.2 Elastic Simulations

| Function (MATLAB) | Description | Priority |
|---|---|---|
| `pstdElastic2D` | 2D elastic (shear + compressional) wave propagation | P2 |
| `pstdElastic3D` | 3D elastic wave propagation | P2 |

**Requirements:** viscoelastic absorption model; coupled compressional and shear wave fields;
stress tensor source support.

### 2.3 Continuous Wave (CW) Propagation

| Function (MATLAB) | Description | Priority |
|---|---|---|
| `acousticFieldPropagator` | CW acoustic field computation | P1 |
| `angularSpectrumCW` | CW angular spectrum projection | P1 |

### 2.4 Thermal Simulations

| Function (MATLAB) | Description | Priority |
|---|---|---|
| `kWaveDiffusion` | Pennes' bioheat equation solver (1D/2D/3D) | P2 |
| `bioheatExact` | Analytical solution for homogeneous bioheat | P2 |

**Requirements:** diffusion + perfusion + metabolic heat generation; coupling with acoustic
absorption for HIFU heating studies.

---

## 3. Data Structures

Swift value types (`struct`) with defaulted stored properties and memberwise/custom inits replace
MATLAB name-value pairs and k-wave-python's dataclass/keyword structs. Numeric fields are
`MLXArray` (which carries its dtype at runtime) or `Double`/`Int` for scalars.

### 3.1 `KWaveGrid`

The computational grid — the central data structure. A single type for all dimensions.

```swift
struct KWaveGrid {
    let dim: Int                       // 1, 2, or 3
    let nx: Int, ny: Int, nz: Int      // ny/nz == 1 for lower dimensions
    let dx: Double, dy: Double, dz: Double

    // Spatial and wavenumber grids (precomputed MLXArray, broadcast-shaped)
    let kx: MLXArray, ky: MLXArray, kz: MLXArray
    let k: MLXArray

    // Time
    private(set) var dt: Double = 0
    private(set) var nt: Int = 0
    private(set) var tArray: MLXArray = []
}
```

**Constructors and methods:**
```swift
KWaveGrid(nx:dx:)                              // 1D
KWaveGrid(nx:dx:ny:dy:)                        // 2D
KWaveGrid(nx:dx:ny:dy:nz:dz:)                 // 3D
mutating func makeTime(soundSpeed:cfl:tEnd:)  // CFL-based dt/nt/tArray
var totalGridPoints: Int
```

### 3.2 `KWaveMedium`

```swift
struct KWaveMedium {
    var soundSpeed: MLXArray            // scalar or array [m/s]
    var density: MLXArray               // scalar or array [kg/m³]
    var alphaCoeff: MLXArray? = nil     // absorption coefficient
    var alphaPower: Double? = nil       // absorption power-law exponent
    var alphaMode: AbsorptionMode = .noAbsorption  // .noAbsorption, .noDispersion, .stokes
    var bOnA: MLXArray? = nil           // nonlinearity parameter B/A
    // Elastic media
    var soundSpeedCompression: MLXArray? = nil
    var soundSpeedShear: MLXArray? = nil
}
```

### 3.3 `KWaveSource`

```swift
struct KWaveSource {
    // Initial value (photoacoustic)
    var p0: MLXArray? = nil

    // Time-varying pressure source
    var pMask: MLXArray? = nil          // boolean mask
    var p: MLXArray? = nil
    var pMode: SourceMode = .additive

    // Time-varying velocity source
    var uMask: MLXArray? = nil
    var ux: MLXArray? = nil
    var uy: MLXArray? = nil
    var uz: MLXArray? = nil
    var uMode: SourceMode = .additive

    // Stress source (elastic)
    var sMask: MLXArray? = nil
    var sxx, syy, szz, sxy, sxz, syz: MLXArray?
}

enum SourceMode { case dirichlet, additive, additiveNoCorrection }
```

### 3.4 `KWaveSensor`

```swift
struct KWaveSensor {
    var mask: MLXArray? = nil
    var record: RecordField = [.p]      // OptionSet
    var timeReversalBoundaryData: MLXArray? = nil
    var directivityAngle: MLXArray? = nil
    var directivitySize: Double? = nil
    var frequencyResponse: (centerFreq: Double, bandwidth: Double)? = nil
}

struct RecordField: OptionSet {
    // .p, .pMax, .pMin, .pRms, .pFinal,
    // .ux, .uy, .uz, .uMax, .uRms, .uFinal, .iAvg, .iMax
}
```

### 3.5 `KWaveArray`

Off-grid transducer array modeling.

```swift
struct KWaveArray { var elements: [ArrayElement] }

enum ArrayElement { case arc(...), bowl(...), disc(...), rect(...), sphere(...) }

// Methods
mutating func addArcElement(position:radius:diameter:focus:)
mutating func addBowlElement(position:radius:diameter:focus:)
mutating func addDiscElement(position:diameter:focus:)
mutating func addRectElement(position:width:height:focus:)
func elementBinaryMask(grid:elementIndex:) -> MLXArray
func distributedSourceSignal(grid:sourceSignal:) -> MLXArray
func combineSensorData(grid:sensorData:) -> MLXArray
```

### 3.6 `KWaveTransducer`

Linear array transducer model (for 3D simulations).

```swift
struct KWaveTransducer {
    var numberElements: Int
    var elementWidth: Int               // grid points
    var elementLength: Int              // grid points
    var elementSpacing: Int = 0         // grid points (kerf)
    var position: (Int, Int, Int) = (1, 1, 1)
    var radius: Double = .infinity      // focus radius (Inf = flat)
    var focusDistance: Double = .infinity
    var steeringAngle: Double = 0
    var transmitApodization: Apodization = .rectangular
    var receiveApodization: Apodization = .rectangular
    var activeElements: [Int]? = nil
    var inputSignal: MLXArray? = nil
}
```

---

## 4. Utility Functions

### 4.1 Geometry / Shape Creation (Binary Masks)

All return binary masks (`MLXArray`) on a grid:

| Function | Description | Priority |
|---|---|---|
| `makeDisc` | Filled circle in 2D | P0 |
| `makeCircle` | Circle perimeter in 2D | P0 |
| `makeBall` | Filled sphere in 3D | P0 |
| `makeSphere` | Sphere surface in 3D | P0 |
| `makeArc` | Arc in 2D | P1 |
| `makeBowl` | Bowl surface in 3D | P1 |
| `makeLine` | Line segment in 2D/3D | P1 |
| `makeSphericalSection` | Spherical cap in 3D | P1 |
| `makeMultiArc` | Multiple arcs | P1 |
| `makeMultiBowl` | Multiple bowls | P1 |

### 4.2 Cartesian Geometry (Off-grid Point Distributions)

| Function | Description | Priority |
|---|---|---|
| `makeCartCircle` | Cartesian circle points | P1 |
| `makeCartSphere` | Cartesian sphere points | P1 |
| `makeCartArc` | Cartesian arc points | P1 |
| `makeCartBowl` | Cartesian bowl points | P1 |
| `makeCartDisc` | Cartesian disc points | P1 |
| `makeCartRect` | Cartesian rectangle points | P1 |

### 4.3 Grid and Matrix Utilities

| Function | Description | Priority |
|---|---|---|
| `cart2grid` | Map Cartesian points onto grid | P0 |
| `grid2cart` | Extract Cartesian coords from grid mask | P0 |
| `expandMatrix` | Expand matrix with boundary conditions | P0 |
| `resize` | Resize matrix (interpolation) | P0 |
| `getOptimalPMLSize` | Compute optimal PML thickness | P0 |
| `interpCartData` | Interpolate Cartesian sensor data | P1 |
| `fourierShift` | Sub-pixel shift via Fourier method | P1 |
| `findClosest` | Find nearest grid point | P1 |
| `offGridPoints` | Off-grid source/sensor interpolation | P1 |

### 4.4 Filtering and Spectral

| Function | Description | Priority |
|---|---|---|
| `applyFilter` | Apply frequency-domain filter | P0 |
| `gaussianFilter` | Apply Gaussian frequency filter | P0 |
| `smooth` | Smooth matrix (N-D) | P0 |
| `getWin` | Generate window functions (Hann, Blackman, etc.) | P0 |
| `filterTimeSeries` | Causal/zero-phase FIR filter | P1 |
| `spect` | Compute amplitude/phase spectrum | P1 |
| `envelopeDetection` | Hilbert transform envelope | P1 |
| `gradientFD` | Finite-difference gradient | P1 |
| `gradientSpect` | Spectral gradient | P1 |

### 4.5 Signal Creation and Processing

| Function | Description | Priority |
|---|---|---|
| `toneBurst` | Generate tone burst signals | P0 |
| `gaussian` | Generate Gaussian pulse | P0 |
| `addNoise` | Add noise at specified SNR | P1 |
| `createCWSignals` | Generate CW source signals | P1 |
| `extractAmpPhase` | Extract amplitude and phase from CW | P1 |
| `logCompression` | Log-compress signal for display | P1 |
| `scanConversion` | Polar to Cartesian scan conversion | P2 |

### 4.6 Absorption and Material Properties

| Function | Description | Priority |
|---|---|---|
| `db2neper` / `neper2db` | Unit conversion | P0 |
| `fitPowerLawParams` | Fit absorption to power law | P1 |
| `waterSoundSpeed` | Temperature-dependent water c | P1 |
| `waterDensity` | Temperature-dependent water rho | P1 |
| `waterAbsorption` | Temperature-dependent water alpha | P1 |
| `waterNonlinearity` | Temperature-dependent water B/A | P1 |
| `hounsfield2density` | CT Hounsfield to density conversion | P2 |
| `attenComp` | Time-domain attenuation compensation | P2 |

### 4.7 Reconstruction

| Function | Description | Priority |
|---|---|---|
| `kspaceLineRecon` | 1D FFT-based reconstruction | P1 |
| `kspacePlaneRecon` | 2D FFT-based reconstruction | P1 |
| Time-reversal (built into solver) | Via `sensor.timeReversalBoundaryData` | P0 |
| Beamforming reconstruction | Delay-and-sum beamforming | P2 |

### 4.8 Reference/Analytical Solutions

| Function | Description | Priority |
|---|---|---|
| `focusedAnnulusONeil` | O'Neil solution for focused annulus | P2 |
| `focusedBowlONeil` | O'Neil solution for focused bowl | P2 |
| `mendousse` | Mendousse solution (nonlinear 1D) | P2 |

---

## 5. I/O and File Formats

### 5.1 HDF5 Support (Critical)

| Capability | Description | Priority |
|---|---|---|
| Write simulation input to HDF5 | Grid, medium, source, sensor, PML params | P0 |
| Read simulation output from HDF5 | Sensor data, field maxima/RMS | P0 |
| `writeMatrix` | Write array to HDF5 dataset | P0 |
| `writeGrid` | Write grid + PML properties | P0 |
| `writeFlags` | Write simulation flags | P0 |
| `writeAttributes` | Write file-level metadata | P0 |
| `h5Compare` | Compare two HDF5 files | P1 |

**Implementation:** There is no native Swift HDF5 framework. Link the system `libhdf5`
(Homebrew `hdf5` at `/opt/homebrew/opt/hdf5`) via a SwiftPM system-library target plus a thin
Swift wrapper. This is an external native dependency, not C source we author. The file *format*
must remain compatible with the k-Wave C++/CUDA binaries (same dataset names, attributes, and
data layout). This is the highest-risk dependency — resolve linking early.

### 5.2 Image and Volume I/O

| Capability | Description | Priority |
|---|---|---|
| `loadImage` | Load image as medium map | P1 |
| `saveTiffStack` | Export 3D volume as TIFF stack | P2 |

**Implementation:** Apple `ImageIO` / `CoreGraphics` for image loading; `ImageIO` TIFF support
for stacks.

---

## 6. Visualization / UI

### 6.1 Design Approach

Swift's native UI stack replaces MATLAB plotting:

- **SwiftUI + Swift Charts** — line plots, heatmaps, overlays, publication-quality static export.
- **Metal / `MTKView`** — GPU-accelerated real-time field display for large grids.
- **Headless mode** — all simulations run without a display (server/CI use); plots export to file.

### 6.2 Visualization Functions

| Function (MATLAB) | Swift Implementation | Priority |
|---|---|---|
| Real-time simulation plot | Metal/`MTKView` live field texture | P0 |
| `beamPlot` | Chart heatmap with orthogonal plane slicing | P1 |
| `flyThrough` | Slider-controlled slice viewer (SwiftUI) | P1 |
| `getColorMap` | k-Wave perceptual colormap (lookup table) | P0 |
| `overlayPlot` | Heatmap with alpha overlay | P1 |
| `stackedPlot` | Line series with vertical offsets | P1 |
| `voxelPlot` | Metal volume rendering | P2 |

### 6.3 Simulation Monitoring

During execution, the UI subscribes to per-step field snapshots published by the solver
(`AsyncStream` / Combine). The solver runs off the main actor; rendering happens on the main
actor via Metal. Requirements:

- Real-time 2D field slice display (pressure or velocity components)
- Configurable layout (single field or multiple panels)
- Adjustable color scale (auto, fixed, symmetric)
- Iteration counter and estimated time remaining
- Non-blocking: simulation on a background task, rendering on the main actor
- Movie capture (`AVFoundation` → MP4)

```swift
let stream = kspaceFirstOrder(grid: grid, medium: medium, source: source, sensor: sensor,
    options: .init(plotSim: true, plotScale: .auto, recordMovie: "sim.mp4"))
```

### 6.4 App / Dashboard (P1 monitoring, P3 full editing)

A SwiftUI app for launching and monitoring simulations, viewing and exporting results
(Phase 3), and eventually full parameter editing — grid setup, medium maps, source/sensor
configuration (Phase 4). Useful as a desktop front-end on macOS.

### 6.5 UI Requirements

- macOS (Apple silicon) via SwiftUI/Metal
- GPU-accelerated rendering for large field displays
- Headless mode for CI/server runs
- Plot export: PNG, PDF via `ImageIO` / SwiftUI `ImageRenderer`

---

## 7. Performance Requirements

### 7.1 Compute Backends

MLX's array abstraction lets the solver be written once and dispatched to CPU or GPU by
selecting a `StreamOrDevice`:

| Backend | Implementation | Priority |
|---|---|---|
| **GPU (Metal)** | MLX default device; `MLXArray` on `.gpu`. float32 only. | P0 |
| **CPU** | MLX `.cpu` stream; supports float64 for parity validation. | P0 |
| **Accelerate fallback** | `vDSP`/BLAS for any primitive MLX lacks | P1 |

**Key advantage:** the solver inner loop operates on `MLXArray`. Selecting the device/stream
moves the whole computation between CPU and GPU with no code duplication:

```swift
// GPU (float32, default)
var opts = SimulationOptions()
let result = kspaceFirstOrder(grid: grid, medium: medium, source: source, sensor: sensor, options: opts)

// CPU + float64 (parity validation)
opts.dtype = .float64
opts.device = .cpu
let resultHiPrec = kspaceFirstOrder(grid: grid, medium: medium, source: source, sensor: sensor, options: opts)
```

### 7.2 Memory

- Support grids up to 512³ (float32 ≈ 0.5 GB per field; float64 CPU-only ≈ 1 GB per field)
- Reuse pre-built wavenumber grids across all time steps
- Wrap the per-step kernel with MLX `compile` to fuse ops and cut allocations
- Call `eval()` once per step to bound the lazy graph and control peak memory
- float32 default; opt into float64 (CPU) explicitly

### 7.3 Performance Targets

- MLX GPU solver competitive with k-Wave C++/CUDA binaries on comparable Apple silicon
- MLX CPU solver competitive with MATLAB's pure-MATLAB solver
- `MLXFFT` is the spectral hot path; verify it dominates step time and is not allocation-bound
- Benchmark continuously (XCTest `measure` + a dedicated bench target)

---

## 8. API Design

### 8.1 Package (`KWave`)

```swift
struct SimulationOptions {
    // PML
    var pmlInside: Bool = true
    var pmlSize: PMLSize = .uniform(20)
    var pmlAlpha: Double = 2.0
    // Smoothing
    var smoothP0: Bool = true
    var smoothC0: Bool = false
    var smoothRho0: Bool = false
    // Precision / device
    var dtype: DType = .float32
    var device: DeviceKind = .gpu
    // Display
    var plotSim: Bool = false
    var plotScale: PlotScale = .auto
    var recordMovie: String? = nil
    // Data
    var saveToDisk: String? = nil
    // Callback
    var progress: ((Int, Int) -> Void)? = nil
}

func kspaceFirstOrder(
    grid: KWaveGrid,
    medium: KWaveMedium,
    source: KWaveSource,
    sensor: KWaveSensor,
    options: SimulationOptions = .init()
) -> SimulationOutput
```

### 8.2 Design Principles

- **Dimension branching, not multiple dispatch:** one `kspaceFirstOrder` entry point switching on
  `grid.dim`; dimension-specific internal step functions.
- **Options struct** for all simulation options (mirrors MATLAB name-value pairs)
- **Generic over `MLXArray`:** the solver operates on `MLXArray`, enabling CPU/GPU via device choice
- **Value semantics:** configuration types are `struct`s; mutation is explicit (`mutating func`)
- **Composable:** individual solver steps exposed for advanced users (gradient, PML, source inject)
- **Errors:** throwing initializers/validators with descriptive messages for invalid configs
  (e.g., a 3D source on a 2D grid)

### 8.3 CLI Tool

A command-line executable target (Swift Argument Parser):

```bash
kwave-cli run --input sim.h5 --output results.h5
kwave-cli validate --input sim.h5
kwave-cli info --input sim.h5
```

### 8.4 Interop

Swift's C interoperability already covers `libhdf5`. Broader Python interop is out of scope for
the Apple-only Swift port (P3 if ever needed).

---

## 9. Testing Strategy

### 9.1 Unit Tests

- Every utility function tested independently (XCTest / Swift Testing)
- Property-based checks for math functions (FFT round-trip, gradient accuracy)
- Edge cases: empty grids, scalar media, single-point sensors

### 9.2 MATLAB Parity Tests

**Critical for validation.** Reuse the existing reference infrastructure:

1. Use MATLAB reference outputs from `k-wave-python/tests/reference_outputs/` (HDF5 files)
2. Run identical simulation configurations in Swift
3. Compare within documented tolerance (tight for float64/CPU, looser for float32/GPU per §1.1)
4. Target: match all 47 parity tests from k-wave-python, then expand
5. Load reference data via the `libhdf5` wrapper

### 9.3 Integration Tests

- End-to-end simulation scenarios matching k-Wave examples
- HDF5 round-trip (write input, read output, compare schema with k-Wave C++ binaries)
- Cross-validation with analytical solutions (O'Neil, Mendousse)

### 9.4 Performance Benchmarks

- XCTest `measure` and a dedicated bench target for FFT, gradient, full step
- Comparison against MATLAB and Python solver times
- GPU benchmarks vs existing k-Wave CUDA binaries (on comparable hardware classes)
- Allocation tracking to keep the inner loop near zero-allocation

---

## 10. Project Structure

```
k-wave-swift/
  Package.swift                       # SPM manifest and dependencies
  Sources/
    KWave/
      KWave.swift                     # umbrella exports
      Grid/
        KWaveGrid.swift               # grid type, wavenumber grids, makeTime
        GridUtils.swift               # cart2grid, grid2cart, expandMatrix, resize
      Medium/KWaveMedium.swift
      Source/KWaveSource.swift        # + SourceMode
      Sensor/KWaveSensor.swift        # + RecordField
      Array/KWaveArray.swift          # off-grid transducer arrays
      Transducer/KWaveTransducer.swift
      Solver/
        KSpaceFirstOrder.swift        # entry point (dim branching) + SimulationOptions
        FirstOrderSteps.swift         # time-stepping components
        Axisymmetric.swift            # kspaceFirstOrderAS
        SecondOrder.swift             # kspaceSecondOrder
        Elastic.swift                 # pstdElastic 2D/3D
        Diffusion.swift               # kWaveDiffusion
        CW.swift                      # acousticFieldPropagator, angularSpectrumCW
      PML/PML.swift                   # getPML, getOptimalPMLSize
      FFT/FFTUtils.swift              # MLXFFT wrappers, gradientSpect, fourierShift
      Geometry/
        Shapes.swift                  # makeDisc, makeBall, makeArc, ...
        Cartesian.swift               # makeCartCircle, makeCartBowl, ...
      Signal/
        Generation.swift              # toneBurst, gaussian
        Processing.swift              # envelopeDetection, logCompression, ...
      Filter/Filters.swift            # applyFilter, gaussianFilter, smooth, getWin
      Material/
        Properties.swift              # water*, hounsfield2density
        Conversion.swift              # db2neper, neper2db, fitPowerLawParams
      Reconstruction/
        FFTRecon.swift                # kspaceLineRecon, kspacePlaneRecon
        TimeReversal.swift
        Beamform.swift
      Reference/Analytical.swift      # focusedBowlONeil, mendousse, ...
      IO/
        HDF5.swift                    # HDF5 read/write (k-Wave C++ compatible)
        Images.swift                  # ImageIO loading / TIFF stacks
      Viz/
        ColorMap.swift                # k-Wave colormap LUT
        FieldDisplay.swift            # real-time Metal monitoring
        Plots.swift                   # beamPlot, overlayPlot, stackedPlot, flyThrough
    CHDF5/                            # system-library target wrapping libhdf5
      module.modulemap
    kwave-cli/                        # CLI executable target
      main.swift
  Tests/
    KWaveTests/
      GridTests.swift
      SolverTests.swift
      GeometryTests.swift
      SignalTests.swift
      FilterTests.swift
      IOTests.swift
      Parity/
        ReferenceOutputs/             # HDF5 reference data from k-wave-python
        ParityTests.swift
  Examples/                           # ported k-Wave example scripts
  Documentation.docc/                 # DocC catalog
```

---

## 11. Dependencies

**Core:**
| Package / Framework | Purpose |
|---|---|
| `mlx-swift` (MLX, MLXFFT, MLXRandom) | n-d arrays, FFT, Metal GPU compute |
| `libhdf5` (system, via Homebrew) | HDF5 file I/O (C interop) |
| Accelerate (system) | `vDSP`/BLAS fallback primitives |
| `swift-argument-parser` | CLI argument parsing |

**Visualization (app / optional targets):**
| Framework | Purpose |
|---|---|
| SwiftUI + Swift Charts | Plots and dashboard |
| Metal / MetalKit | Real-time field rendering, voxel volumes |
| AVFoundation | Movie capture |
| ImageIO / CoreGraphics | Image and TIFF I/O |

**Dev/test only:**
| Tool | Purpose |
|---|---|
| XCTest / Swift Testing | Unit, integration, parity tests |
| DocC | API documentation |

---

## 12. Implementation Phases

Status legend: ✅ done · ⚠️ partial · ❌ not started

### Phase 1: Foundation (P0) — ✅ Complete

**Goal:** Minimal working 2D simulation with validation.

1. ✅ `KWaveGrid` (1D/2D/3D construction + wavenumber grids)
2. ✅ `KWaveMedium`, `KWaveSource`, `KWaveSensor` (structs + RecordField)
3. ✅ FFT layer (`MLXFFT` wrappers) + spectral gradient + staggered derivative operators
4. ✅ PML implementation (`pmlProfile`, `vectorToColumn`)
5. ✅ `kspaceFirstOrder` for 1D, 2D and 3D (linear, lossless, **homogeneous only**)
6. ✅ Geometry: `makeDisc`, `makeCircle`, `makeBall`, `makeSphere`
7. ⚠️ Grid utilities: `cart2grid2D`, `grid2cart2D`, `expandMatrix`, `resize`, `getOptimalPMLSize` done; `interpCartData`, `fourierShift`, `findClosest`, `offGridPoints` **missing**
8. ✅ Filtering: `applyFilter`, `gaussianFilter`, `smooth` (1D+2D+3D), `getWin` (Hann+Blackman), `filterTimeSeries`
9. ✅ Signal: `toneBurst`, `gaussian`
10. ✅ HDF5 I/O (k-Wave C++ format compatible)
11. ✅ Unit conversion: `db2neper`, `neper2db`
12. ✅ `getColorMap`
13. ⚠️ Parity test: 2D IVP (homogeneous) exists; requires reference HDF5 generated by `Scripts/parity/generate_reference.py`
14. ✅ Basic test suite: FFT, grid, IO, geometry, solver (1D/2D/3D), parity, util tests all passing (run via `xcodebuild test -scheme KWave -destination 'platform=macOS'`; the `swift test` CLI cannot load the MLX metallib)

**Phase 1 done; the items below moved to Phase 2 / Phase 3 (not blocking):**
- ✅ `kspaceFirstOrder1D` — implemented (1D reduction of the homogeneous 2D/3D solver)
- ✅ `resize` — separable linear interpolation (`Grid/GridUtils.swift`)
- ✅ `getOptimalPMLSize` — prime-factor heuristic returning per-dim sizes (`PML/PML.swift`)
- ✅ `SimulationOptions.progress` callback — fires once per step in 1D/2D/3D
- ✅ `SimulationOptions.smoothC0`, `smoothRho0` — fields added (functional once heterogeneous media lands)

### Phase 2: Full Fluid Solver (P0 continued) — ❌ Not started

1. ✅ Heterogeneous media (varying sound speed + density) — staggered-density velocity update, `dt·ρ0` density update, `c0²·Σρ` EOS, `c_ref=max(c0)`; verified against k-wave-python's NumPy solver (`ParityTests.test2DHeterogeneousParity`)
2. ✅ Power-law absorption and dispersion — EOS gains `+ τ·∇^(y-2)(ρ0·∇·u) − η·∇^(y-1)(ρ)` (power-law) or `+ τ·ρ0·∇·u` (Stokes, y=2); `τ=-2αₙₚc0^(y-1)`, `η=2αₙₚc0^y·tan(πy/2)`; `AbsorptionMode` = `.powerLaw`/`.noDispersion`/`.stokes`/`.noAbsorption`; verified against NumPy solver (`ParityTests.test2DPowerLawAbsorptionParity`, `test2DStokesAbsorptionParity`)
3. ✅ Nonlinear propagation (B/A) — EOS gains `+ B/A·ρ²/(2ρ0)`; mass-conservation source term scaled by `nl_factor = (2·Σρ + ρ0)/ρ0` (previous-step split densities); `medium.bOnA`; verified against NumPy solver (`ParityTests.test2DNonlinearParity`)
4. ✅ Time-reversal reconstruction — `timeReversal()` helper in `Solver/TimeReversal.swift`: flips recorded sensor pressure in time, re-injects as a Dirichlet pressure source on the sensor mask, runs the solver, returns `compensationFactor·pFinal` with positivity clamp. Pure composition of the existing (parity-tested) Dirichlet source path. Verified against NumPy (`ParityTests.test2DTimeReversalParity`)
5. ✅ Sensor recording fields beyond `p`/`pFinal` — pMax/pMin/pRms, collocated ux/uy/uz time series (unstaggered via `exp(-i·k·d/2)`), uMax/uRms (per-component), uFinal, iAvg (time-averaged intensity with Fourier half-sample velocity shift). Driven by `RecordField` option set + `RecordPlan`; aggregates/intensity post-processed in `Sensor/Recording.swift`. Verified against NumPy (`ParityRecordTests.test2DRecordingFieldsParity`). NOTE: `iMax` not implemented (k-wave-python's `acoustic_intensity` has no I_max output to validate against)
6. ✅ Sensor directivity — 2D, `pressure` (sinc) + `gradient` patterns; k-space plane-wave decomposition per unique angle, ported from MATLAB k-Wave `directionalResponse.m` (`Sensor/Directivity.swift`). Verified against a NumPy re-port of the same formula on a simulated field (`ParityDirectivityTests`)
7. ❌ Real-time monitoring UI (Metal)
8. ❌ Movie recording (AVFoundation)
9. ❌ Full parity test suite (target: all 47 k-wave-python parity tests)
10. ❌ Performance benchmarks vs MATLAB

**Deliverable:** Feature-complete fluid acoustic solver matching MATLAB `kspaceFirstOrder*`.

### Phase 3: Extended Features (P1) — ❌ Not started

1. ❌ Axisymmetric solver `kspaceFirstOrderAS` → `Solver/Axisymmetric.swift`
2. ❌ CW propagation: `acousticFieldPropagator`, `angularSpectrumCW` → `Solver/CW.swift`
3. ❌ `KWaveArray` (off-grid transducer arrays) → `Array/KWaveArray.swift`
4. ❌ `KWaveTransducer` (linear array model) → `Transducer/KWaveTransducer.swift`
5. ❌ Remaining grid geometry: `makeArc`, `makeLine`, `makeBowl`, `makeMultiArc`, `makeMultiBowl`, `makeSphericalSection` → add to `Geometry/Shapes.swift`
6. ❌ Cartesian geometry: `makeCartCircle`, `makeCartSphere`, `makeCartArc`, `makeCartBowl`, `makeCartDisc`, `makeCartRect` → `Geometry/Cartesian.swift`
7. ❌ Reconstruction: `kspaceLineRecon`, `kspacePlaneRecon` → `Reconstruction/FFTRecon.swift`
8. ❌ Signal processing: `addNoise`, `createCWSignals`, `extractAmpPhase`, `logCompression`, `envelopeDetection`, `gradientFD`, `gradientSpect`, `spect` → `Signal/Processing.swift`
9. ❌ Material properties: `waterSoundSpeed`, `waterDensity`, `waterAbsorption`, `waterNonlinearity`, `fitPowerLawParams` → `Material/Properties.swift`
10. ❌ Visualization: `SimulationDisplay`, `beamPlot`, `flyThrough`, `overlayPlot`, `stackedPlot` → `Viz/FieldDisplay.swift`, `Viz/Plots.swift`
11. ❌ CLI tool → `kwave-cli/main.swift`
12. ❌ SwiftUI dashboard prototype
13. ❌ DocC API docs and tutorials

### Phase 4: Advanced Solvers and Ecosystem (P2+) — ❌ Not started

1. ❌ Elastic wave solvers `pstdElastic2D`, `pstdElastic3D` → `Solver/Elastic.swift`
2. ❌ Thermal simulation `kwaveDiffusion`, `bioheatExact` → `Solver/Diffusion.swift`
3. ❌ `voxelPlot`, `isosurfacePlot`, `maxIntensityProjection` → `Viz/Voxel.swift`
4. ❌ Analytical reference solutions: `focusedAnnulusONeil`, `focusedBowlONeil`, `mendousse` → `Reference/Analytical.swift`
5. ❌ Beamforming reconstruction: `beamformDelayAndSum`, `scanConversion` → `Reconstruction/Beamform.swift`
6. ❌ Full parameter-editing dashboard
7. ❌ PackageCompiler-equivalent standalone app bundle
8. ❌ Ported example suite (~80 examples matching MATLAB)

---

## 13. Acceptance Criteria

1. **Functional parity:** All P0 and P1 functions produce outputs matching MATLAB k-Wave within
   documented numerical tolerance. Per §1.1, tolerance is precision-dependent: tight for
   float64/CPU runs, looser for float32/GPU runs.
2. **Parity test coverage:** Pass all 47 existing parity tests from k-wave-python (at the
   appropriate precision tolerance), plus additional tests for uncovered functions.
3. **Performance:** MLX GPU solver competitive with k-Wave C++/CUDA binaries on comparable Apple
   silicon; MLX CPU solver competitive with MATLAB's pure-MATLAB solver.
4. **UI:** Interactive simulation monitoring with real-time Metal field display.
5. **HDF5 compatibility:** Input/output files compatible with existing k-Wave C++/CUDA binaries
   (same dataset names, attributes, and layout).
6. **Platform:** Runs on macOS / Apple silicon.
7. **Documentation:** DocC docs for all exported types and functions; ported examples with comments.
8. **Code quality:** Near-zero allocation in inner solver loops; clean build with no warnings.
9. **Precision policy:** float32/GPU default and float64/CPU validation path both functional, with
   documented tolerances applied in the parity suite.

---

## 14. Resolved Design Decisions

1. **Float32 vs Float64:** float32 is the default (GPU-capable, jwave-style). A `dtype = .float64`
   option runs on the CPU stream for tightest parity validation, since Metal has no `double`. The
   solver and utilities operate on `MLXArray` and work at either precision; the parity suite
   applies precision-dependent tolerances. (See §1.1.)

2. **GPU FFT:** `MLXFFT` runs on the Metal GPU for float32 and on the CPU stream for float64.
   Verify accuracy against the parity suite during bring-up; document any FFT-implementation
   tolerance deltas relative to MATLAB/FFTW.

3. **Optional backends/visualization:** The core `KWave` library depends only on `mlx-swift` and
   `libhdf5`. Visualization (SwiftUI/Charts/Metal/AVFoundation) and the CLI live in separate
   targets so the compute core stays lean and headless-capable.

4. **App scope:** A SwiftUI desktop app provides launch + real-time monitoring + result export
   (Phase 3), with full parameter editing in Phase 4 if warranted.

5. **Compatibility with k-wave-python:** Numerically equivalent results within documented
   floating-point tolerance, not bit-for-bit identical files. HDF5 file *format* must remain
   compatible with the k-Wave C++/CUDA binaries (same dataset names, attributes, data layout).

6. **Platform floor:** Apple silicon / macOS, Swift ≥ 6.0. MLX requires Apple silicon for GPU;
   the CPU path also runs there. (Cross-platform Linux support is explicitly out of scope; it
   would require replacing MLX with FFTW + a custom array layer.)

7. **License:** Apache-2.0. More permissive than upstream's LGPL-3.0, enabling broader adoption in
   commercial and production workflows. This is a clean-room port, not a derivative of the LGPL code.
</content>

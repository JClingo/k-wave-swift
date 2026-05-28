# k-wave-jl: Julia Port of k-Wave — Requirements Document

## 1. Project Overview

**Goal:** Port the k-Wave acoustic simulation toolbox from MATLAB to Julia, providing both a 1:1 functional match with the MATLAB original and an improved UI/UX for production workflows.

**Source references:**
- MATLAB original: https://github.com/ucl-bug/k-wave (v1.4.1, LGPL-3.0)
- Python port: https://github.com/waltsims/k-wave-python (v0.6.1, LGPL-3.0)

**License:** Apache-2.0 (clean-room port; see §14.7)
**Julia version:** ≥ 1.12

**Why Julia:**
- Near-1:1 syntax mapping from MATLAB (arrays, broadcasting, FFT, linear algebra)
- LLVM-compiled performance matching C/Fortran for numerical workloads
- First-class GPU support via CUDA.jl (native Julia kernels on GPU)
- Makie.jl ecosystem for real-time scientific visualization (GLMakie for native, WGLMakie for web)
- Multiple dispatch naturally handles 1D/2D/3D solver specialization
- FFTW.jl, HDF5.jl, and mature scientific computing ecosystem
- PackageCompiler.jl for standalone executables when needed

**k-Wave** is a toolbox for time-domain simulation of acoustic wave propagation using the k-space pseudospectral method. It supports 1D, 2D, and 3D simulations of linear/nonlinear propagation in heterogeneous media with power-law absorption.

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
- FFT-based spatial gradient computation (via FFTW.jl)
- Perfectly Matched Layer (PML) absorbing boundary conditions
- Support for heterogeneous sound speed and density fields
- Power-law acoustic absorption and dispersion
- Nonlinear propagation via B/A parameter
- Time-reversal mode for photoacoustic image reconstruction
- Staggered grid implementation for velocity/pressure fields
- CFL-based automatic time step calculation

**Julia design:** Use multiple dispatch to specialize on grid dimensionality:
```julia
kspace_first_order(grid::KWaveGrid1D, medium, source, sensor; kwargs...)
kspace_first_order(grid::KWaveGrid2D, medium, source, sensor; kwargs...)
kspace_first_order(grid::KWaveGrid3D, medium, source, sensor; kwargs...)
```
A shared `AbstractKWaveGrid` supertype captures the common interface. Internal solver steps (gradient computation, PML application, source injection) dispatch on dimension similarly.

### 2.2 Elastic Simulations

| Function (MATLAB) | Description | Priority |
|---|---|---|
| `pstdElastic2D` | 2D elastic (shear + compressional) wave propagation | P2 |
| `pstdElastic3D` | 3D elastic wave propagation | P2 |

**Requirements:**
- Viscoelastic absorption model
- Coupled compressional and shear wave fields
- Stress tensor source support

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

**Requirements:**
- Diffusion + perfusion + metabolic heat generation
- Coupling with acoustic absorption for HIFU heating studies

---

## 3. Data Structures

### 3.1 `KWaveGrid`

The computational grid — the central data structure. Uses an abstract type hierarchy for dimension-specialized dispatch.

```julia
abstract type AbstractKWaveGrid end

struct KWaveGrid1D <: AbstractKWaveGrid
    nx::Int; dx::Float64
    x::Vector{Float64}
    kx::Vector{Float64}; k::Vector{Float64}
    dt::Float64; nt::Int; t_array::Vector{Float64}
end

struct KWaveGrid2D <: AbstractKWaveGrid
    nx::Int; dx::Float64; ny::Int; dy::Float64
    x::Matrix{Float64}; y::Matrix{Float64}
    kx::Matrix{Float64}; ky::Matrix{Float64}; k::Matrix{Float64}
    dt::Float64; nt::Int; t_array::Vector{Float64}
end

struct KWaveGrid3D <: AbstractKWaveGrid
    nx::Int; dx::Float64; ny::Int; dy::Float64; nz::Int; dz::Float64
    x::Array{Float64,3}; y::Array{Float64,3}; z::Array{Float64,3}
    kx::Array{Float64,3}; ky::Array{Float64,3}; kz::Array{Float64,3}; k::Array{Float64,3}
    dt::Float64; nt::Int; t_array::Vector{Float64}
end
```

**Constructor and methods:**
```julia
KWaveGrid(nx, dx)                    # → KWaveGrid1D
KWaveGrid(nx, dx, ny, dy)            # → KWaveGrid2D
KWaveGrid(nx, dx, ny, dy, nz, dz)   # → KWaveGrid3D
make_time!(grid, sound_speed; cfl=0.3, t_end=nothing)
ndims(grid::AbstractKWaveGrid)
total_grid_points(grid::AbstractKWaveGrid)
```

### 3.2 `KWaveMedium`

Acoustic medium properties. Parametric on the array/scalar type for GPU compatibility.

```julia
@kwarg struct KWaveMedium{T}
    sound_speed::T                              # scalar or array [m/s]
    density::T                                  # scalar or array [kg/m³]
    alpha_coeff::Union{Nothing, T} = nothing    # absorption coefficient
    alpha_power::Union{Nothing, Float64} = nothing  # absorption power law exponent
    alpha_mode::Symbol = :no_absorption         # :no_absorption, :no_dispersion, :stokes
    BonA::Union{Nothing, T} = nothing           # nonlinearity parameter B/A
    # Elastic media
    sound_speed_compression::Union{Nothing, T} = nothing
    sound_speed_shear::Union{Nothing, T} = nothing
end
```

### 3.3 `KWaveSource`

Source definitions.

```julia
@kwarg struct KWaveSource{T}
    # Initial value (photoacoustic)
    p0::Union{Nothing, T} = nothing

    # Time-varying pressure source
    p_mask::Union{Nothing, AbstractArray{Bool}} = nothing
    p::Union{Nothing, T} = nothing
    p_mode::SourceMode = Additive               # Dirichlet, Additive, AdditiveNoCorrection

    # Time-varying velocity source
    u_mask::Union{Nothing, AbstractArray{Bool}} = nothing
    ux::Union{Nothing, T} = nothing
    uy::Union{Nothing, T} = nothing
    uz::Union{Nothing, T} = nothing
    u_mode::SourceMode = Additive

    # Stress source (elastic)
    s_mask::Union{Nothing, AbstractArray{Bool}} = nothing
    sxx::Union{Nothing, T} = nothing
    syy::Union{Nothing, T} = nothing
    szz::Union{Nothing, T} = nothing
    sxy::Union{Nothing, T} = nothing
    sxz::Union{Nothing, T} = nothing
    syz::Union{Nothing, T} = nothing
end

@enum SourceMode Dirichlet Additive AdditiveNoCorrection
```

### 3.4 `KWaveSensor`

Sensor/detector definitions.

```julia
@kwarg struct KWaveSensor
    mask::Union{AbstractArray{Bool}, Matrix{Float64}, Nothing} = nothing
    record::Vector{Symbol} = [:p]       # :p, :p_max, :p_min, :p_rms, :p_final,
                                        # :ux, :uy, :uz, :u_max, :u_rms, :u_final,
                                        # :I_avg, :I_max
    time_reversal_boundary_data::Union{Nothing, AbstractArray} = nothing
    directivity_angle::Union{Nothing, AbstractArray} = nothing
    directivity_size::Union{Nothing, Float64} = nothing
    frequency_response::Union{Nothing, Tuple{Float64, Float64}} = nothing  # (center_freq, bandwidth)
end
```

### 3.5 `KWaveArray`

Off-grid transducer array modeling.

```julia
struct KWaveArray
    elements::Vector{ArrayElement}
end

abstract type ArrayElement end
struct ArcElement <: ArrayElement ... end
struct BowlElement <: ArrayElement ... end
struct DiscElement <: ArrayElement ... end
struct RectElement <: ArrayElement ... end
struct SphereElement <: ArrayElement ... end

# Methods
add_arc_element!(array, position, radius, diameter, focus)
add_bowl_element!(array, position, radius, diameter, focus)
add_disc_element!(array, position, diameter, focus)
add_rect_element!(array, position, width, height, focus)
get_element_binary_mask(array, kgrid, element_index)
get_distributed_source_signal(array, kgrid, source_signal)
combine_sensor_data(array, kgrid, sensor_data)
```

### 3.6 `KWaveTransducer`

Linear array transducer model (for 3D simulations).

```julia
@kwarg struct KWaveTransducer
    number_elements::Int
    element_width::Int                  # grid points
    element_length::Int                 # grid points
    element_spacing::Int = 0            # grid points (kerf)
    position::NTuple{3, Int} = (1,1,1)
    radius::Float64 = Inf              # focus radius (Inf = flat)
    focus_distance::Float64 = Inf
    steering_angle::Float64 = 0.0
    transmit_apodization::Symbol = :rectangular
    receive_apodization::Symbol = :rectangular
    active_elements::Union{Nothing, Vector{Int}} = nothing
    input_signal::Union{Nothing, AbstractVector} = nothing
end
```

---

## 4. Utility Functions

### 4.1 Geometry / Shape Creation (Binary Masks)

All functions generate binary masks on a grid:

| Function | Description | Priority |
|---|---|---|
| `make_disc` | Filled circle in 2D | P0 |
| `make_circle` | Circle perimeter in 2D | P0 |
| `make_ball` | Filled sphere in 3D | P0 |
| `make_sphere` | Sphere surface in 3D | P0 |
| `make_arc` | Arc in 2D | P1 |
| `make_bowl` | Bowl surface in 3D | P1 |
| `make_line` | Line segment in 2D/3D | P1 |
| `make_spherical_section` | Spherical cap in 3D | P1 |
| `make_multi_arc` | Multiple arcs | P1 |
| `make_multi_bowl` | Multiple bowls | P1 |

### 4.2 Cartesian Geometry (Off-grid Point Distributions)

| Function | Description | Priority |
|---|---|---|
| `make_cart_circle` | Cartesian circle points | P1 |
| `make_cart_sphere` | Cartesian sphere points | P1 |
| `make_cart_arc` | Cartesian arc points | P1 |
| `make_cart_bowl` | Cartesian bowl points | P1 |
| `make_cart_disc` | Cartesian disc points | P1 |
| `make_cart_rect` | Cartesian rectangle points | P1 |

### 4.3 Grid and Matrix Utilities

| Function | Description | Priority |
|---|---|---|
| `cart2grid` | Map Cartesian points onto grid | P0 |
| `grid2cart` | Extract Cartesian coords from grid mask | P0 |
| `expand_matrix` | Expand matrix with boundary conditions | P0 |
| `resize` | Resize matrix (interpolation) | P0 |
| `get_optimal_pml_size` | Compute optimal PML thickness | P0 |
| `interp_cart_data` | Interpolate Cartesian sensor data | P1 |
| `fourier_shift` | Sub-pixel shift via Fourier method | P1 |
| `find_closest` | Find nearest grid point | P1 |
| `off_grid_points` | Off-grid source/sensor interpolation | P1 |

### 4.4 Filtering and Spectral

| Function | Description | Priority |
|---|---|---|
| `apply_filter` | Apply frequency-domain filter | P0 |
| `gaussian_filter` | Apply Gaussian frequency filter | P0 |
| `smooth` | Smooth matrix (N-D) | P0 |
| `get_win` | Generate window functions (Hann, Blackman, etc.) | P0 |
| `filter_time_series` | Causal/zero-phase FIR filter | P1 |
| `spect` | Compute amplitude/phase spectrum | P1 |
| `envelope_detection` | Hilbert transform envelope | P1 |
| `gradient_fd` | Finite-difference gradient | P1 |
| `gradient_spect` | Spectral gradient | P1 |

### 4.5 Signal Creation and Processing

| Function | Description | Priority |
|---|---|---|
| `tone_burst` | Generate tone burst signals | P0 |
| `gaussian` | Generate Gaussian pulse | P0 |
| `add_noise` | Add noise at specified SNR | P1 |
| `create_cw_signals` | Generate CW source signals | P1 |
| `extract_amp_phase` | Extract amplitude and phase from CW | P1 |
| `log_compression` | Log-compress signal for display | P1 |
| `scan_conversion` | Polar to Cartesian scan conversion | P2 |

### 4.6 Absorption and Material Properties

| Function | Description | Priority |
|---|---|---|
| `db2neper` / `neper2db` | Unit conversion | P0 |
| `fit_power_law_params` | Fit absorption to power law | P1 |
| `water_sound_speed` | Temperature-dependent water c | P1 |
| `water_density` | Temperature-dependent water rho | P1 |
| `water_absorption` | Temperature-dependent water alpha | P1 |
| `water_nonlinearity` | Temperature-dependent water B/A | P1 |
| `hounsfield2density` | CT Hounsfield to density conversion | P2 |
| `atten_comp` | Time-domain attenuation compensation | P2 |

### 4.7 Reconstruction

| Function | Description | Priority |
|---|---|---|
| `kspace_line_recon` | 1D FFT-based reconstruction | P1 |
| `kspace_plane_recon` | 2D FFT-based reconstruction | P1 |
| Time-reversal (built into solver) | Via `sensor.time_reversal_boundary_data` | P0 |
| Beamforming reconstruction | Delay-and-sum beamforming | P2 |

### 4.8 Reference/Analytical Solutions

| Function | Description | Priority |
|---|---|---|
| `focused_annulus_oneil` | O'Neil solution for focused annulus | P2 |
| `focused_bowl_oneil` | O'Neil solution for focused bowl | P2 |
| `mendousse` | Mendousse solution (nonlinear 1D) | P2 |

---

## 5. I/O and File Formats

### 5.1 HDF5 Support (Critical)

| Capability | Description | Priority |
|---|---|---|
| Write simulation input to HDF5 | Grid, medium, source, sensor, PML params | P0 |
| Read simulation output from HDF5 | Sensor data, field maxima/RMS | P0 |
| `write_matrix` | Write array to HDF5 dataset | P0 |
| `write_grid` | Write grid + PML properties | P0 |
| `write_flags` | Write simulation flags | P0 |
| `write_attributes` | Write file-level metadata | P0 |
| `h5_compare` | Compare two HDF5 files | P1 |

**Implementation:** Use `HDF5.jl` — mature, well-maintained, supports all HDF5 features. Must maintain format compatibility with existing k-Wave C++/CUDA binaries.

### 5.2 Image and Volume I/O

| Capability | Description | Priority |
|---|---|---|
| `load_image` | Load image as medium map | P1 |
| `save_tiff_stack` | Export 3D volume as TIFF stack | P2 |

**Implementation:** `Images.jl` / `FileIO.jl` for image loading; `TiffImages.jl` for TIFF stacks.

---

## 6. Visualization / UI

### 6.1 Design Approach

Julia's Makie.jl ecosystem provides GPU-accelerated scientific visualization that directly matches (and exceeds) MATLAB's plotting capabilities. Two rendering backends:

- **GLMakie** — Native OpenGL windows for desktop use. Real-time, interactive, GPU-accelerated.
- **WGLMakie** — WebGL rendering in the browser. Enables a web-based UI via Bonito.jl for remote/server workflows.

Both share the same Makie API, so all visualization code works across backends.

### 6.2 Visualization Functions

| Function (MATLAB) | Julia Implementation | Priority |
|---|---|---|
| Real-time simulation plot | `Observable`-based live field display | P0 |
| `beam_plot` | `volume` / `heatmap` with orthogonal plane slicing | P1 |
| `fly_through` | Interactive `Slider`-controlled slice viewer | P1 |
| `get_color_map` | Custom `ColorScheme` (k-Wave perceptual colormap) | P0 |
| `overlay_plot` | `heatmap` with alpha overlay | P1 |
| `stacked_plot` | `series` / `lines` with vertical offsets | P1 |
| `voxel_plot` | `meshscatter` or `volume` rendering | P2 |

### 6.3 Simulation Monitoring

During simulation execution, the UI leverages Makie's `Observable` system for reactive updates:

```julia
# Conceptual usage
fig, sensor_data = kspace_first_order(grid, medium, source, sensor;
    plot_sim=true,          # enable real-time display
    plot_layout=:default,   # :default, :dual, :four_panel
    plot_scale=:auto,       # :auto, (-1.0, 1.0), :symmetric
    record_movie="sim.mp4", # optional movie capture
)
```

**Requirements:**
- Real-time 2D field slice display (selectable: pressure, velocity components)
- Configurable plot layout (single field, multiple subplots via Makie `GridLayout`)
- Adjustable color scale (auto, fixed, symmetric)
- Iteration counter and estimated time remaining
- Non-blocking: simulation runs on a separate thread; rendering on the main thread
- Movie recording via Makie's `record()` function (MP4, GIF)

### 6.4 Web-Based Dashboard (P1 monitoring, P3 full editing)

Using WGLMakie + Bonito.jl, provide a browser-based interface for:
- **Phase 3:** Launching and monitoring simulations remotely, viewing and exporting results
- **Phase 4:** Full simulation parameter editing — grid setup, medium property maps, source/sensor configuration
- Useful for cluster/server deployments where native windows aren't available

### 6.5 UI Requirements

- Cross-platform (macOS, Linux, Windows) via GLMakie
- GPU-accelerated rendering (OpenGL) for large field displays
- Headless mode: all simulations runnable without display (server/cluster use)
- Plot export: PNG, SVG, PDF via `save()` from Makie/CairoMakie

---

## 7. Performance Requirements

### 7.1 Compute Backends

Julia's array abstraction allows writing solver code once and dispatching to different backends:

| Backend | Implementation | Priority |
|---|---|---|
| **CPU (single-threaded)** | Base Julia arrays + FFTW.jl | P0 |
| **CPU (multi-threaded)** | `@threads`, `@turbo` (LoopVectorization.jl), threaded FFTW | P0 |
| **GPU (NVIDIA CUDA)** | CUDA.jl `CuArray` — solver code works unchanged via dispatch | P1 |
| **GPU (AMD ROCm)** | AMDGPU.jl `ROCArray` — same code, different array type | P2 |
| **GPU (Apple Metal)** | Metal.jl `MtlArray` — same code, different array type | P2 |

**Key advantage:** The solver inner loop can be written generically over `AbstractArray`. Swapping `Array` for `CuArray` moves the entire computation to GPU with minimal code changes:
```julia
# CPU
grid = KWaveGrid(128, 1e-4, 128, 1e-4)
medium = KWaveMedium(sound_speed=1500.0, density=1000.0)
result = kspace_first_order(grid, medium, source, sensor)

# GPU — same code, different array type
using CUDA
medium_gpu = KWaveMedium(sound_speed=cu(sound_speed_map), density=cu(density_map))
result = kspace_first_order(grid, medium_gpu, source_gpu, sensor)
```

### 7.2 Memory

- Support simulations with grids up to 512³ (~4 GB for a single Float64 field)
- In-place FFT via FFTW.jl `plan_fft!` to minimize allocations
- Pre-allocated work arrays reused across time steps
- `Float32` mode for GPU simulations (configurable via `data_cast` option)
- Memory-mapped arrays via `Mmap.mmap` for out-of-core processing on very large grids

### 7.3 Performance Targets

- Julia CPU solver should match or exceed MATLAB's pure-MATLAB solver (expected 2-5x faster due to compiled loops)
- With CUDA.jl, should be competitive with existing k-Wave C++/CUDA binaries
- FFTW.jl calls the same FFTW library as MATLAB — FFT performance should be identical
- Use `@turbo` from LoopVectorization.jl for tight inner loops (source injection, PML application)
- Benchmark with `BenchmarkTools.jl` early and continuously

---

## 8. API Design

### 8.1 Package (`KWave.jl`)

```julia
module KWave

# Primary simulation entry point
function kspace_first_order(
    grid::AbstractKWaveGrid,
    medium::KWaveMedium,
    source::KWaveSource,
    sensor::KWaveSensor;
    # PML
    pml_inside::Bool = true,
    pml_size::Union{Int, NTuple} = 20,
    pml_alpha::Float64 = 2.0,
    # Smoothing
    smooth_p0::Bool = true,
    smooth_c0::Bool = false,
    smooth_rho0::Bool = false,
    # Display
    plot_sim::Bool = false,
    plot_layout::Symbol = :default,
    plot_scale::Union{Symbol, Tuple} = :auto,
    record_movie::Union{Nothing, String} = nothing,
    # Data
    data_cast::Type = Float64,
    save_to_disk::Union{Nothing, String} = nothing,
    # Callback
    progress_callback::Union{Nothing, Function} = nothing,
) -> SimulationOutput

end
```

### 8.2 Design Principles

- **Multiple dispatch over grid dimension:** Single `kspace_first_order` entry point, with internal methods specializing on `KWaveGrid1D`, `KWaveGrid2D`, `KWaveGrid3D`
- **Keyword arguments** for all simulation options (mirrors MATLAB name-value pairs)
- **Generic array types:** Solver code operates on `AbstractArray`, enabling CPU/GPU/distributed backends
- **Mutating conventions:** Julia convention of `!` suffix for mutating functions (e.g., `make_time!`, `smooth!`)
- **Composable:** Individual solver steps exported for advanced users (gradient computation, PML application, source injection)
- **Type-parametric structs:** `KWaveMedium{T}` allows any array/scalar type, including GPU arrays
- **Error handling:** Informative error messages with `@assert` and custom exceptions for invalid configurations (e.g., 3D source on 2D grid)
- **Units-aware:** Optional integration with `Unitful.jl` for physical quantities (P2)

### 8.3 CLI Tool

A command-line interface via a standalone script or compiled binary (PackageCompiler.jl):

```bash
julia --project -e 'using KWave; KWave.CLI.main()' -- run --input sim.h5 --output results.h5
julia --project -e 'using KWave; KWave.CLI.main()' -- validate --input sim.h5
julia --project -e 'using KWave; KWave.CLI.main()' -- info --input sim.h5

# Or compiled:
kwave-cli run --input sim.h5 --output results.h5
```

Uses `Comonicon.jl` or `ArgParse.jl` for argument parsing.

### 8.4 Python Interop

For users with existing Python workflows, provide interop via `PythonCall.jl` / `juliacall`:

```python
from juliacall import Main as jl
jl.seval("using KWave")
grid = jl.KWaveGrid(128, 1e-4, 128, 1e-4)
# ... set up medium, source, sensor
result = jl.kspace_first_order(grid, medium, source, sensor)
```

Priority: P2 (after core package is stable)

---

## 9. Testing Strategy

### 9.1 Unit Tests

- Every utility function tested independently via Julia's built-in `Test` stdlib
- Property-based testing for mathematical functions (FFT round-trip, gradient accuracy)
- Edge cases: empty grids, scalar media, single-point sensors
- Type stability checks via `@inferred` and `JET.jl`

### 9.2 MATLAB Parity Tests

**Critical for validation.** Leverage the existing test infrastructure:

1. Use MATLAB reference outputs from `k-wave-python/tests/reference_outputs/` (HDF5 files)
2. Run identical simulation configurations in Julia
3. Compare results to machine precision (or documented tolerance)
4. Target: match all 47 parity tests from k-wave-python, then expand
5. Use `HDF5.jl` to load reference data directly

### 9.3 Integration Tests

- End-to-end simulation scenarios matching k-Wave examples
- HDF5 round-trip (write input, read output, compare with k-Wave C++ binaries)
- Cross-validation with analytical solutions (O'Neil, Mendousse)

### 9.4 Performance Benchmarks

- `BenchmarkTools.jl` benchmarks for FFT, gradient computation, full simulation step
- Comparison against MATLAB and Python solver times
- GPU benchmarks comparing CUDA.jl vs existing k-Wave CUDA binaries
- Memory allocation tracking via `@allocated` and `Profile.Allocs`
- Regression benchmarks in CI via `PkgBenchmark.jl`

---

## 10. Project Structure

```
KWave.jl/
  Project.toml                      # Package metadata and dependencies
  Manifest.toml                     # Locked dependency versions
  src/
    KWave.jl                        # Main module, exports
    grid.jl                         # KWaveGrid types and constructors
    medium.jl                       # KWaveMedium
    source.jl                       # KWaveSource
    sensor.jl                       # KWaveSensor
    array.jl                        # KWaveArray (off-grid transducers)
    transducer.jl                   # KWaveTransducer (linear arrays)
    solver/
      first_order.jl                # kspace_first_order (1D/2D/3D dispatch)
      first_order_steps.jl          # Individual time-stepping components
      axisymmetric.jl               # kspace_first_order_as
      second_order.jl               # kspace_second_order
      elastic.jl                    # pstd_elastic (2D/3D)
      diffusion.jl                  # kwave_diffusion
      cw.jl                         # acoustic_field_propagator, angular_spectrum_cw
    pml.jl                          # PML generation and application
    fft_utils.jl                    # FFT planning and helpers
    geometry/
      shapes.jl                     # make_disc, make_ball, make_arc, etc.
      cartesian.jl                  # make_cart_circle, make_cart_bowl, etc.
    signal/
      generation.jl                 # tone_burst, gaussian
      processing.jl                 # envelope_detection, log_compression, etc.
    filter/
      filters.jl                    # apply_filter, gaussian_filter, smooth, get_win
    material/
      properties.jl                 # water_*, hounsfield2density
      conversion.jl                 # db2neper, neper2db, fit_power_law_params
    reconstruction/
      fft_recon.jl                  # kspace_line_recon, kspace_plane_recon
      time_reversal.jl              # Time-reversal support
      beamform.jl                   # Delay-and-sum beamforming
    reference/
      analytical.jl                 # focused_annulus_oneil, mendousse, etc.
    io/
      hdf5.jl                       # HDF5 read/write (compatible with k-Wave C++ format)
      images.jl                     # Image loading / TIFF stack export
    visualization/
      colormap.jl                   # k-Wave colormap as ColorScheme
      field_display.jl              # Real-time simulation monitoring
      plots.jl                      # beam_plot, overlay_plot, stacked_plot, fly_through
      voxel.jl                      # 3D voxel rendering
    cli.jl                          # CLI entry point
    utils.jl                        # Shared helpers (cart2grid, expand_matrix, resize, etc.)
  test/
    runtests.jl                     # Test runner
    test_grid.jl
    test_solver_1d.jl
    test_solver_2d.jl
    test_solver_3d.jl
    test_geometry.jl
    test_signal.jl
    test_filter.jl
    test_io.jl
    test_visualization.jl
    parity/                         # MATLAB parity tests
      reference_outputs/            # HDF5 reference data from k-wave-python
      test_parity.jl
  bench/
    benchmarks.jl                   # PkgBenchmark suite
  examples/                         # Ported k-Wave example scripts
    ivp_*.jl                        # Initial value problems
    tvsp_*.jl                       # Time-varying source problems
    pr_*.jl                         # Photoacoustic reconstruction
    us_*.jl                         # Ultrasound
    sd_*.jl                         # Sensor directivity
    ewp_*.jl                        # Elastic wave propagation
    diff_*.jl                       # Diffusion
  docs/
    make.jl                         # Documenter.jl build script
    src/
      index.md
      tutorials/
      api/
```

---

## 11. Dependencies

**Core (always loaded):**
| Package | Purpose |
|---|---|
| `FFTW.jl` | FFT computation (wraps FFTW C library) |
| `HDF5.jl` | HDF5 file I/O |
| `LoopVectorization.jl` | SIMD-optimized inner loops |
| `LinearAlgebra` (stdlib) | Matrix operations |
| `Statistics` (stdlib) | Basic statistics |
| `DSP.jl` | Signal processing (filters, windows) |
| `Interpolations.jl` | Grid interpolation |
| `ProgressMeter.jl` | Progress bars |
| `ArgParse.jl` | CLI argument parsing |

**Package extensions (loaded when user imports the trigger package):**
| Extension | Trigger Package | Purpose |
|---|---|---|
| `KWaveCUDAExt` | `CUDA.jl` | NVIDIA GPU arrays and kernels |
| `KWaveAMDGPUExt` | `AMDGPU.jl` | AMD GPU support |
| `KWaveMetalExt` | `Metal.jl` | Apple GPU support |
| `KWaveGLMakieExt` | `GLMakie.jl` | Native GPU-accelerated visualization |
| `KWaveCairoMakieExt` | `CairoMakie.jl` | Publication-quality static plots, headless rendering |
| `KWaveWGLMakieExt` | `WGLMakie.jl` | Web-based visualization |
| `KWaveBonitoExt` | `Bonito.jl` | Web dashboard framework |
| `KWaveUnitfulExt` | `Unitful.jl` | Physical units |
| `KWaveImagesExt` | `Images.jl` / `FileIO.jl` | Image I/O |
| `KWaveTiffExt` | `TiffImages.jl` | TIFF stack I/O |

**Dev/test only:**
| Package | Purpose |
|---|---|
| `Documenter.jl` | API documentation generation |
| `BenchmarkTools.jl` | Performance benchmarking |
| `JET.jl` | Static analysis and type stability |
| `Aqua.jl` | Package quality checks |
| `PackageCompiler.jl` | Standalone binary compilation |

---

## 12. Implementation Phases

### Phase 1: Foundation (P0)

**Goal:** Minimal working 2D simulation with validation.

1. `KWaveGrid` (1D, 2D, 3D construction + wavenumber grids)
2. `KWaveMedium`, `KWaveSource`, `KWaveSensor` data structures
3. FFT planning and utility layer (FFTW.jl wrappers)
4. PML implementation
5. `kspace_first_order` for 2D (linear, homogeneous)
6. Basic geometry: `make_disc`, `make_circle`, `make_ball`, `make_sphere`
7. Grid utilities: `cart2grid`, `grid2cart`, `expand_matrix`, `resize`, `get_optimal_pml_size`
8. Filtering: `apply_filter`, `gaussian_filter`, `smooth`, `get_win`
9. Signal: `tone_burst`, `gaussian`
10. HDF5 I/O (read/write simulation data, k-Wave C++ format compatible)
11. Unit conversion: `db2neper`, `neper2db`
12. `get_color_map` (k-Wave colormap as ColorScheme)
13. First parity tests against MATLAB reference data
14. Basic test suite with type stability checks

**Deliverable:** Can run a 2D initial-value-problem simulation and produce validated output.

### Phase 2: Full Fluid Solver (P0 continued)

**Goal:** Complete fluid simulation capabilities.

1. Heterogeneous media (varying sound speed + density)
2. Power-law absorption and dispersion
3. Nonlinear propagation (B/A)
4. 1D and 3D solvers
5. Time-reversal reconstruction
6. Time-varying pressure and velocity sources (all source modes)
7. All sensor recording fields (p_max, p_rms, u, I_avg, etc.)
8. Sensor directivity
9. Real-time simulation monitoring UI (GLMakie Observable-based)
10. Movie recording via Makie `record()`
11. Full parity test suite (target: all 47 k-wave-python parity tests)
12. Performance benchmarks vs MATLAB

**Deliverable:** Feature-complete fluid acoustic solver matching MATLAB `kspaceFirstOrder*`.

### Phase 3: Extended Features (P1)

1. Axisymmetric solver (`kspace_first_order_as`)
2. CW propagation (`acoustic_field_propagator`, `angular_spectrum_cw`)
3. `KWaveArray` (off-grid transducer arrays)
4. `KWaveTransducer` (linear array model)
5. Remaining geometry functions (arcs, bowls, lines, multi-element)
6. Cartesian geometry functions
7. Reconstruction: `kspace_line_recon`, `kspace_plane_recon`
8. Extended signal processing and material property functions
9. Visualization: `beam_plot`, `fly_through`, `overlay_plot`, `stacked_plot`
10. CUDA.jl GPU backend (swap Array → CuArray)
11. CLI tool
12. Web dashboard prototype (WGLMakie + Bonito.jl)
13. Documenter.jl API docs and tutorials

### Phase 4: Advanced Solvers and Ecosystem (P2+)

1. Elastic wave solvers (`pstd_elastic_2d`, `pstd_elastic_3d`)
2. Thermal simulation (`kwave_diffusion`, `bioheat_exact`)
3. `voxel_plot` and advanced 3D visualization
4. Python interop via juliacall
5. Analytical reference solutions
6. Scan conversion, beamforming reconstruction
7. AMD/Metal GPU backends
8. Unitful.jl integration
9. PackageCompiler.jl standalone binary
10. Ported example suite (~80 examples matching MATLAB)

---

## 13. Acceptance Criteria

1. **Functional parity:** All P0 and P1 functions produce outputs matching MATLAB k-Wave to within machine precision (or documented numerical tolerance due to FFT implementation differences)
2. **Parity test coverage:** Pass all 47 existing parity tests from k-wave-python, plus additional tests for functions not covered
3. **Performance:** Julia CPU solver at least as fast as MATLAB's pure-MATLAB solver (target: 2-5x faster for compiled inner loops); GPU solver competitive with k-Wave C++/CUDA binaries
4. **UI:** Interactive simulation monitoring with real-time field display via GLMakie, matching and exceeding MATLAB's plotting workflow
5. **HDF5 compatibility:** Input/output files compatible with existing k-Wave C++/CUDA binaries
6. **Cross-platform:** Runs on macOS, Linux, and Windows
7. **Documentation:** Documenter.jl API docs for all exported types and functions; ported examples with explanatory comments
8. **Code quality:** All exported functions type-stable (`@inferred` passing); zero allocations in inner solver loops; `Aqua.jl` checks passing
9. **GPU support:** CUDA.jl backend functional for 2D and 3D solvers with no code duplication vs CPU path

---

## 14. Resolved Design Decisions

1. **Float32 vs Float64:** Float64 is the default everywhere (matching MATLAB). A `data_cast=Float32` keyword option is available for GPU simulations where memory is constrained. The solver and all utility functions are generic over `AbstractFloat`, so both precisions work throughout. GPU simulations default to Float64 — users opt into Float32 explicitly.

2. **GPU FFT:** CUDA.jl dispatches to cuFFT automatically for `CuArray`. Verify accuracy parity with FFTW during Phase 3 GPU bring-up via the existing parity test suite. If discrepancies exceed documented tolerance, add cuFFT-specific tolerance thresholds to affected tests.

3. **Package extensions:** Yes. Use Julia 1.9+ package extensions for all optional backends and heavy visualization deps. The base `KWave` package depends only on `FFTW.jl`, `HDF5.jl`, `LinearAlgebra`, `Statistics`, and lightweight utilities. GPU support (`CUDA.jl`, `AMDGPU.jl`, `Metal.jl`), visualization (`GLMakie.jl`, `WGLMakie.jl`, `CairoMakie.jl`), and `Unitful.jl` load via extensions when the user imports those packages.

4. **Web UI scope:** Full simulation parameter editing in the web dashboard — grid setup, medium property maps, source/sensor configuration, simulation launch, real-time monitoring, and result export. Target Phase 3 for the initial version. If the full editing UI proves too complex for Phase 3, ship monitoring + result viewing first and add parameter editing in Phase 4.

5. **Compatibility with k-wave-python:** Numerically equivalent results (within documented floating-point tolerance), not bit-for-bit identical HDF5 files. HDF5 file *format* must remain compatible with the k-Wave C++/CUDA binaries (same dataset names, attributes, and data layout).

6. **Julia version floor:** Julia ≥ 1.12. This gives access to package extensions, improved compiler optimizations, and the latest `Memory` improvements. Set `julia = "1.12"` in `Project.toml` `[compat]`.

7. **License:** Apache-2.0. More permissive than upstream's LGPL-3.0, enabling broader adoption in commercial and production workflows (which aligns with the project goal of making k-Wave more practical for production use). This is a clean-room port, not a derivative work of the LGPL code.

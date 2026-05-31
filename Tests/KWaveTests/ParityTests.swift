import XCTest
import MLX
@testable import KWave

/// Parity against a k-wave-python (k-Wave C++ OMP) 2D IVP reference. The reference HDF5 stores the
/// raw disc p0 and the final pressure field; this test loads the same p0, runs the Swift solver
/// with the matching grid/medium, and compares p_final.
///
/// Regenerate the reference with: `.venv-kwave/bin/python Scripts/parity/generate_reference.py`.
final class ParityTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)              // .../Tests/KWaveTests/ParityTests.swift
            .deletingLastPathComponent()             // .../Tests/KWaveTests
            .deletingLastPathComponent()             // .../Tests
            .deletingLastPathComponent()             // repo root
            .appendingPathComponent("Scripts/parity/reference_2d_ivp.h5").path
    }

    func test2DInitialValueProblemParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference.py")
        }
        let f = try HDF5File(open: refPath)

        let nx = Int(try f.readFloatDataset("Nx").data[0])
        let ny = Int(try f.readFloatDataset("Ny").data[0])
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let dy = Double(try f.readFloatDataset("dy").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let rho0 = Double(try f.readFloatDataset("rho0").data[0])
        let pmlSize = Int(try f.readFloatDataset("pml_size").data[0])

        let (p0Shape, p0Data) = try f.readFloatDataset("p0")
        let (pfShape, pfData) = try f.readFloatDataset("p_final")
        XCTAssertEqual(p0Shape, [nx, ny])
        XCTAssertEqual(pfShape, [nx, ny])

        var grid = KWaveGrid(nx: nx, dx: dx, ny: ny, dy: dy)
        grid.makeTime(soundSpeedMax: c0, cfl: 0.3)

        let medium = KWaveMedium(soundSpeed: c0, density: rho0)
        var source = KWaveSource()
        source.p0 = MLXArray(p0Data).reshaped([nx, ny])
        var options = SimulationOptions()
        options.pmlSize = .uniform(pmlSize)

        let out = kspaceFirstOrder(grid: grid, medium: medium, source: source,
                                   sensor: KWaveSensor(), options: options)
        let mine = out.pFinal!
        let ref = MLXArray(pfData).reshaped([nx, ny])

        let diff = MLX.abs(mine - ref)
        let maxErr = MLX.max(diff).item(Float.self)
        let refMax = MLX.max(MLX.abs(ref)).item(Float.self)
        let l2 = sqrt(MLX.sum(diff * diff).item(Float.self) / Float(nx * ny))
        let refL2 = sqrt(MLX.sum(ref * ref).item(Float.self) / Float(nx * ny))

        // The Swift solver mirrors the same k-space PSTD algorithm as k-Wave's C++ engine, so the
        // full 302-step run matches to float32 precision (≈1e-5 relative). Tolerances are set an
        // order of magnitude above the observed error to absorb FFT/accumulation rounding.
        XCTAssertEqual(grid.nt, 302)
        XCTAssertLessThan(l2 / refL2, 1e-3)
        XCTAssertLessThan(maxErr / refMax, 1e-3)
    }

    private var heteroRefPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_2d_hetero.h5").path
    }

    /// Parity against a k-wave-python 2D IVP reference in a *spatially varying* medium (a circular
    /// inclusion of higher sound speed and density). Exercises the staggered-density velocity
    /// update, `dt·ρ0` density update, `c0²·Σρ` equation of state, and `c_ref = max(c0)`.
    ///
    /// The reference comes from k-wave-python's pure-NumPy solver (`kspace_solver`), not the C++
    /// engine: in a heterogeneous medium the C++ engine and that NumPy solver diverge ~2% at the
    /// medium discontinuity (a C++ internal detail the NumPy solver doesn't reproduce). The Swift
    /// solver is a 1:1 port of the NumPy formulation, so it matches the NumPy field to float32
    /// precision but inherits the same ~2% offset from C++. Validating against the NumPy reference
    /// keeps a tight tolerance; see Scripts/parity/generate_reference_hetero.py for the full
    /// rationale and diag_numpy_vs_cpp.py for the measurement.
    func test2DHeterogeneousParity() throws {
        guard FileManager.default.fileExists(atPath: heteroRefPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_hetero.py")
        }
        let f = try HDF5File(open: heteroRefPath)

        let nx = Int(try f.readFloatDataset("Nx").data[0])
        let ny = Int(try f.readFloatDataset("Ny").data[0])
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let dy = Double(try f.readFloatDataset("dy").data[0])
        let dt = Double(try f.readFloatDataset("dt").data[0])
        let nt = Int(try f.readFloatDataset("Nt").data[0])
        let pmlSize = Int(try f.readFloatDataset("pml_size").data[0])

        let (c0Shape, c0Data) = try f.readFloatDataset("c0")
        let (rho0Shape, rho0Data) = try f.readFloatDataset("rho0")
        let (p0Shape, p0Data) = try f.readFloatDataset("p0")
        let (pfShape, pfData) = try f.readFloatDataset("p_final")
        XCTAssertEqual(c0Shape, [nx, ny])
        XCTAssertEqual(rho0Shape, [nx, ny])
        XCTAssertEqual(p0Shape, [nx, ny])
        XCTAssertEqual(pfShape, [nx, ny])

        var grid = KWaveGrid(nx: nx, dx: dx, ny: ny, dy: dy)
        grid.setTime(nt: nt, dt: dt)

        let medium = KWaveMedium(soundSpeed: MLXArray(c0Data).reshaped([nx, ny]),
                                 density: MLXArray(rho0Data).reshaped([nx, ny]))
        var source = KWaveSource()
        source.p0 = MLXArray(p0Data).reshaped([nx, ny])
        var options = SimulationOptions()
        options.pmlSize = .uniform(pmlSize)

        let out = kspaceFirstOrder(grid: grid, medium: medium, source: source,
                                   sensor: KWaveSensor(), options: options)
        let mine = out.pFinal!
        let ref = MLXArray(pfData).reshaped([nx, ny])

        let diff = MLX.abs(mine - ref)
        let maxErr = MLX.max(diff).item(Float.self)
        let refMax = MLX.max(MLX.abs(ref)).item(Float.self)
        let l2 = sqrt(MLX.sum(diff * diff).item(Float.self) / Float(nx * ny))
        let refL2 = sqrt(MLX.sum(ref * ref).item(Float.self) / Float(nx * ny))

        // Swift mirrors the NumPy solver's update equations exactly, so the full 363-step run
        // matches to float32 precision (≈1e-5 relative). Tolerances sit an order of magnitude above.
        XCTAssertEqual(grid.nt, nt)
        XCTAssertLessThan(l2 / refL2, 1e-4)
        XCTAssertLessThan(maxErr / refMax, 1e-4)
    }

    private func absorptionRefPath(_ name: String) -> String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/\(name)").path
    }

    /// Parity against a k-wave-python NumPy 2D IVP reference in a homogeneous medium with power-law
    /// absorption. Exercises the `+ absorption(div_u) - dispersion(rho)` terms in the equation of
    /// state (the `|k|^(y-2)` / `|k|^(y-1)` fractional Laplacians and the `tau`/`eta` coefficients).
    ///
    /// Reference is the pure-NumPy solver (which Swift ports 1:1), so the tolerance is tight.
    /// Regenerate with `Scripts/parity/generate_reference_absorption.py` (ALPHA_MODE/SUFFIX env).
    private func runAbsorptionParity(refName: String, mode: AbsorptionMode) throws {
        let path = absorptionRefPath(refName)
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_absorption.py")
        }
        let f = try HDF5File(open: path)

        let nx = Int(try f.readFloatDataset("Nx").data[0])
        let ny = Int(try f.readFloatDataset("Ny").data[0])
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let dy = Double(try f.readFloatDataset("dy").data[0])
        let dt = Double(try f.readFloatDataset("dt").data[0])
        let nt = Int(try f.readFloatDataset("Nt").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let rho0 = Double(try f.readFloatDataset("rho0").data[0])
        let pmlSize = Int(try f.readFloatDataset("pml_size").data[0])
        let alphaCoeff = Double(try f.readFloatDataset("alpha_coeff").data[0])
        let alphaPower = Double(try f.readFloatDataset("alpha_power").data[0])

        let (p0Shape, p0Data) = try f.readFloatDataset("p0")
        let (pfShape, pfData) = try f.readFloatDataset("p_final")
        XCTAssertEqual(p0Shape, [nx, ny])
        XCTAssertEqual(pfShape, [nx, ny])

        var grid = KWaveGrid(nx: nx, dx: dx, ny: ny, dy: dy)
        grid.setTime(nt: nt, dt: dt)

        let medium = KWaveMedium(soundSpeed: c0, density: rho0,
                                 alphaCoeff: alphaCoeff, alphaPower: alphaPower, alphaMode: mode)
        var source = KWaveSource()
        source.p0 = MLXArray(p0Data).reshaped([nx, ny])
        var options = SimulationOptions()
        options.pmlSize = .uniform(pmlSize)

        let out = kspaceFirstOrder(grid: grid, medium: medium, source: source,
                                   sensor: KWaveSensor(), options: options)
        let mine = out.pFinal!
        let ref = MLXArray(pfData).reshaped([nx, ny])

        let diff = MLX.abs(mine - ref)
        let maxErr = MLX.max(diff).item(Float.self)
        let refMax = MLX.max(MLX.abs(ref)).item(Float.self)
        let l2 = sqrt(MLX.sum(diff * diff).item(Float.self) / Float(nx * ny))
        let refL2 = sqrt(MLX.sum(ref * ref).item(Float.self) / Float(nx * ny))

        XCTAssertEqual(grid.nt, nt)
        XCTAssertLessThan(l2 / refL2, 1e-4)
        XCTAssertLessThan(maxErr / refMax, 1e-4)
    }

    /// Full power-law absorption + dispersion (`alpha_power = 1.5`).
    func test2DPowerLawAbsorptionParity() throws {
        try runAbsorptionParity(refName: "reference_2d_absorption.h5", mode: .powerLaw)
    }

    /// Stokes absorption (`alpha_power = 2`, direct multiply, no dispersion term).
    func test2DStokesAbsorptionParity() throws {
        try runAbsorptionParity(refName: "reference_2d_absorption_stokes.h5", mode: .stokes)
    }
}

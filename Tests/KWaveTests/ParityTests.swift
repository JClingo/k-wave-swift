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
}

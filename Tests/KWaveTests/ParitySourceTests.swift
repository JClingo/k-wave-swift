import XCTest
import MLX
@testable import KWave

/// Parity for time-varying pressure-source injection against a k-wave-python (python/NumPy backend)
/// 2D reference. The python backend with `pml_inside=True` returns the interior, so the Swift
/// full-grid result is cropped to the interior before comparison.
///
/// Regenerate references with:
///   `.venv-kwave/bin/python Scripts/parity/generate_reference_source.py`                 (additive)
///   `MODE=dirichlet SUFFIX=_dir ... generate_reference_source.py`                         (dirichlet)
///   `MODE=additive-no-correction SUFFIX=_nc ... generate_reference_source.py`             (no correction)
final class ParitySourceTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private func refPath(_ suffix: String) -> String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_source\(suffix).h5").path
    }

    private func runParity(suffix: String, mode: SourceMode, velocity: Bool = false)
        throws -> (relL2: Float, relMax: Float)
    {
        let path = refPath(suffix)
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("reference not generated: run generate_reference_source.py")
        }
        let f = try HDF5File(open: path)
        let nx = Int(try f.readFloatDataset("Nx").data[0])
        let ny = Int(try f.readFloatDataset("Ny").data[0])
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let rho0 = Double(try f.readFloatDataset("rho0").data[0])
        let dt = Double(try f.readFloatDataset("dt").data[0])
        let nt = Int(try f.readFloatDataset("Nt").data[0])
        let pml = Int(try f.readFloatDataset("pml_size").data[0])
        let sx = Int(try f.readFloatDataset("src_x").data[0])
        let sy = Int(try f.readFloatDataset("src_y").data[0])
        let signal = try f.readFloatDataset("signal").data
        let (pfShape, pfData) = try f.readFloatDataset("p_final")

        var grid = KWaveGrid(nx: nx, dx: dx, ny: ny, dy: dx)
        grid.makeTime(soundSpeedMax: c0, cfl: 0.3)
        grid.setTime(nt: nt, dt: dt)

        // Single-point pressure source at (sx, sy).
        var maskHost = [Float](repeating: 0, count: nx * ny)
        maskHost[sx * ny + sy] = 1
        let mask = MLXArray(maskHost).reshaped([nx, ny])
        var source = KWaveSource()
        if velocity {
            source.uMask = mask
            source.ux = MLXArray(signal)
            source.uMode = mode
        } else {
            source.pMask = mask
            source.p = MLXArray(signal)
            source.pMode = mode
        }

        var opts = SimulationOptions()
        opts.pmlSize = .uniform(pml)
        opts.smoothP0 = false
        let out = kspaceFirstOrder(grid: grid,
                                   medium: KWaveMedium(soundSpeed: c0, density: rho0),
                                   source: source, sensor: KWaveSensor(), options: opts)

        let mine = out.pFinal![pml ..< (nx - pml), pml ..< (ny - pml)]
        let ref = MLXArray(pfData).reshaped(pfShape)
        let diff = MLX.abs(mine - ref)
        let n = pfData.count
        let relL2 = sqrt(MLX.sum(diff * diff).item(Float.self) / Float(n))
            / sqrt(MLX.sum(ref * ref).item(Float.self) / Float(n))
        let relMax = MLX.max(diff).item(Float.self) / MLX.max(MLX.abs(ref)).item(Float.self)
        return (relL2, relMax)
    }

    func testAdditivePressureSourceParity() throws {
        let (relL2, relMax) = try runParity(suffix: "", mode: .additive)
        XCTAssertLessThan(relL2, 1e-3)
        XCTAssertLessThan(relMax, 1e-3)
    }

    func testDirichletPressureSourceParity() throws {
        let (relL2, relMax) = try runParity(suffix: "_dir", mode: .dirichlet)
        XCTAssertLessThan(relL2, 1e-3)
        XCTAssertLessThan(relMax, 1e-3)
    }

    func testAdditiveNoCorrectionPressureSourceParity() throws {
        let (relL2, relMax) = try runParity(suffix: "_nc", mode: .additiveNoCorrection)
        XCTAssertLessThan(relL2, 1e-3)
        XCTAssertLessThan(relMax, 1e-3)
    }

    func testAdditiveVelocitySourceParity() throws {
        let (relL2, relMax) = try runParity(suffix: "_ux", mode: .additive, velocity: true)
        XCTAssertLessThan(relL2, 1e-3)
        XCTAssertLessThan(relMax, 1e-3)
    }
}

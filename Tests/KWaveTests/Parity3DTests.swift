import XCTest
import MLX
@testable import KWave

/// Parity for the 3D solver against a k-wave-python (python/NumPy backend) 3D IVP reference.
/// The python backend with `pml_inside=True` returns the interior (PML stripped), so the Swift
/// full-grid result is cropped to the interior before comparison.
///
/// Regenerate references with:
///   `.venv-kwave/bin/python Scripts/parity/generate_reference_py_3d.py`            (smoothed full run)
///   `SMOOTH=1 NSTEPS=40 SUFFIX=_s40 ... generate_reference_py_3d.py`               (smoothed, 40 steps)
///   `SMOOTH=0 SUFFIX=_ns ... generate_reference_py_3d.py`                          (no-smooth full run)
final class Parity3DTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private func refPath(_ suffix: String) -> String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_3d_ivp\(suffix).h5").path
    }

    /// Run the 3D solver from a reference HDF5 (p0, grid, medium) and return the interior-cropped
    /// `p_final` alongside the reference field, plus relative L2 / max errors.
    private func runParity(suffix: String, nt: Int?, smooth: Bool)
        throws -> (relL2: Float, relMax: Float)
    {
        let path = refPath(suffix)
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("reference not generated: run generate_reference_py_3d.py")
        }
        let f = try HDF5File(open: path)
        let nx = Int(try f.readFloatDataset("Nx").data[0])
        let ny = Int(try f.readFloatDataset("Ny").data[0])
        let nz = Int(try f.readFloatDataset("Nz").data[0])
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let rho0 = Double(try f.readFloatDataset("rho0").data[0])
        let pml = Int(try f.readFloatDataset("pml_size").data[0])
        let (_, p0Data) = try f.readFloatDataset("p0")
        let (pfShape, pfData) = try f.readFloatDataset("p_final")

        var grid = KWaveGrid(nx: nx, dx: dx, ny: ny, dy: dx, nz: nz, dz: dx)
        grid.makeTime(soundSpeedMax: c0, cfl: 0.3)
        if let nt { grid.setTime(nt: nt, dt: grid.dt) }

        var source = KWaveSource()
        source.p0 = MLXArray(p0Data).reshaped([nx, ny, nz])
        var opts = SimulationOptions()
        opts.pmlSize = .uniform(pml)
        opts.smoothP0 = smooth
        let out = kspaceFirstOrder(grid: grid,
                                   medium: KWaveMedium(soundSpeed: c0, density: rho0),
                                   source: source, sensor: KWaveSensor(), options: opts)
        // Crop full-grid result to the interior the python backend returns.
        let mine = out.pFinal![pml ..< (nx - pml), pml ..< (ny - pml), pml ..< (nz - pml)]
        let ref = MLXArray(pfData).reshaped(pfShape)

        let diff = MLX.abs(mine - ref)
        let n = pfData.count
        let l2 = sqrt(MLX.sum(diff * diff).item(Float.self) / Float(n))
        let refL2 = sqrt(MLX.sum(ref * ref).item(Float.self) / Float(n))
        let relMax = MLX.max(diff).item(Float.self) / MLX.max(MLX.abs(ref)).item(Float.self)
        return (l2 / refL2, relMax)
    }

    /// Smoothed initial-pressure 3D IVP, snapshotted at 40 steps while the wave is still inside the
    /// interior (the full ~185-step run radiates the field out, leaving a near-zero interior that is
    /// dominated by the float32 noise floor — an ill-conditioned target for a relative tolerance).
    func test3DInitialValueProblemParity() throws {
        let (relL2, relMax) = try runParity(suffix: "_s40", nt: 40, smooth: true)
        XCTAssertLessThan(relL2, 1e-3)
        XCTAssertLessThan(relMax, 1e-3)
    }

    /// Rigorous full-run parity without source smoothing: the sharp ball retains in-interior energy
    /// for the whole run, so the relative tolerance stays meaningful over all ~185 steps.
    func test3DNoSmoothFullRunParity() throws {
        let (relL2, relMax) = try runParity(suffix: "_ns", nt: nil, smooth: false)
        XCTAssertLessThan(relL2, 1e-3)
        XCTAssertLessThan(relMax, 1e-3)
    }
}

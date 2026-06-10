import XCTest
import MLX
@testable import KWave

/// Parity for the axisymmetric solver (`kspaceFirstOrderAS`, WSWA-FFT symmetry). The C++ OMP
/// binary aborts on axisymmetric inputs on macOS (its FFTW lacks real-to-real transforms) and
/// k-wave-python has no NumPy AS time loop, so the reference is an independent NumPy port of the
/// same WSWA-FFT branch of MATLAB kspaceFirstOrderAS.m; agreement to float32 precision over the
/// full 120-step run cross-checks both implementations.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_as.py`.
final class ParityAxisymmetricTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_as.h5").path
    }

    func testAxisymmetricIVPParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_as.py")
        }
        let f = try HDF5File(open: refPath)
        let nx = Int(try f.readFloatDataset("Nx").data[0])
        let ny = Int(try f.readFloatDataset("Ny").data[0])
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let dy = Double(try f.readFloatDataset("dy").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let rho0 = Double(try f.readFloatDataset("rho0").data[0])
        let dt = Double(try f.readFloatDataset("dt").data[0])
        let nt = Int(try f.readFloatDataset("Nt").data[0])
        let pml = Int(try f.readFloatDataset("pml").data[0])

        let (p0Shape, p0Data) = try f.readFloatDataset("p0")
        let (maskShape, maskData) = try f.readFloatDataset("mask")
        let (tsShape, tsData) = try f.readFloatDataset("p_ts")
        let (pfShape, pfData) = try f.readFloatDataset("p_final")

        var grid = KWaveGrid(nx: nx, dx: dx, ny: ny, dy: dy)
        grid.setTime(nt: nt, dt: dt)

        var source = KWaveSource()
        source.p0 = MLXArray(p0Data).reshaped(p0Shape)
        let sensor = KWaveSensor(mask: MLXArray(maskData).reshaped(maskShape))

        var opts = SimulationOptions()
        opts.pmlSize = .uniform(pml)
        opts.smoothP0 = false        // the stored p0 is already smoothed.

        let out = kspaceFirstOrderAS(grid: grid, medium: KWaveMedium(soundSpeed: c0, density: rho0),
                                     source: source, sensor: sensor, options: opts)

        func check(_ mine: MLXArray, _ refData: [Float], _ refShape: [Int], _ label: String) {
            let ref = MLXArray(refData).reshaped(refShape)
            XCTAssertEqual(mine.shape, refShape, "\(label): shape")
            let diff = MLX.abs(mine - ref)
            let n = Float(refData.count)
            let scale = MLX.max(MLX.abs(ref)).item(Float.self)
            let relL2 = sqrt(MLX.sum(diff * diff).item(Float.self) / n) / scale
            let relMax = MLX.max(diff).item(Float.self) / scale
            XCTAssertLessThan(relL2, 1e-4, "\(label): rel-l2")
            XCTAssertLessThan(relMax, 1e-4, "\(label): rel-max")
        }
        check(out.p!, tsData, tsShape, "p time series")
        check(out.pFinal!, pfData, pfShape, "p_final")
    }
}

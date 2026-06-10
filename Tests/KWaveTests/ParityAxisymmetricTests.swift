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

    private struct Ref {
        let grid: KWaveGrid
        let c0: Double, rho0: Double, pml: Int
        let f: HDF5File
    }

    private func load() throws -> Ref {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_as.py")
        }
        let f = try HDF5File(open: refPath)
        var grid = KWaveGrid(nx: Int(try f.readFloatDataset("Nx").data[0]),
                             dx: Double(try f.readFloatDataset("dx").data[0]),
                             ny: Int(try f.readFloatDataset("Ny").data[0]),
                             dy: Double(try f.readFloatDataset("dy").data[0]))
        grid.setTime(nt: Int(try f.readFloatDataset("Nt").data[0]),
                     dt: Double(try f.readFloatDataset("dt").data[0]))
        return Ref(grid: grid,
                   c0: Double(try f.readFloatDataset("c0").data[0]),
                   rho0: Double(try f.readFloatDataset("rho0").data[0]),
                   pml: Int(try f.readFloatDataset("pml").data[0]), f: f)
    }

    private func runCase(_ r: Ref, source: KWaveSource, refKey: String) throws {
        let (maskShape, maskData) = try r.f.readFloatDataset("mask")
        let sensor = KWaveSensor(mask: MLXArray(maskData).reshaped(maskShape))
        var opts = SimulationOptions()
        opts.pmlSize = .uniform(r.pml)
        opts.smoothP0 = false        // the stored p0 is already smoothed.

        let out = kspaceFirstOrderAS(grid: r.grid,
                                     medium: KWaveMedium(soundSpeed: r.c0, density: r.rho0),
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
        let (tsShape, tsData) = try r.f.readFloatDataset("p_ts_\(refKey)")
        let (pfShape, pfData) = try r.f.readFloatDataset("p_final_\(refKey)")
        check(out.p!, tsData, tsShape, "\(refKey): p time series")
        check(out.pFinal!, pfData, pfShape, "\(refKey): p_final")
    }

    func testAxisymmetricIVPParity() throws {
        let r = try load()
        let (p0Shape, p0Data) = try r.f.readFloatDataset("p0")
        var source = KWaveSource()
        source.p0 = MLXArray(p0Data).reshaped(p0Shape)
        try runCase(r, source: source, refKey: "ivp")
    }

    func testAxisymmetricAdditivePressureSourceParity() throws {
        let r = try load()
        let (smShape, smData) = try r.f.readFloatDataset("src_mask")
        let sig = try r.f.readFloatDataset("sig").data
        var source = KWaveSource()
        source.pMask = MLXArray(smData).reshaped(smShape)
        source.p = MLXArray(sig)
        source.pMode = .additive
        try runCase(r, source: source, refKey: "add")
    }

    func testAxisymmetricDirichletPressureSourceParity() throws {
        let r = try load()
        let (smShape, smData) = try r.f.readFloatDataset("src_mask")
        let sig = try r.f.readFloatDataset("sig").data
        var source = KWaveSource()
        source.pMask = MLXArray(smData).reshaped(smShape)
        source.p = MLXArray(sig)
        source.pMode = .dirichlet
        try runCase(r, source: source, refKey: "dir")
    }
}

import XCTest
import MLX
@testable import KWave

/// Parity for the extended sensor recording fields against a k-wave-python (NumPy backend) 2D
/// reference: collocated velocity time series (`ux`/`uy`), aggregates (`p_max`/`p_min`/`p_rms`,
/// `u_max`/`u_rms`), and time-averaged intensity (`Ix_avg`/`Iy_avg`). A tone-burst pressure source
/// radiates to a line sensor; the Swift solver mirrors the NumPy recording, so tolerances are tight.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_record.py`.
final class ParityRecordTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_2d_record.h5").path
    }

    /// Assert `mine` matches `ref` (same shape) to a tight relative l2/max tolerance.
    private func assertClose(_ mine: MLXArray?, _ refData: [Float], _ refShape: [Int],
                             _ label: String, tol: Float = 1e-4) {
        guard let mine else { return XCTFail("\(label): output field is nil") }
        let ref = MLXArray(refData).reshaped(refShape)
        XCTAssertEqual(mine.shape, refShape, "\(label): shape mismatch")
        let diff = MLX.abs(mine - ref)
        let n = Float(refData.count)
        let relL2 = sqrt(MLX.sum(diff * diff).item(Float.self) / n)
            / sqrt(MLX.sum(ref * ref).item(Float.self) / n)
        let relMax = MLX.max(diff).item(Float.self) / MLX.max(MLX.abs(ref)).item(Float.self)
        XCTAssertLessThan(relL2, tol, "\(label): rel-l2")
        XCTAssertLessThan(relMax, tol, "\(label): rel-max")
    }

    func test2DRecordingFieldsParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_record.py")
        }
        let f = try HDF5File(open: refPath)

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
        let (maskShape, maskData) = try f.readFloatDataset("mask")
        XCTAssertEqual(maskShape, [nx, ny])

        var grid = KWaveGrid(nx: nx, dx: dx, ny: ny, dy: dx)
        grid.setTime(nt: nt, dt: dt)

        var maskHost = [Float](repeating: 0, count: nx * ny)
        maskHost[sx * ny + sy] = 1
        var source = KWaveSource()
        source.pMask = MLXArray(maskHost).reshaped([nx, ny])
        source.p = MLXArray(signal)
        source.pMode = .additive

        var sensor = KWaveSensor(mask: MLXArray(maskData).reshaped([nx, ny]))
        sensor.record = [.p, .ux, .uy, .pMax, .pMin, .pRms, .uMax, .uRms, .iAvg]

        var opts = SimulationOptions()
        opts.pmlSize = .uniform(pml)
        opts.smoothP0 = false

        let out = kspaceFirstOrder(grid: grid, medium: KWaveMedium(soundSpeed: c0, density: rho0),
                                   source: source, sensor: sensor, options: opts)

        func ref(_ name: String) throws -> ([Float], [Int]) {
            let (shape, data) = try f.readFloatDataset(name)
            return (data, shape)
        }

        let (pD, pS) = try ref("p");           assertClose(out.p, pD, pS, "p")
        let (uxD, uxS) = try ref("ux");        assertClose(out.ux, uxD, uxS, "ux")
        let (uyD, uyS) = try ref("uy");        assertClose(out.uy, uyD, uyS, "uy")
        let (pmaxD, pmaxS) = try ref("p_max"); assertClose(out.pMax, pmaxD, pmaxS, "p_max")
        let (pminD, pminS) = try ref("p_min"); assertClose(out.pMin, pminD, pminS, "p_min")
        let (prmsD, prmsS) = try ref("p_rms"); assertClose(out.pRms, prmsD, prmsS, "p_rms")
        let (uxmD, uxmS) = try ref("ux_max");  assertClose(out.uxMax, uxmD, uxmS, "ux_max")
        let (uymD, uymS) = try ref("uy_max");  assertClose(out.uyMax, uymD, uymS, "uy_max")
        let (uxrD, uxrS) = try ref("ux_rms");  assertClose(out.uxRms, uxrD, uxrS, "ux_rms")
        let (uyrD, uyrS) = try ref("uy_rms");  assertClose(out.uyRms, uyrD, uyrS, "uy_rms")
        // Intensity involves an extra Fourier half-sample shift + product; allow a touch more slack.
        let (ixD, ixS) = try ref("Ix_avg");    assertClose(out.ixAvg, ixD, ixS, "Ix_avg", tol: 1e-3)
        let (iyD, iyS) = try ref("Iy_avg");    assertClose(out.iyAvg, iyD, iyS, "Iy_avg", tol: 1e-3)
    }
}

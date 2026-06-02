import XCTest
import MLX
@testable import KWave

/// Parity for photoacoustic FFT planar reconstruction (`kspacePlaneRecon`) against k-wave-python.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_planerecon.py`.
final class ParityPlaneReconTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_planerecon.h5").path
    }

    func testKSpacePlaneReconParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_planerecon.py")
        }
        let f = try HDF5File(open: refPath)
        let dy = Double(try f.readFloatDataset("dy").data[0])
        let dz = Double(try f.readFloatDataset("dz").data[0])
        let dt = Double(try f.readFloatDataset("dt").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let (pShape, pData) = try f.readFloatDataset("p_yzt")
        let (rShape, rData) = try f.readFloatDataset("recon")

        let pYzt = MLXArray(pData).reshaped(pShape)
        let mine = kspacePlaneRecon(p: pYzt, dy: dy, dz: dz, dt: dt, c: c0, dataOrder: "yzt")
        XCTAssertEqual(mine.shape, rShape)

        let ref = MLXArray(rData).reshaped(rShape)
        let diff = MLX.abs(mine - ref)
        let n = Float(rData.count)
        let scale = MLX.max(MLX.abs(ref)).item(Float.self)
        let relL2 = sqrt(MLX.sum(diff * diff).item(Float.self) / n) / scale
        let relMax = MLX.max(diff).item(Float.self) / scale
        XCTAssertLessThan(relL2, 1e-4, "rel-l2")
        XCTAssertLessThan(relMax, 1e-4, "rel-max")
    }
}

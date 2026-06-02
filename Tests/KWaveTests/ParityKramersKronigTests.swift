import XCTest
import MLX
@testable import KWave

/// Parity for `powerLawKramersKronig` (power-law-absorption sound-speed dispersion) vs k-wave-python.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_kk.py`.
final class ParityKramersKronigTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_kk.h5").path
    }

    func testPowerLawKramersKronigParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_kk.py")
        }
        let f = try HDF5File(open: refPath)
        let w0 = Double(try f.readFloatDataset("w0").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let (wShape, wData) = try f.readFloatDataset("w")
        let w = MLXArray(wData).reshaped(wShape)

        for key in ["y0p5", "y1", "y1p5"] {
            let y = Double(try f.readFloatDataset(key + "_y").data[0])
            let a0 = Double(try f.readFloatDataset(key + "_a0").data[0])
            let (refShape, refData) = try f.readFloatDataset(key)

            let mine = powerLawKramersKronig(w: w, w0: w0, c0: c0, a0: a0, y: y)
            let ref = MLXArray(refData).reshaped(refShape)
            let relMax = MLX.max(MLX.abs(mine - ref)).item(Float.self) / MLX.max(MLX.abs(ref)).item(Float.self)
            XCTAssertLessThan(relMax, 1e-4, "\(key)")
        }
    }
}

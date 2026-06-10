import XCTest
import MLX
@testable import KWave

/// Parity for discrete trig transforms (`dct`/`dst` types I-IV) against `scipy.fft` (unnormalized
/// FFTW convention) — the foundation of the axisymmetric solver's symmetry machinery.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_dtt.py`.
final class ParityDTTTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_dtt.h5").path
    }

    private func assertClose(_ mine: MLXArray, _ refData: [Float], _ refShape: [Int], _ label: String) {
        let ref = MLXArray(refData).reshaped(refShape)
        XCTAssertEqual(mine.shape, refShape, "\(label): shape")
        let maxErr = MLX.max(MLX.abs(mine - ref)).item(Float.self)
        let scale = MLX.max(MLX.abs(ref)).item(Float.self)
        XCTAssertLessThan(maxErr / scale, 1e-5, "\(label): max rel error")
    }

    func testDTTParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_dtt.py")
        }
        let f = try HDF5File(open: refPath)
        let x1 = MLXArray(try f.readFloatDataset("x1").data)
        let x2 = MLXArray(try f.readFloatDataset("x2").data)

        for t in 1...4 {
            let cType = DiscreteCosineType(rawValue: t)!
            let sType = DiscreteSineType(rawValue: t)!
            for (name, x) in [("x1", x1), ("x2", x2)] {
                let (cShape, cData) = try f.readFloatDataset("dct\(t)_\(name)")
                assertClose(dct(x, type: cType), cData, cShape, "dct\(t) \(name)")
                let (sShape, sData) = try f.readFloatDataset("dst\(t)_\(name)")
                assertClose(dst(x, type: sType), sData, sShape, "dst\(t) \(name)")
            }
        }

        // 2D axis cases.
        let (x3Shape, x3Data) = try f.readFloatDataset("x3")
        let x3 = MLXArray(x3Data).reshaped(x3Shape)
        let (c0Shape, c0Data) = try f.readFloatDataset("dct2_x3_ax0")
        assertClose(dct(x3, type: .ii, axis: 0), c0Data, c0Shape, "dct2 x3 axis0")
        let (s1Shape, s1Data) = try f.readFloatDataset("dst3_x3_ax1")
        assertClose(dst(x3, type: .iii, axis: 1), s1Data, s1Shape, "dst3 x3 axis1")
    }
}

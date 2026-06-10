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

    /// Parity for `makeDTTDim` (DTT wavenumber vectors + implied periods) vs k-wave-python
    /// `kgrid.kx_vec_dtt`, all 8 transform types.
    ///
    /// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_dttdim.py`.
    func testMakeDTTDimParity() throws {
        let path = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_dttdim.h5").path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_dttdim.py")
        }
        let f = try HDF5File(open: path)
        let n = Int(try f.readFloatDataset("N").data[0])
        let dx = Double(try f.readFloatDataset("dx").data[0])

        let types: [(String, DTTType)] = [
            ("dct1", .cosine(.i)), ("dct2", .cosine(.ii)), ("dct3", .cosine(.iii)), ("dct4", .cosine(.iv)),
            ("dst1", .sine(.i)), ("dst2", .sine(.ii)), ("dst3", .sine(.iii)), ("dst4", .sine(.iv)),
        ]
        for (key, type) in types {
            let refK = try f.readFloatDataset("\(key)_k").data
            let refM = Int(try f.readFloatDataset("\(key)_M").data[0])
            let (kVec, m) = makeDTTDim(n: n, d: dx, type: type)
            XCTAssertEqual(m, refM, "\(key): M")
            XCTAssertEqual(kVec.count, refK.count, "\(key): count")
            for (mine, ref) in zip(kVec, refK) {
                XCTAssertEqual(Float(mine), ref, accuracy: max(abs(ref) * 1e-5, 1e-3), key)
            }
        }
    }
}

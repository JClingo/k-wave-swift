import XCTest
import MLX
@testable import KWave

/// Parity for `acousticFieldPropagator` (exact CW Green's-function propagation). k-wave-python has
/// no port, so the reference is an independent NumPy float64 implementation of the same MATLAB
/// formulas (`acousticFieldPropagator.m`); agreement cross-checks both implementations.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_afp.py`.
final class ParityAFPTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_afp.h5").path
    }

    func testAcousticFieldPropagatorParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_afp.py")
        }
        let f = try HDF5File(open: refPath)
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let f0 = Double(try f.readFloatDataset("f0").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])

        func checkComplex(_ p: MLXArray, _ reKey: String, _ imKey: String, tol: Float) throws {
            let (reShape, reData) = try f.readFloatDataset(reKey)
            let (_, imData) = try f.readFloatDataset(imKey)
            XCTAssertEqual(p.shape, reShape, "\(reKey): shape")
            let refRe = MLXArray(reData).reshaped(reShape)
            let refIm = MLXArray(imData).reshaped(reShape)
            let refMag = MLX.sqrt(refRe * refRe + refIm * refIm)
            let scale = MLX.max(refMag).item(Float.self)
            let dRe = MLX.max(MLX.abs(p.realPart() - refRe)).item(Float.self)
            let dIm = MLX.max(MLX.abs(p.imaginaryPart() - refIm)).item(Float.self)
            XCTAssertLessThan(max(dRe, dIm) / scale, tol, reKey)
        }

        // 2D, ramped (default) and unramped.
        let (a2Shape, a2Data) = try f.readFloatDataset("amp2")
        let (_, ph2Data) = try f.readFloatDataset("phase2")
        let amp2 = MLXArray(a2Data).reshaped(a2Shape)
        let phase2 = MLXArray(ph2Data).reshaped(a2Shape)
        let (p2, _, _) = acousticFieldPropagator(ampIn: amp2, phaseIn: phase2,
                                                 dx: dx, f0: f0, c0: c0)
        try checkComplex(p2, "p2_re", "p2_im", tol: 1e-4)
        let (p2nr, _, _) = acousticFieldPropagator(ampIn: amp2, phaseIn: phase2,
                                                   dx: dx, f0: f0, c0: c0, useRamp: false)
        try checkComplex(p2nr, "p2nr_re", "p2nr_im", tol: 1e-4)

        // 3D plane source (zero phase).
        let (a3Shape, a3Data) = try f.readFloatDataset("amp3")
        let amp3 = MLXArray(a3Data).reshaped(a3Shape)
        let (p3, _, _) = acousticFieldPropagator(ampIn: amp3, phaseIn: MLXArray(Float(0)),
                                                 dx: dx, f0: f0, c0: c0)
        try checkComplex(p3, "p3_re", "p3_im", tol: 1e-4)
    }
}

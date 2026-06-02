import XCTest
import MLX
@testable import KWave

/// Parity for CW angular-spectrum projection (`angularSpectrumCW`) against k-wave-python
/// `angular_spectrum_cw`: a disc piston source projected to several parallel planes.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_aspectrum.py`.
final class ParityAngularSpectrumTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_aspectrum.h5").path
    }

    func testAngularSpectrumCWParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_aspectrum.py")
        }
        let f = try HDF5File(open: refPath)
        let nx = Int(try f.readFloatDataset("Nx").data[0])
        let ny = Int(try f.readFloatDataset("Ny").data[0])
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let f0 = Double(try f.readFloatDataset("f0").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let zpos = try f.readFloatDataset("zpos").data.map { Double($0) }
        let (inShape, inData) = try f.readFloatDataset("input")
        let (reShape, reData) = try f.readFloatDataset("out_re")
        let (imShape, imData) = try f.readFloatDataset("out_im")
        XCTAssertEqual(inShape, [nx, ny])

        let input = MLXArray(inData).reshaped([nx, ny])
        let out = angularSpectrumCW(inputPlane: input, dx: dx, zPos: zpos, f0: f0,
                                    soundSpeed: c0, angularRestriction: true)
        XCTAssertEqual(out.shape, reShape)

        let refRe = MLXArray(reData).reshaped(reShape)
        let refIm = MLXArray(imData).reshaped(imShape)
        let refMag = MLX.sqrt(refRe * refRe + refIm * refIm)
        let scale = MLX.max(refMag).item(Float.self)

        // Compare the complex field via real & imaginary parts.
        let dRe = MLX.abs(out.realPart() - refRe)
        let dIm = MLX.abs(out.imaginaryPart() - refIm)
        let maxErr = max(MLX.max(dRe).item(Float.self), MLX.max(dIm).item(Float.self))
        let n = Float(reData.count)
        let l2 = sqrt((MLX.sum(dRe * dRe).item(Float.self) + MLX.sum(dIm * dIm).item(Float.self)) / n)
        XCTAssertLessThan(maxErr / scale, 1e-4, "max")
        XCTAssertLessThan(l2 / scale, 1e-4, "l2")
    }
}

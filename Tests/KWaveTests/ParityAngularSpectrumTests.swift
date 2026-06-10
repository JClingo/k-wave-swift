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

    // MARK: - Broadband

    private var broadbandRefPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_angspec.h5").path
    }

    private func assertClose(_ mine: MLXArray, _ refData: [Float], _ refShape: [Int],
                             _ label: String, tol: Float = 1e-4) {
        let ref = MLXArray(refData).reshaped(refShape)
        XCTAssertEqual(mine.shape, refShape, "\(label): shape")
        let maxErr = MLX.max(MLX.abs(mine - ref)).item(Float.self)
        let scale = MLX.max(MLX.abs(ref)).item(Float.self)
        XCTAssertLessThan(maxErr / scale, tol, "\(label): max rel error")
    }

    /// Parity for the broadband (time-domain) `angularSpectrum` against k-wave-python. Odd `Nt`
    /// only — the upstream even-`Nt` spectrum rebuild drops a bin (`[1:-2]` vs MATLAB `2:end-1`).
    ///
    /// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_angspec.py`.
    func testBroadbandAngularSpectrumParity() throws {
        guard FileManager.default.fileExists(atPath: broadbandRefPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_angspec.py")
        }
        let f = try HDF5File(open: broadbandRefPath)
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let dt = Double(try f.readFloatDataset("dt").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let zVals = try f.readFloatDataset("z_vals").data.map { Double($0) }
        let (inShape, inData) = try f.readFloatDataset("input")
        let input = MLXArray(inData).reshaped(inShape)

        let (pmaxShape, pmaxData) = try f.readFloatDataset("pmax")
        let (ptShape, ptData) = try f.readFloatDataset("ptime_z1")

        let (pressureMax, pressureTime) = angularSpectrum(
            inputPlane: input, dx: dx, dt: dt, zPos: zVals, soundSpeed: c0,
            recordTimeSeries: true)
        assertClose(pressureMax, pmaxData, pmaxShape, "pressure_max")

        // Time series at z = zVals[1].
        let nt = inShape[2]
        let ptZ1 = pressureTime![0..<inShape[0], 0..<inShape[1], 0..<nt, 1..<2]
            .reshaped([inShape[0], inShape[1], nt])
        assertClose(ptZ1, ptData, ptShape, "pressure_time @ z1")

        // Grid-expansion case.
        let (geShape, geData) = try f.readFloatDataset("pmax_ge")
        let (pmaxGE, _) = angularSpectrum(
            inputPlane: input, dx: dx, dt: dt, zPos: [1e-3], soundSpeed: c0, gridExpansion: 4)
        let geSlice = pmaxGE[0..<geShape[0], 0..<geShape[1], 0..<1].reshaped(geShape)
        assertClose(geSlice, geData, geShape, "pressure_max grid_expansion")
    }

    /// Parity for the absorbing branch (Eq. 11). The upstream Python absorbing path is dead code
    /// (dict attribute access crashes), so the reference reproduces the loop with that line fixed.
    ///
    /// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_angspec_abs.py`.
    func testBroadbandAngularSpectrumAbsorbingParity() throws {
        let path = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_angspec_abs.h5").path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_angspec_abs.py")
        }
        let f = try HDF5File(open: path)
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let dt = Double(try f.readFloatDataset("dt").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let z = Double(try f.readFloatDataset("z").data[0])
        let alphaCoeff = Double(try f.readFloatDataset("alpha_coeff").data[0])
        let alphaPower = Double(try f.readFloatDataset("alpha_power").data[0])
        let (inShape, inData) = try f.readFloatDataset("input")
        let (pmaxShape, pmaxData) = try f.readFloatDataset("pmax")
        let (ptShape, ptData) = try f.readFloatDataset("ptime")

        let (pressureMax, pressureTime) = angularSpectrum(
            inputPlane: MLXArray(inData).reshaped(inShape), dx: dx, dt: dt, zPos: [z],
            soundSpeed: c0, alphaCoeff: alphaCoeff, alphaPower: alphaPower,
            recordTimeSeries: true)

        let pmaxSlice = pressureMax[0..<pmaxShape[0], 0..<pmaxShape[1], 0..<1].reshaped(pmaxShape)
        assertClose(pmaxSlice, pmaxData, pmaxShape, "absorbing pressure_max")
        let ptSlice = pressureTime![0..<ptShape[0], 0..<ptShape[1], 0..<ptShape[2], 0..<1]
            .reshaped(ptShape)
        assertClose(ptSlice, ptData, ptShape, "absorbing pressure_time")
    }
}

import XCTest
import MLX
@testable import KWave

/// Parity for `focusedBowlONeil` (O'Neil analytic focused-bowl pressure) against k-wave-python.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_oneil.py`.
final class ParityONeilTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_oneil.h5").path
    }

    private func assertClose(_ mine: MLXArray, _ refData: [Float], _ refShape: [Int], _ label: String) {
        let ref = MLXArray(refData).reshaped(refShape)
        XCTAssertEqual(mine.shape, refShape, "\(label): shape")
        let maxErr = MLX.max(MLX.abs(mine - ref)).item(Float.self)
        let scale = MLX.max(MLX.abs(ref)).item(Float.self)
        XCTAssertLessThan(maxErr / scale, 1e-4, "\(label): max rel error")
    }

    func testFocusedBowlONeilParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_oneil.py")
        }
        let f = try HDF5File(open: refPath)
        let radius = Double(try f.readFloatDataset("radius").data[0])
        let diameter = Double(try f.readFloatDataset("diameter").data[0])
        let velocity = Double(try f.readFloatDataset("velocity").data[0])
        let freq = Double(try f.readFloatDataset("freq").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let rho0 = Double(try f.readFloatDataset("rho0").data[0])
        let ax = try f.readFloatDataset("ax").data.map { Double($0) }
        let lat = try f.readFloatDataset("lat").data.map { Double($0) }

        let (axial, lateral, axialC) = focusedBowlONeil(
            radius: radius, diameter: diameter, velocity: velocity, frequency: freq,
            soundSpeed: c0, density: rho0, axialPositions: ax, lateralPositions: lat)

        let (paShape, paData) = try f.readFloatDataset("pa")
        let (plShape, plData) = try f.readFloatDataset("pl")
        assertClose(axial!, paData, paShape, "axial")
        assertClose(lateral!, plData, plShape, "lateral")

        let (reShape, reData) = try f.readFloatDataset("pac_re")
        let (_, imData) = try f.readFloatDataset("pac_im")
        assertClose(axialC!.realPart(), reData, reShape, "axialComplex.re")
        assertClose(axialC!.imaginaryPart(), imData, reShape, "axialComplex.im")
    }
}

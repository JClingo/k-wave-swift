import XCTest
import MLX
@testable import KWave

/// Parity for `focusedAnnulusONeil` (O'Neil analytic focused-annulus axial pressure) vs k-wave-python.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_annulus.py`.
final class ParityAnnulusTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_annulus.h5").path
    }

    func testFocusedAnnulusONeilParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_annulus.py")
        }
        let f = try HDF5File(open: refPath)
        let radius = Double(try f.readFloatDataset("radius").data[0])
        let freq = Double(try f.readFloatDataset("freq").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let rho0 = Double(try f.readFloatDataset("rho0").data[0])
        let (diaShape, diaData) = try f.readFloatDataset("diameters")
        let amplitude = try f.readFloatDataset("amplitude").data.map { Double($0) }
        let phase = try f.readFloatDataset("phase").data.map { Double($0) }
        let ax = try f.readFloatDataset("ax").data.map { Double($0) }
        let (paShape, paData) = try f.readFloatDataset("pa")

        let mine = focusedAnnulusONeil(
            radius: radius, diameters: MLXArray(diaData).reshaped(diaShape),
            amplitude: amplitude, phase: phase, frequency: freq, soundSpeed: c0, density: rho0,
            axialPositions: ax)

        let ref = MLXArray(paData).reshaped(paShape)
        XCTAssertEqual(mine.shape, paShape)
        let relMax = MLX.max(MLX.abs(mine - ref)).item(Float.self) / MLX.max(MLX.abs(ref)).item(Float.self)
        XCTAssertLessThan(relMax, 1e-4, "annulus axial")
    }
}

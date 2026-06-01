import XCTest
import MLX
@testable import KWave

/// Parity for Cartesian point-set geometry (`makeCartCircle`, `makeCartRect`) against k-wave-python.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_cartshapes.py`.
final class ParityCartShapesTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_cartshapes.h5").path
    }

    private func assertClose(_ mine: MLXArray, _ refData: [Float], _ refShape: [Int], _ label: String) {
        let ref = MLXArray(refData).reshaped(refShape)
        XCTAssertEqual(mine.shape, refShape, "\(label): shape")
        let maxErr = MLX.max(MLX.abs(mine - ref)).item(Float.self)
        let scale = MLX.max(MLX.abs(ref)).item(Float.self)
        XCTAssertLessThan(maxErr / scale, 1e-5, "\(label): max rel error")
    }

    func testMakeCartCircleAndRectParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_cartshapes.py")
        }
        let f = try HDF5File(open: refPath)

        let (cfShape, cfData) = try f.readFloatDataset("circle_full")
        assertClose(makeCartCircle(radius: 5e-3, numPoints: 30, center: (1e-3, -2e-3)),
                    cfData, cfShape, "circle_full")

        let (caShape, caData) = try f.readFloatDataset("circle_arc")
        assertClose(makeCartCircle(radius: 4e-3, numPoints: 20, center: (0, 0), arcAngle: .pi),
                    caData, caShape, "circle_arc")

        let (rpShape, rpData) = try f.readFloatDataset("rect_plain")
        assertClose(makeCartRect(center: (0, 0), lx: 4e-3, ly: 2e-3, numPoints: 50),
                    rpData, rpShape, "rect_plain")

        let (rrShape, rrData) = try f.readFloatDataset("rect_rot")
        assertClose(makeCartRect(center: (1e-3, 2e-3), lx: 4e-3, ly: 2e-3, theta: 30, numPoints: 50),
                    rrData, rrShape, "rect_rot")
    }
}

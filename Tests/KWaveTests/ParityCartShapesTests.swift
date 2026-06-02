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

    func testMakeCartArcParity() throws {
        let path = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_cartarc.h5").path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_shapes.py")
        }
        let f = try HDF5File(open: path)
        let (aShape, aData) = try f.readFloatDataset("arc")
        assertClose(makeCartArc(arcPos: (0, 0), radius: 8e-3, diameter: 6e-3,
                                focusPos: (0, 10e-3), numPoints: 25),
                    aData, aShape, "cart_arc")
    }

    func testMakeCartSphereParity() throws {
        let path = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_cartsphere.h5").path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_shapes.py")
        }
        let f = try HDF5File(open: path)
        let (sShape, sData) = try f.readFloatDataset("sphere")
        assertClose(makeCartSphere(radius: 5e-3, numPoints: 40, center: (1e-3, -2e-3, 3e-3)),
                    sData, sShape, "cart_sphere")
    }

    func testMakeArcParity() throws {
        let arcRef = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_arc.h5").path
        guard FileManager.default.fileExists(atPath: arcRef) else {
            throw XCTSkip("reference not generated")
        }
        let f = try HDF5File(open: arcRef)
        let nx = Int(try f.readFloatDataset("Nx").data[0])
        let ny = Int(try f.readFloatDataset("Ny").data[0])
        let ax = Int(try f.readFloatDataset("ax").data[0])
        let ay = Int(try f.readFloatDataset("ay").data[0])
        let fx = Int(try f.readFloatDataset("fx").data[0])
        let fy = Int(try f.readFloatDataset("fy").data[0])
        let radius = Double(try f.readFloatDataset("radius").data[0])
        let diameter = Int(try f.readFloatDataset("diameter").data[0])
        let (arcShape, arcData) = try f.readFloatDataset("arc")

        // Python positions are 1-based; makeArc takes 0-based grid indices.
        let mine = makeArc(nx: nx, ny: ny, arcPos: (ax - 1, ay - 1), radius: radius,
                           diameter: diameter, focusPos: (fx - 1, fy - 1))
        let ref = MLXArray(arcData).reshaped(arcShape)
        XCTAssertEqual(mine.shape, arcShape)
        XCTAssertEqual(MLX.max(MLX.abs(mine - ref)).item(Float.self), 0, "arc: exact binary match")
    }
}

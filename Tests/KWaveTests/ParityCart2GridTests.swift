import XCTest
import MLX
@testable import KWave

/// Parity for `cart2grid` (Cartesian points → nearest-node binary grid mask) against k-wave-python.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_cart2grid.py`.
final class ParityCart2GridTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_cart2grid.h5").path
    }

    func testCart2GridParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_cart2grid.py")
        }
        let f = try HDF5File(open: refPath)

        let (c2Shape, c2Data) = try f.readFloatDataset("c2")
        let (m2Shape, m2Data) = try f.readFloatDataset("m2")
        let g2 = KWaveGrid(nx: 50, dx: 1e-4, ny: 50, dy: 1e-4)
        let mine2 = cart2grid(grid: g2, cartData: MLXArray(c2Data).reshaped(c2Shape))
        XCTAssertEqual(mine2.shape, m2Shape)
        XCTAssertEqual(MLX.max(MLX.abs(mine2 - MLXArray(m2Data).reshaped(m2Shape))).item(Float.self), 0, "2D")

        let (c3Shape, c3Data) = try f.readFloatDataset("c3")
        let (m3Shape, m3Data) = try f.readFloatDataset("m3")
        let g3 = KWaveGrid(nx: 30, dx: 1e-4, ny: 30, dy: 1e-4, nz: 30, dz: 1e-4)
        let mine3 = cart2grid(grid: g3, cartData: MLXArray(c3Data).reshaped(c3Shape))
        XCTAssertEqual(mine3.shape, m3Shape)
        XCTAssertEqual(MLX.max(MLX.abs(mine3 - MLXArray(m3Data).reshaped(m3Shape))).item(Float.self), 0, "3D")
    }
}

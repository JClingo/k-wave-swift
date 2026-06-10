import XCTest
import MLX
@testable import KWave

/// Parity for `offGridPoints` (exact band-limited off-grid source spreading) against k-wave-python's
/// `get_delta_bli` primitive (off_grid_points itself is broken upstream for the exact path).
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_offgrid.py`.
final class ParityOffGridTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_offgrid.h5").path
    }

    private func assertClose(_ mine: MLXArray, _ refData: [Float], _ refShape: [Int], _ label: String) {
        let ref = MLXArray(refData).reshaped(refShape)
        XCTAssertEqual(mine.shape, refShape, "\(label): shape")
        let maxErr = MLX.max(MLX.abs(mine - ref)).item(Float.self)
        let scale = MLX.max(MLX.abs(ref)).item(Float.self)
        XCTAssertLessThan(maxErr / scale, 1e-4, "\(label): max rel error")
    }

    func testOffGridPointsParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_offgrid.py")
        }
        let f = try HDF5File(open: refPath)

        // 2D: grid [40, 32], dx = 1e-4.
        let (pts2Shape, pts2Data) = try f.readFloatDataset("pts2")
        let scale2 = try f.readFloatDataset("scale2").data.map { Double($0) }
        let (m2Shape, m2Data) = try f.readFloatDataset("m2")
        let g2 = KWaveGrid(nx: 40, dx: 1e-4, ny: 32, dy: 1e-4)
        let mine2 = offGridPoints(grid: g2, points: MLXArray(pts2Data).reshaped(pts2Shape), scale: scale2)
        assertClose(mine2, m2Data, m2Shape, "2D")

        // 3D: grid [24, 20, 16], dx = 1e-4.
        let (pts3Shape, pts3Data) = try f.readFloatDataset("pts3")
        let scale3 = try f.readFloatDataset("scale3").data.map { Double($0) }
        let (m3Shape, m3Data) = try f.readFloatDataset("m3")
        let g3 = KWaveGrid(nx: 24, dx: 1e-4, ny: 20, dy: 1e-4, nz: 16, dz: 1e-4)
        let mine3 = offGridPoints(grid: g3, points: MLXArray(pts3Data).reshaped(pts3Shape), scale: scale3)
        assertClose(mine3, m3Data, m3Shape, "3D")
    }

    /// Truncated-sinc path (`bli_tolerance = 0.1`, k-Wave's default — the `kWaveArray` path),
    /// against the working python `off_grid_points` default path (a true oracle, unlike the
    /// broken exact path). One 2D point lies on-grid in y, exercising the tolStar axis collapse.
    func testOffGridPointsTruncatedParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_offgrid.py")
        }
        let f = try HDF5File(open: refPath)

        let (pts2Shape, pts2Data) = try f.readFloatDataset("pts2")
        let scale2 = try f.readFloatDataset("scale2").data.map { Double($0) }
        let (m2Shape, m2Data) = try f.readFloatDataset("m2_tol")
        let g2 = KWaveGrid(nx: 40, dx: 1e-4, ny: 32, dy: 1e-4)
        let mine2 = offGridPoints(grid: g2, points: MLXArray(pts2Data).reshaped(pts2Shape),
                                  scale: scale2, bliTolerance: 0.1)
        assertClose(mine2, m2Data, m2Shape, "2D tol")

        let (pts3Shape, pts3Data) = try f.readFloatDataset("pts3")
        let scale3 = try f.readFloatDataset("scale3").data.map { Double($0) }
        let (m3Shape, m3Data) = try f.readFloatDataset("m3_tol")
        let g3 = KWaveGrid(nx: 24, dx: 1e-4, ny: 20, dy: 1e-4, nz: 16, dz: 1e-4)
        let mine3 = offGridPoints(grid: g3, points: MLXArray(pts3Data).reshaped(pts3Shape),
                                  scale: scale3, bliTolerance: 0.1)
        assertClose(mine3, m3Data, m3Shape, "3D tol")
    }
}

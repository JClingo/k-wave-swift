import XCTest
import MLX
@testable import KWave

final class GridTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    func testGrid2DShapeAndWavenumbers() {
        let grid = KWaveGrid(nx: 8, dx: 1e-3, ny: 4, dy: 2e-3)
        XCTAssertEqual(grid.dim, 2)
        XCTAssertEqual(grid.totalGridPoints, 32)
        XCTAssertEqual(grid.kx.shape, [8, 4])
        XCTAssertEqual(grid.k.shape, [8, 4])
        // DC component of the wavenumber vector is zero.
        XCTAssertEqual(grid.kxVec[0].item(Float.self), 0, accuracy: 1e-6)
    }

    func testMakeTime() {
        var grid = KWaveGrid(nx: 16, dx: 1e-3, ny: 16, dy: 1e-3)
        grid.makeTime(soundSpeedMax: 1500, cfl: 0.3)
        XCTAssertGreaterThan(grid.nt, 0)
        XCTAssertEqual(grid.dt, 0.3 * 1e-3 / 1500, accuracy: 1e-12)
        XCTAssertEqual(grid.tArray.shape, [grid.nt])
    }
}

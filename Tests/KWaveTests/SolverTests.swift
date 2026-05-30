import XCTest
import MLX
@testable import KWave

final class SolverTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    // Circular reflection b[i] = a[(n-i) mod n] = roll(reverse(a), 1). This is the physical
    // mirror about grid index n/2, which the FFT-based spectral operators preserve exactly
    // (unlike plain array reversal, which mirrors about (n-1)/2).
    private func creflX(_ a: MLXArray) -> MLXArray { MLX.roll(a[.stride(by: -1), 0...], shift: 1, axis: 0) }
    private func creflY(_ a: MLXArray) -> MLXArray { MLX.roll(a[0..., .stride(by: -1)], shift: 1, axis: 1) }

    /// A centred-disc initial pressure on a homogeneous medium must evolve to a finite field that
    /// keeps its 4-fold (circular-reflection) symmetry while the wave stays clear of the PML.
    func testDiscIVPFiniteAndSymmetric() {
        // Even n so the disc centre (n/2) is a fixed point of circular reflection. Short tEnd keeps
        // the ring in the interior, where the reversal-symmetric PML profile is still ~1 and so
        // does not perturb the symmetry the spectral operators preserve.
        let n = 64
        let dx = 1e-4
        var grid = KWaveGrid(nx: n, dx: dx, ny: n, dy: dx)
        grid.makeTime(soundSpeedMax: 1500, cfl: 0.3, tEnd: 3e-7)

        let medium = KWaveMedium(soundSpeed: 1500, density: 1000)
        var source = KWaveSource()
        source.p0 = makeDisc(nx: n, ny: n, radius: 5) * 1.0

        let out = kspaceFirstOrder(grid: grid, medium: medium, source: source, sensor: KWaveSensor())
        let pf = out.pFinal!
        XCTAssertEqual(pf.shape, [n, n])

        let maxAbs = MLX.max(MLX.abs(pf)).item(Float.self)
        XCTAssertTrue(maxAbs.isFinite)
        XCTAssertGreaterThan(maxAbs, 0)

        let errX = MLX.max(MLX.abs(pf - creflX(pf))).item(Float.self) / maxAbs
        let errY = MLX.max(MLX.abs(pf - creflY(pf))).item(Float.self) / maxAbs
        XCTAssertLessThan(errX, 1e-3)
        XCTAssertLessThan(errY, 1e-3)
    }

    /// A line sensor must record a finite pressure time series of shape [numPoints, nt].
    func testSensorTimeSeriesShape() {
        let n = 48
        let dx = 1e-4
        var grid = KWaveGrid(nx: n, dx: dx, ny: n, dy: dx)
        grid.makeTime(soundSpeedMax: 1500, cfl: 0.3, tEnd: 4e-6)

        let medium = KWaveMedium(soundSpeed: 1500, density: 1000)
        var source = KWaveSource()
        source.p0 = makeDisc(nx: n, ny: n, radius: 4) * 1.0

        var maskHost = [Float](repeating: 0, count: n * n)
        let row = 8
        for col in 0..<n { maskHost[row * n + col] = 1 }
        var sensor = KWaveSensor()
        sensor.mask = MLXArray(maskHost).reshaped([n, n])

        let out = kspaceFirstOrder(grid: grid, medium: medium, source: source, sensor: sensor)
        let p = out.p!
        XCTAssertEqual(p.shape, [n, grid.nt])
        XCTAssertTrue(MLX.max(MLX.abs(p)).item(Float.self).isFinite)
    }
}

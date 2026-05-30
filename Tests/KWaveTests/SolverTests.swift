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
    private func crefl1D(_ a: MLXArray) -> MLXArray { MLX.roll(a[.stride(by: -1)], shift: 1, axis: 0) }

    /// A symmetric 1-D initial pressure pulse must evolve to a finite field that keeps its
    /// circular-reflection symmetry about index n/2 while the wave stays clear of the PML, and the
    /// per-step progress callback must fire once per time step.
    func test1DSymmetricIVPAndProgress() {
        let n = 64
        let dx = 1e-4
        var grid = KWaveGrid(nx: n, dx: dx)
        // Short tEnd keeps the pulse in the interior, clear of the (non-mirror-symmetric) staggered
        // PML, where the spectral operators preserve circular-reflection symmetry about index n/2.
        grid.makeTime(soundSpeedMax: 1500, cfl: 0.3, tEnd: 4e-7)

        let medium = KWaveMedium(soundSpeed: 1500, density: 1000)
        var source = KWaveSource()
        var p0 = [Float](repeating: 0, count: n)
        for i in (n / 2 - 3)...(n / 2 + 3) { p0[i] = 1 }   // symmetric about index n/2
        source.p0 = MLXArray(p0)

        var steps = 0
        var opts = SimulationOptions()
        opts.progress = { _, _ in steps += 1 }

        let out = kspaceFirstOrder(grid: grid, medium: medium, source: source,
                                   sensor: KWaveSensor(), options: opts)
        let pf = out.pFinal!
        XCTAssertEqual(pf.shape, [n])
        XCTAssertEqual(steps, grid.nt)

        let maxAbs = MLX.max(MLX.abs(pf)).item(Float.self)
        XCTAssertTrue(maxAbs.isFinite)
        XCTAssertGreaterThan(maxAbs, 0)
        let err = MLX.max(MLX.abs(pf - crefl1D(pf))).item(Float.self) / maxAbs
        XCTAssertLessThan(err, 1e-3)
    }

    /// A 1-D time-varying pressure source must drive a finite, nonzero sensor time series of
    /// shape [numPoints, nt].
    func test1DTimeVaryingSource() {
        let n = 96
        let dx = 1e-4
        var grid = KWaveGrid(nx: n, dx: dx)
        grid.makeTime(soundSpeedMax: 1500, cfl: 0.3, tEnd: 5e-6)

        let medium = KWaveMedium(soundSpeed: 1500, density: 1000)
        var source = KWaveSource()
        var maskHost = [Float](repeating: 0, count: n)
        maskHost[16] = 1
        source.pMask = MLXArray(maskHost)
        source.p = MLXArray(toneBurst(sampleFreq: 1 / grid.dt, signalFreq: 1e6, numCycles: 3).map { Float($0) })

        var sensorHost = [Float](repeating: 0, count: n)
        sensorHost[64] = 1
        var sensor = KWaveSensor()
        sensor.mask = MLXArray(sensorHost)

        let out = kspaceFirstOrder(grid: grid, medium: medium, source: source, sensor: sensor)
        let p = out.p!
        XCTAssertEqual(p.shape, [1, grid.nt])
        XCTAssertTrue(MLX.max(MLX.abs(p)).item(Float.self).isFinite)
        XCTAssertGreaterThan(MLX.max(MLX.abs(p)).item(Float.self), 0)
    }

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

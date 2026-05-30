import XCTest
import MLX
@testable import KWave

final class UtilTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    func testMakeCircleMatchesReference() {
        let c = makeCircle(nx: 11, ny: 11, radius: 3)
        let host = c.reshaped([121]).asArray(Float.self)
        XCTAssertEqual(Int(host.reduce(0, +)), 16)
        // A few midpoint-circle points (x,y) from k-wave-python make_circle.
        for (x, y) in [(2, 4), (3, 3), (4, 2), (5, 8), (8, 6)] {
            XCTAssertEqual(host[x * 11 + y], 1, "missing point (\(x),\(y))")
        }
    }

    func testMakeSphereMatchesReference() throws {
        let path = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_sphere.h5").path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_sphere.py")
        }
        let f = try HDF5File(open: path)
        let ncases = Int(try f.readFloatDataset("ncases").data[0])
        for i in 0..<ncases {
            let r = Double(try f.readFloatDataset("r_\(i)").data[0])
            let (shape, refData) = try f.readFloatDataset("sphere_\(i)")
            let n = shape[0]
            let mine = makeSphere(nx: n, ny: n, nz: n, radius: r)
            let ref = MLXArray(refData).reshaped(shape)
            let diff = MLX.sum(MLX.abs(mine - ref)).item(Float.self)
            XCTAssertEqual(diff, 0, "makeSphere N=\(n) r=\(r) differs at \(Int(diff)) voxels")
        }
    }

    func testApplyFilterMatchesReference() throws {
        let path = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_filter.h5").path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("reference not generated: run generate_reference_filter.py")
        }
        let f = try HDF5File(open: path)
        let fs = Double(try f.readFloatDataset("fs").data[0])
        let signal = try f.readFloatDataset("signal").data.map(Double.init)

        let lp = applyFilter(signal, fs: fs, cutoff: 1e6, type: .lowPass)
        let hp = applyFilter(signal, fs: fs, cutoff: 1e6, type: .highPass)
        let bp = applyFilter(signal, fs: fs, cutoff: 0, type: .bandPass, bandCutoff: (0.3e6, 1.5e6))
        let refLP = try f.readFloatDataset("lp").data
        let refHP = try f.readFloatDataset("hp").data
        let refBP = try f.readFloatDataset("bp").data
        func compare(_ a: [Double], _ b: [Float], _ name: String) {
            XCTAssertEqual(a.count, b.count, "\(name) length")
            for i in 0..<min(a.count, b.count) {
                XCTAssertEqual(a[i], Double(b[i]), accuracy: 1e-4, "\(name)[\(i)]")
            }
        }
        compare(lp, refLP, "lowPass")
        compare(hp, refHP, "highPass")
        compare(bp, refBP, "bandPass")
    }

    func testFilterTimeSeriesMatchesReference() throws {
        let path = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_filter.h5").path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("reference not generated: run generate_reference_filter.py")
        }
        let f = try HDF5File(open: path)
        let dt = Double(try f.readFloatDataset("dt").data[0])
        let nt = Int(try f.readFloatDataset("Nt").data[0])
        let sig2 = try f.readFloatDataset("sig2").data.map(Double.init)
        let refFTS = try f.readFloatDataset("fts").data

        var grid = KWaveGrid(nx: 64, dx: 1e-4, ny: 64, dy: 1e-4)
        grid.setTime(nt: nt, dt: dt)
        let out = filterTimeSeries(grid: grid,
                                   medium: KWaveMedium(soundSpeed: 1500.0, density: 1000.0),
                                   signal: sig2)
        XCTAssertEqual(out.count, refFTS.count)
        for i in 0..<min(out.count, refFTS.count) {
            XCTAssertEqual(out[i], Double(refFTS[i]), accuracy: 1e-4, "fts[\(i)]")
        }
    }

    func testColorMapMatchesReference() {
        let cm = getColorMap(numColors: 8)
        let expected: [RGB] = [
            (0.35, 0.4125, 0.475), (0.525, 0.65, 0.65), (0.7625, 0.825, 0.825), (1, 1, 1),
            (1, 1, 1), (1, 1, 0.5), (1, 1, 0), (1, 0, 0),
        ]
        XCTAssertEqual(cm.count, 8)
        for (a, b) in zip(cm, expected) {
            XCTAssertEqual(a.r, b.r, accuracy: 1e-4)
            XCTAssertEqual(a.g, b.g, accuracy: 1e-4)
            XCTAssertEqual(a.b, b.b, accuracy: 1e-4)
        }
    }

    func testGaussianFilterMatchesReference() {
        let sig = (0..<64).map { sin(2 * .pi * 1e6 * Double($0) / 10e6) }
        let out = gaussianFilter(sig, fs: 10e6, frequency: 1e6, bandwidth: 50)
        let expectedFirst = [0.329753, 0.342151, 0.336547, 0.269088, 0.096024]
        for (a, b) in zip(out.prefix(5), expectedFirst) {
            XCTAssertEqual(a, b, accuracy: 1e-4)
        }
        XCTAssertEqual(out.map(abs).max()!, 0.95103, accuracy: 1e-4)
    }

    func testExpandMatrixEdgeAndConstant() {
        let m = MLXArray([1, 2, 3, 4].map { Float($0) }).reshaped([2, 2])
        let edge = expandMatrix(m, padX: (1, 1), padY: (1, 1))
        XCTAssertEqual(edge.shape, [4, 4])
        let e = edge.reshaped([16]).asArray(Float.self)
        XCTAssertEqual(e[0], 1)   // top-left corner replicates m[0,0]
        XCTAssertEqual(e[15], 4)  // bottom-right replicates m[1,1]
        let con = expandMatrix(m, padX: (1, 1), padY: (1, 1), edgeVal: 0)
        XCTAssertEqual(con.reshaped([16]).asArray(Float.self)[0], 0)
    }

    func testCartGridRoundTrip() {
        var grid = KWaveGrid(nx: 21, dx: 1e-3, ny: 21, dy: 1e-3)
        grid.setTime(nt: 1, dt: 1)
        let pts: [(x: Double, y: Double)] = [(0, 0), (2e-3, -3e-3), (-5e-3, 4e-3)]
        let (mask, _) = cart2grid2D(grid: grid, points: pts)
        XCTAssertEqual(Int(MLX.sum(mask).item(Float.self)), 3)
        let (back, _) = grid2cart2D(grid: grid, mask: mask)
        // Every original point must reappear (nearest-neighbour is exact on-grid here).
        for p in pts {
            XCTAssertTrue(back.contains { abs($0.x - p.x) < 1e-9 && abs($0.y - p.y) < 1e-9 },
                          "missing \(p)")
        }
    }

    func testNeperRoundTrip() {
        let alphaDB = 0.75
        let np = db2neper(alphaDB, y: 1.5)
        XCTAssertEqual(neper2db(np, y: 1.5), alphaDB, accuracy: 1e-12)
    }

    func testGaussianPeak() {
        // Unit-magnitude Gaussian peaks at the mean with value `magnitude`.
        let g = gaussian([-1, 0, 1], magnitude: 1, mean: 0, variance: 1)
        XCTAssertEqual(g[1], 1, accuracy: 1e-12)
        XCTAssertEqual(g[0], g[2], accuracy: 1e-12)
        XCTAssertLessThan(g[0], g[1])
    }

    func testGetOptimalPMLSizeMinimisesMaxPrimeFactor() {
        // Brute-force reference for the same heuristic (min largest-prime-factor, ties → smallest).
        func maxPrime(_ n: Int) -> Int {
            guard n > 1 else { return 1 }
            var m = n, d = 2, best = 1
            while d * d <= m { while m % d == 0 { best = max(best, d); m /= d }; d += 1 }
            return m > 1 ? max(best, m) : best
        }
        func expected(_ n: Int, _ range: ClosedRange<Int>) -> Int {
            var bestPML = range.lowerBound, bestScore = Int.max
            for s in range where maxPrime(n + 2 * s) < bestScore {
                bestScore = maxPrime(n + 2 * s); bestPML = s
            }
            return bestPML
        }
        for n in [100, 128, 150, 256] {
            XCTAssertEqual(getOptimalPMLSize([n]), [expected(n, 10...60)])
        }
        // One size per dimension.
        XCTAssertEqual(getOptimalPMLSize([128, 150, 100]),
                       [expected(128, 10...60), expected(150, 10...60), expected(100, 10...60)])
    }

    func testResizeLinearInterpolation() {
        // Upscale a ramp: endpoints preserved, interior linearly interpolated.
        let ramp = MLXArray([0, 1, 2, 3].map { Float($0) })
        let up = resize(ramp, to: [7]).asArray(Float.self)
        XCTAssertEqual(up.count, 7)
        for (a, b) in zip(up, [0.0, 0.5, 1, 1.5, 2, 2.5, 3].map { Float($0) }) {
            XCTAssertEqual(a, b, accuracy: 1e-5)
        }
        // Identity when sizes match.
        let same = resize(ramp, to: [4]).asArray(Float.self)
        for (a, b) in zip(same, [0, 1, 2, 3].map { Float($0) }) { XCTAssertEqual(a, b, accuracy: 1e-6) }
    }

    func testResize2DAnd3DShapeAndCorners() {
        let m = MLXArray((0..<16).map { Float($0) }).reshaped([4, 4])
        let r = resize(m, to: [8, 6])
        XCTAssertEqual(r.shape, [8, 6])
        let h = r.reshaped([48]).asArray(Float.self)
        XCTAssertEqual(h[0], 0, accuracy: 1e-5)         // top-left corner preserved
        XCTAssertEqual(h[47], 15, accuracy: 1e-5)       // bottom-right corner preserved
        let v = MLXArray((0..<8).map { Float($0) }).reshaped([2, 2, 2])
        XCTAssertEqual(resize(v, to: [3, 4, 5]).shape, [3, 4, 5])
    }

    func testToneBurstShapeAndEnvelope() {
        let sampleFreq = 10e6, signalFreq = 1e6, numCycles = 3.0
        let burst = toneBurst(sampleFreq: sampleFreq, signalFreq: signalFreq, numCycles: numCycles)
        let expectedCount = Int(floor((numCycles / signalFreq) / (1 / sampleFreq))) + 1
        XCTAssertEqual(burst.count, expectedCount)
        // Gaussian envelope tapers the ends below the centre amplitude.
        let mid = burst.map(abs).max()!
        XCTAssertLessThan(abs(burst.first!), mid)
        XCTAssertLessThan(abs(burst.last!), mid)
    }
}

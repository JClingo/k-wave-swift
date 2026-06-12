import XCTest
import Metal
import MLX
@testable import KWave

/// GPU smoke tests: run a representative solver subset on the Metal GPU stream and assert the
/// outputs are finite and match the CPU backend. All other tests force the CPU backend (see
/// TestSupport.swift), so without these the GPU code paths have zero coverage — which previously
/// hid a fatal "float64 is not supported on the GPU" crash from `MLXArray([Double])` construction.
final class GPUSmokeTests: XCTestCase {

    /// MLX loads its Metal kernels from `default.metallib` inside the SwiftPM resource bundle
    /// `mlx-swift_Cmlx.bundle`, which only contains the metallib when built by xcodebuild
    /// (Scripts/test.sh). Under `swift test` GPU evaluation fatally errors, so skip instead.
    private static let gpuAvailable: Bool = {
        guard MTLCreateSystemDefaultDevice() != nil else { return false }
        let fm = FileManager.default
        for bundle in Bundle.allBundles {
            for root in [bundle.bundleURL, bundle.resourceURL].compactMap({ $0 }) {
                let lib = root.appendingPathComponent(
                    "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
                if fm.fileExists(atPath: lib.path) { return true }
            }
        }
        return false
    }()

    override func setUpWithError() throws {
        if !Self.gpuAvailable {
            throw XCTSkip("Metal GPU or MLX metallib unavailable; run via Scripts/test.sh")
        }
    }

    /// Evaluate `compute` once on the CPU backend and once on the GPU backend, then assert each
    /// output is finite, nonzero, and that the GPU result matches the CPU result to `rtol`
    /// relative to the CPU max magnitude. `MLX.abs` makes the comparison work for complex outputs.
    private func assertGPUMatchesCPU(
        rtol: Float = 1e-3, labels: [String], _ compute: () -> [MLXArray],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let cpu = Device.withDefaultDevice(.cpu) { compute() }
        let gpu = Device.withDefaultDevice(.gpu) { compute() }
        XCTAssertEqual(cpu.count, labels.count, file: file, line: line)
        XCTAssertEqual(gpu.count, labels.count, file: file, line: line)
        for (i, label) in labels.enumerated() {
            XCTAssertEqual(gpu[i].shape, cpu[i].shape, label, file: file, line: line)
            let cpuMax = MLX.max(MLX.abs(cpu[i])).item(Float.self)
            let gpuMax = MLX.max(MLX.abs(gpu[i])).item(Float.self)
            XCTAssertTrue(cpuMax.isFinite, "\(label): CPU result not finite", file: file, line: line)
            XCTAssertTrue(gpuMax.isFinite, "\(label): GPU result not finite", file: file, line: line)
            XCTAssertGreaterThan(cpuMax, 0, "\(label): CPU result is all zeros", file: file, line: line)
            let diff = MLX.max(MLX.abs(gpu[i] - cpu[i])).item(Float.self)
            XCTAssertLessThanOrEqual(diff, rtol * cpuMax, "\(label): GPU vs CPU mismatch",
                                     file: file, line: line)
        }
    }

    func testKSpaceFirstOrder2DIVP() {
        assertGPUMatchesCPU(labels: ["pFinal", "p"]) {
            let n = 48
            var grid = KWaveGrid(nx: n, dx: 1e-4, ny: n, dy: 1e-4)
            // Long enough for the wavefront to cross the sensor row, so the recorded series has
            // a real signal (not just spectral-tail noise) for the relative comparison.
            grid.makeTime(soundSpeedMax: 1500, cfl: 0.3, tEnd: 1.2e-6)
            let medium = KWaveMedium(soundSpeed: 1500, density: 1000)
            var source = KWaveSource()
            source.p0 = makeDisc(nx: n, ny: n, radius: 4) * 1.0
            var maskHost = [Float](repeating: 0, count: n * n)
            for col in 0..<n { maskHost[8 * n + col] = 1 }
            var sensor = KWaveSensor()
            sensor.mask = MLXArray(maskHost).reshaped([n, n])
            let out = kspaceFirstOrder(grid: grid, medium: medium, source: source, sensor: sensor)
            return [out.pFinal!, out.p!]
        }
    }

    func testKSpaceFirstOrder3DIVP() {
        assertGPUMatchesCPU(labels: ["pFinal"]) {
            let n = 24
            var grid = KWaveGrid(nx: n, dx: 1e-4, ny: n, dy: 1e-4, nz: n, dz: 1e-4)
            grid.makeTime(soundSpeedMax: 1500, cfl: 0.3, tEnd: 3e-7)
            let medium = KWaveMedium(soundSpeed: 1500, density: 1000)
            var source = KWaveSource()
            source.p0 = makeBall(nx: n, ny: n, nz: n, radius: 3) * 1.0
            let out = kspaceFirstOrder(grid: grid, medium: medium, source: source,
                                       sensor: KWaveSensor())
            return [out.pFinal!]
        }
    }

    func testAngularSpectrum() {
        assertGPUMatchesCPU(labels: ["pressureMax"]) {
            let n = 16
            let dt = 5e-8
            let sig = toneBurst(sampleFreq: 1 / dt, signalFreq: 1e6, numCycles: 3)
                .map { Float($0) }
            let plane = makeDisc(nx: n, ny: n, radius: 4).reshaped([n, n, 1])
                * MLXArray(sig).reshaped([1, 1, sig.count])
            let out = angularSpectrum(inputPlane: plane, dx: 1e-4, dt: dt,
                                      zPos: [5e-4, 1e-3], soundSpeed: 1500)
            return [out.pressureMax]
        }
    }

    func testAngularSpectrumCW() {
        assertGPUMatchesCPU(labels: ["pressure"]) {
            let n = 16
            let plane = makeDisc(nx: n, ny: n, radius: 4) * 1.0
            return [angularSpectrumCW(inputPlane: plane, dx: 1e-4, zPos: [5e-4, 1e-3],
                                      f0: 1e6, soundSpeed: 1500)]
        }
    }

    func testAcousticFieldPropagator() {
        assertGPUMatchesCPU(labels: ["pressure", "amp"]) {
            let n = 24
            let amp = makeDisc(nx: n, ny: n, cx: n / 2, cy: 4, radius: 3) * 1.0
            let out = acousticFieldPropagator(ampIn: amp, phaseIn: MLXArray(Float(0)),
                                              dx: 1e-4, f0: 2e6, c0: 1500)
            return [out.pressure, out.amp]
        }
    }

    func testKSpaceFirstOrderASIVP() {
        assertGPUMatchesCPU(labels: ["pFinal"]) {
            let nx = 32, ny = 24
            var grid = KWaveGrid(nx: nx, dx: 1e-4, ny: ny, dy: 1e-4)
            grid.makeTime(soundSpeedMax: 1500, cfl: 0.3, tEnd: 3e-7)
            let medium = KWaveMedium(soundSpeed: 1500, density: 1000)
            var source = KWaveSource()
            // Disc centred on the symmetry axis (y = 0 is the axis in the axisymmetric solver).
            source.p0 = makeDisc(nx: nx, ny: ny, cx: nx / 2, cy: 0, radius: 4) * 1.0
            let out = kspaceFirstOrderAS(grid: grid, medium: medium, source: source,
                                         sensor: KWaveSensor())
            return [out.pFinal!]
        }
    }
}

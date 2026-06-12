import Foundation
import KWave
import MLX

// Performance benchmark for k-wave-swift. Mirrors Scripts/bench/bench_python.py — scenario
// names, grids, step counts, and physics must stay in sync so compare.py can join the results.
//
// Usage: kwave-bench [--repeats N] [--warmup N] [--filter substr] [--out path.json] [--device gpu|cpu]
// Build with xcodebuild (not `swift run`): SwiftPM CLI cannot bundle MLX's metallib.

struct BenchConfig {
    var repeats = 3
    var warmup = 1
    var filter: String? = nil
    var out: String? = nil
    var device: DeviceKind = .gpu
}

func parseArgs() -> BenchConfig {
    var cfg = BenchConfig()
    var args = ArraySlice(CommandLine.arguments.dropFirst())
    while let flag = args.popFirst() {
        switch flag {
        case "--repeats": cfg.repeats = Int(args.popFirst() ?? "") ?? cfg.repeats
        case "--warmup": cfg.warmup = Int(args.popFirst() ?? "") ?? cfg.warmup
        case "--filter": cfg.filter = args.popFirst()
        case "--out": cfg.out = args.popFirst()
        case "--device": cfg.device = (args.popFirst() == "cpu") ? .cpu : .gpu
        default:
            FileHandle.standardError.write("unknown flag: \(flag)\n".data(using: .utf8)!)
            exit(2)
        }
    }
    return cfg
}

struct BenchResult {
    let name: String
    let seconds: [Double]
    let meta: [String: String]

    var best: Double { seconds.min() ?? 0 }
    var median: Double {
        let s = seconds.sorted()
        return s.isEmpty ? 0 : s[s.count / 2]
    }
}

/// Time `body` with warmup. `body` must synchronize (eval) before returning.
func timeIt(repeats: Int, warmup: Int, _ body: () -> Void) -> [Double] {
    for _ in 0..<warmup { body() }
    return (0..<repeats).map { _ in
        let t0 = DispatchTime.now()
        body()
        return Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
    }
}

// MARK: - Scenarios (keep in sync with bench_python.py)

let dxRef = 1e-4
let c0Ref = 1500.0
let rho0Ref = 1000.0

func ivpOptions(pml: Int) -> SimulationOptions {
    var opts = SimulationOptions()
    opts.pmlSize = .uniform(pml)
    opts.pmlInside = true
    opts.smoothP0 = true
    opts.dtype = .float32
    return opts
}

/// Half-plane step map: `base` for x < nx/2, `base * 1.2` beyond — exercises the
/// heterogeneous-medium branch without data-dependent cost.
func stepMap(_ shape: [Int], base: Double) -> MLXArray {
    let nx = shape[0]
    let rest = shape.dropFirst().reduce(1, *)
    var host = [Float](repeating: Float(base), count: nx * rest)
    for i in (nx / 2 * rest)..<host.count { host[i] = Float(base * 1.2) }
    return MLXArray(host).reshaped(shape)
}

func bench2D(n: Int, nt: Int, hetero: Bool = false, absorbing: Bool = false,
             cfg: BenchConfig) -> BenchResult {
    var grid = KWaveGrid(nx: n, dx: dxRef, ny: n, dy: dxRef)
    grid.makeTime(soundSpeedMax: c0Ref, cfl: 0.3)
    grid.setTime(nt: nt, dt: grid.dt)

    let medium: KWaveMedium
    if hetero {
        medium = KWaveMedium(soundSpeed: stepMap([n, n], base: c0Ref),
                             density: stepMap([n, n], base: rho0Ref))
    } else if absorbing {
        medium = KWaveMedium(soundSpeed: c0Ref, density: rho0Ref,
                             alphaCoeff: 0.75, alphaPower: 1.5, alphaMode: .powerLaw)
    } else {
        medium = KWaveMedium(soundSpeed: c0Ref, density: rho0Ref)
    }

    var source = KWaveSource()
    source.p0 = makeDisc(nx: n, ny: n, radius: Double(n) / 16)
    let sensor = KWaveSensor(record: [.pFinal])
    var opts = ivpOptions(pml: 20)
    opts.device = cfg.device

    let seconds = timeIt(repeats: cfg.repeats, warmup: cfg.warmup) {
        let out = kspaceFirstOrder(grid: grid, medium: medium, source: source,
                                   sensor: sensor, options: opts)
        eval(out.pFinal!)
    }
    var name = "kspace2d_\(n)"
    if hetero { name += "_hetero" }
    if absorbing { name += "_absorbing" }
    return BenchResult(name: name, seconds: seconds,
                       meta: ["grid": "\(n)x\(n)", "steps": "\(nt)"])
}

func bench2DSource(n: Int, nt: Int, cfg: BenchConfig) -> BenchResult {
    var grid = KWaveGrid(nx: n, dx: dxRef, ny: n, dy: dxRef)
    grid.makeTime(soundSpeedMax: c0Ref, cfl: 0.3)
    grid.setTime(nt: nt, dt: grid.dt)
    let medium = KWaveMedium(soundSpeed: c0Ref, density: rho0Ref)

    // Line pressure source across row 20, tone burst broadcast to every point.
    var srcMask = [Float](repeating: 0, count: n * n)
    for j in 0..<n { srcMask[20 * n + j] = 1 }
    var source = KWaveSource()
    source.pMask = MLXArray(srcMask).reshaped([n, n])
    source.p = MLXArray(toneBurst(sampleFreq: 1 / grid.dt, signalFreq: 1e6, numCycles: 5)
        .map { Float($0) })
    source.pMode = .additive

    // Line sensor across row n-20 recording the full pressure time series.
    var senMask = [Float](repeating: 0, count: n * n)
    for j in 0..<n { senMask[(n - 20) * n + j] = 1 }
    let sensor = KWaveSensor(mask: MLXArray(senMask).reshaped([n, n]), record: [.p])
    var opts = ivpOptions(pml: 20)
    opts.smoothP0 = false
    opts.device = cfg.device

    let seconds = timeIt(repeats: cfg.repeats, warmup: cfg.warmup) {
        let out = kspaceFirstOrder(grid: grid, medium: medium, source: source,
                                   sensor: sensor, options: opts)
        eval(out.p!)
    }
    return BenchResult(name: "kspace2d_\(n)_source", seconds: seconds,
                       meta: ["grid": "\(n)x\(n)", "steps": "\(nt)"])
}

func bench3D(n: Int, nt: Int, cfg: BenchConfig) -> BenchResult {
    var grid = KWaveGrid(nx: n, dx: dxRef, ny: n, dy: dxRef, nz: n, dz: dxRef)
    grid.makeTime(soundSpeedMax: c0Ref, cfl: 0.3)
    grid.setTime(nt: nt, dt: grid.dt)
    let medium = KWaveMedium(soundSpeed: c0Ref, density: rho0Ref)

    var source = KWaveSource()
    source.p0 = makeBall(nx: n, ny: n, nz: n, radius: Double(n) / 8)
    let sensor = KWaveSensor(record: [.pFinal])
    var opts = ivpOptions(pml: 10)
    opts.device = cfg.device

    let seconds = timeIt(repeats: cfg.repeats, warmup: cfg.warmup) {
        let out = kspaceFirstOrder(grid: grid, medium: medium, source: source,
                                   sensor: sensor, options: opts)
        eval(out.pFinal!)
    }
    return BenchResult(name: "kspace3d_\(n)", seconds: seconds,
                       meta: ["grid": "\(n)x\(n)x\(n)", "steps": "\(nt)"])
}

func benchAngularSpectrum(n: Int, nt: Int, nz: Int, cfg: BenchConfig) -> BenchResult {
    let dt = 2e-8
    // Disc aperture × tone burst time series.
    let disc = makeDisc(nx: n, ny: n, radius: Double(n) / 8)
    var burst = toneBurst(sampleFreq: 1 / dt, signalFreq: 1e6, numCycles: 5).map { Float($0) }
    burst += [Float](repeating: 0, count: max(0, nt - burst.count))
    let signal = MLXArray(Array(burst[0..<nt])).reshaped([1, 1, nt])
    let plane = disc.reshaped([n, n, 1]) * signal
    eval(plane)
    let zPos = (1...nz).map { Double($0) * dxRef }

    let seconds = timeIt(repeats: cfg.repeats, warmup: cfg.warmup) {
        let out = angularSpectrum(inputPlane: plane, dx: dxRef, dt: dt, zPos: zPos,
                                  soundSpeed: c0Ref)
        eval(out.pressureMax)
    }
    return BenchResult(name: "angular_spectrum_\(n)", seconds: seconds,
                       meta: ["grid": "\(n)x\(n)x\(nt)", "steps": "\(nz) planes"])
}

func benchAFP(n: Int, cfg: BenchConfig) -> BenchResult {
    let amp = makeBall(nx: n, ny: n, nz: n, radius: Double(n) / 8)
    eval(amp)
    let seconds = timeIt(repeats: cfg.repeats, warmup: cfg.warmup) {
        let out = acousticFieldPropagator(ampIn: amp, phaseIn: MLXArray(Float(0)),
                                          dx: dxRef, f0: 1e6, c0: c0Ref)
        eval(out.pressure)
    }
    return BenchResult(name: "afp_\(n)", seconds: seconds,
                       meta: ["grid": "\(n)x\(n)x\(n)", "steps": "1"])
}

// MARK: - Main

let cfg = parseArgs()

let scenarios: [(String, (BenchConfig) -> BenchResult)] = [
    ("kspace2d_64", { bench2D(n: 64, nt: 256, cfg: $0) }),
    ("kspace2d_128", { bench2D(n: 128, nt: 256, cfg: $0) }),
    ("kspace2d_256", { bench2D(n: 256, nt: 256, cfg: $0) }),
    ("kspace2d_512", { bench2D(n: 512, nt: 256, cfg: $0) }),
    ("kspace2d_1024", { bench2D(n: 1024, nt: 256, cfg: $0) }),
    ("kspace2d_256_hetero", { bench2D(n: 256, nt: 256, hetero: true, cfg: $0) }),
    ("kspace2d_256_absorbing", { bench2D(n: 256, nt: 256, absorbing: true, cfg: $0) }),
    ("kspace2d_256_source", { bench2DSource(n: 256, nt: 256, cfg: $0) }),
    ("kspace3d_32", { bench3D(n: 32, nt: 128, cfg: $0) }),
    ("kspace3d_64", { bench3D(n: 64, nt: 128, cfg: $0) }),
    ("kspace3d_128", { bench3D(n: 128, nt: 128, cfg: $0) }),
    ("angular_spectrum_128", { benchAngularSpectrum(n: 128, nt: 64, nz: 32, cfg: $0) }),
    ("afp_64", { benchAFP(n: 64, cfg: $0) }),
]

var results: [BenchResult] = []
for (name, run) in scenarios {
    if let f = cfg.filter, !name.contains(f) { continue }
    // Buffers cached by a large scenario otherwise distort the next one's timings.
    Memory.clearCache()
    let r = run(cfg)
    results.append(r)
    let times = r.seconds.map { String(format: "%.4f", $0) }.joined(separator: ", ")
    print("\(r.name): best \(String(format: "%.4f", r.best))s  median " +
          "\(String(format: "%.4f", r.median))s  [\(times)]")
}

if let outPath = cfg.out {
    let payload: [String: Any] = [
        "impl": "swift",
        "device": cfg.device == .gpu ? "gpu" : "cpu",
        "repeats": cfg.repeats,
        "results": results.map {
            ["name": $0.name, "seconds": $0.seconds, "best": $0.best,
             "median": $0.median, "meta": $0.meta] as [String: Any]
        },
    ]
    let data = try JSONSerialization.data(withJSONObject: payload,
                                          options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: outPath))
    print("wrote \(outPath)")
}

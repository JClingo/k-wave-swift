import Foundation
import MLX
import MLXFFT

/// PML thickness specification.
public enum PMLSize: Sendable {
    case uniform(Int)
}

/// Plotting / colour-scale options (placeholder for Phase-2 visualization).
public enum PlotScale: Sendable {
    case auto
    case fixed(Double, Double)
}

/// Simulation configuration (mirrors MATLAB name-value pairs).
public struct SimulationOptions {
    public var pmlInside: Bool = true
    public var pmlSize: PMLSize = .uniform(20)
    public var pmlAlpha: Double = 2.0
    public var smoothP0: Bool = true
    public var smoothC0: Bool = false
    public var smoothRho0: Bool = false
    public var dtype: DTypePrecision = .float32
    public var device: DeviceKind = .gpu
    public var plotSim: Bool = false
    public var plotScale: PlotScale = .auto
    public var recordMovie: String? = nil
    public var saveToDisk: String? = nil
    /// Per-step progress callback: `(tIndex, nt)`. Called once per time step.
    public var progress: ((Int, Int) -> Void)? = nil

    public init() {}
}

/// Recorded simulation results. Which fields are populated is controlled by `sensor.record`
/// (a `RecordField` option set); `pFinal` is always returned. Time series have shape
/// `[numSensorPoints, nt]`; aggregates have shape `[numSensorPoints]`; final/`*Final` fields are
/// the whole-grid snapshot at the last step. Velocity time series are interpolated to the
/// (collocated) pressure grid, matching k-Wave's `ux`/`uy`/`uz` outputs.
public struct SimulationOutput {
    // Pressure.
    public var p: MLXArray?
    public var pMax: MLXArray?
    public var pMin: MLXArray?
    public var pRms: MLXArray?
    public var pFinal: MLXArray?
    // Velocity time series (collocated to the pressure grid).
    public var ux: MLXArray?
    public var uy: MLXArray?
    public var uz: MLXArray?
    // Velocity aggregates (per component).
    public var uxMax: MLXArray?
    public var uyMax: MLXArray?
    public var uzMax: MLXArray?
    public var uxRms: MLXArray?
    public var uyRms: MLXArray?
    public var uzRms: MLXArray?
    // Velocity final fields (whole grid).
    public var uxFinal: MLXArray?
    public var uyFinal: MLXArray?
    public var uzFinal: MLXArray?
    // Time-averaged acoustic intensity (per component).
    public var ixAvg: MLXArray?
    public var iyAvg: MLXArray?
    public var izAvg: MLXArray?
}

/// k-space pseudospectral first-order acoustic solver. Branches on `grid.dim`.
/// Supports the linear, lossless case for homogeneous or spatially varying sound speed and
/// density, with an initial-pressure source and/or time-varying pressure / velocity sources
/// (additive, additive-no-correction, dirichlet modes).
public func kspaceFirstOrder(
    grid: KWaveGrid,
    medium: KWaveMedium,
    source: KWaveSource,
    sensor: KWaveSensor,
    options: SimulationOptions = .init()
) -> SimulationOutput {
    switch grid.dim {
    case 1:
        return kspaceFirstOrder1D(grid: grid, medium: medium, source: source,
                                  sensor: sensor, options: options)
    case 2:
        return kspaceFirstOrder2D(grid: grid, medium: medium, source: source,
                                  sensor: sensor, options: options)
    case 3:
        return kspaceFirstOrder3D(grid: grid, medium: medium, source: source,
                                  sensor: sensor, options: options)
    default:
        fatalError("kspaceFirstOrder: dim \(grid.dim) not yet implemented")
    }
}

// MARK: - Time-varying source injection

/// One field-variable source operator, mirroring k-wave-python `_build_source_op`.
private struct SourceOp {
    let indices: MLXArray        // Int32 flat indices into the grid, length nSrc.
    let signal: [[Float]]        // [rows][signalLen]; rows is 1 (broadcast) or nSrc.
    let scale: [Float]           // length nSrc; per-source-point scale factor.
    let mode: SourceMode
    let nSrc: Int
    let size: Int                // total grid points.
    let shape: [Int]             // grid shape.

    var signalLen: Int { signal.first?.count ?? 0 }

    /// Scaled source values at time index `t`, or `nil` once the signal is exhausted.
    func valuesAt(_ t: Int) -> MLXArray? {
        guard t < signalLen else { return nil }
        let v: [Float]
        if signal.count == 1 {
            let base = signal[0][t]
            v = scale.map { base * $0 }
        } else {
            v = (0..<nSrc).map { signal[$0][t] * scale[$0] }
        }
        return MLXArray(v)
    }

    /// Apply this source to `field` at time `t`. `fwd`/`inv` are the n-D FFT pair used by the
    /// additive k-space correction (`field + ifft(source_kappa * fft(src))`).
    func apply(_ field: MLXArray, t: Int, sourceKappa: MLXArray,
               fwd: (MLXArray) -> MLXArray, inv: (MLXArray) -> MLXArray) -> MLXArray {
        guard let vals = valuesAt(t) else { return field }
        switch mode {
        case .dirichlet:
            let flat = field.reshaped([size])
            flat[indices] = vals
            return flat.reshaped(shape)
        case .additiveNoCorrection:
            let buf = MLXArray.zeros([size], dtype: .float32)
            buf[indices] = vals
            return field + buf.reshaped(shape)
        case .additive:
            let buf = MLXArray.zeros([size], dtype: .float32)
            buf[indices] = vals
            let src = buf.reshaped(shape)
            let corrected = inv(sourceKappa * fwd(src.asType(.complex64))).realPart()
            return field + corrected
        }
    }
}

/// Flatten an MLX source signal into `[rows][signalLen]` host arrays.
/// 1-D input is one broadcast row; 2-D input is one row per source point.
private func signalRows(_ a: MLXArray) -> [[Float]] {
    if a.ndim <= 1 {
        return [a.asArray(Float.self)]
    }
    let rows = a.shape[0]
    let cols = a.shape[1]
    let flat = a.reshaped([rows * cols]).asArray(Float.self)
    return (0..<rows).map { Array(flat[$0 * cols ..< ($0 + 1) * cols]) }
}

/// Build a `SourceOp` for one field variable, or `nil` if the mask or signal is absent. `scale`
/// maps the local sound speed at each source point to its per-point scale factor, so the operator
/// is correct for both homogeneous and spatially varying `c0`.
private func makeSourceOp(mask: MLXArray?, signal: MLXArray?, mode: SourceMode,
                          c0: MLXArray, scale: (Float) -> Float, shape: [Int]) -> SourceOp? {
    guard let mask, let signal else { return nil }
    let idx = flatNonzeroIndices(mask)
    let nSrc = idx.size
    guard nSrc > 0 else { return nil }
    return SourceOp(indices: idx, signal: signalRows(signal),
                    scale: soundSpeedSamples(c0, at: idx, count: nSrc).map(scale),
                    mode: mode, nSrc: nSrc, size: shape.reduce(1, *), shape: shape)
}

private func kspaceFirstOrder1D(
    grid: KWaveGrid,
    medium: KWaveMedium,
    source: KWaveSource,
    sensor: KWaveSensor,
    options: SimulationOptions
) -> SimulationOutput {
    precondition(grid.nt > 0, "call grid.makeTime(...) before running the solver")
    precondition(source.p0 != nil || source.pMask != nil || source.uMask != nil,
                 "1D solver requires source.p0 or a time-varying p/u source")

    let nx = grid.nx
    let shape = [nx]
    let ndim = 1
    let dt = grid.dt

    var c0Grid = medium.soundSpeed.asType(.float32)
    var rho0Grid = medium.density.asType(.float32)
    precondition(c0Grid.size == 1 || c0Grid.shape == shape, "soundSpeed must be scalar or [nx]")
    precondition(rho0Grid.size == 1 || rho0Grid.shape == shape, "density must be scalar or [nx]")
    if options.smoothC0 && c0Grid.size > 1 { c0Grid = smooth1D(c0Grid) }
    if options.smoothRho0 && rho0Grid.size > 1 { rho0Grid = smooth1D(rho0Grid) }
    let cRef = referenceSoundSpeed(c0Grid)

    // k-space operators: kappa = sinc(c_ref*dt*|k|/2); source_kappa = cos(c_ref*dt*|k|/2).
    let arg = grid.k * (cRef * dt / 2.0)
    let kappa = MLX.which(arg .== 0, MLXArray(Float(1)), MLX.sin(arg) / arg).asType(.float32)
    let sourceKappa = MLX.cos(arg).asType(.float32)

    let ddxPos = derivativeOperator(grid.kxVec, spacing: grid.dx, shift: +1)
    let ddxNeg = derivativeOperator(grid.kxVec, spacing: grid.dx, shift: -1)

    let pmlSizeN: Int = { switch options.pmlSize { case let .uniform(s): return s } }()
    let pmlX = MLXArray(converting: pmlProfile(n: nx, dx: grid.dx, dt: dt, c: cRef,
                                               pmlSize: pmlSizeN, pmlAlpha: options.pmlAlpha,
                                               staggered: false)).asType(.float32)
    let pmlXsg = MLXArray(converting: pmlProfile(n: nx, dx: grid.dx, dt: dt, c: cRef,
                                                 pmlSize: pmlSizeN, pmlAlpha: options.pmlAlpha,
                                                 staggered: true)).asType(.float32)

    let c2Grid = c0Grid * c0Grid
    let dtRho0 = Float(dt) * rho0Grid
    let dtOverRhoX = Float(dt) / staggerDensity(rho0Grid, axis: 0)

    let n = Double(ndim)
    func pScale(_ d: Double) -> (Float) -> Float {
        if source.pMode == .dirichlet {
            return { c in Float(1.0 / (n * Double(c) * Double(c))) }
        }
        return { c in Float(2.0 * dt / (n * Double(c) * d)) }
    }
    func uScale(_ d: Double) -> (Float) -> Float {
        if source.uMode == .dirichlet { return { _ in Float(1) } }
        return { c in Float(2.0 * Double(c) * dt / d) }
    }
    let pOpX = makeSourceOp(mask: source.pMask, signal: source.p, mode: source.pMode,
                            c0: c0Grid, scale: pScale(grid.dx), shape: shape)
    let uOpX = makeSourceOp(mask: source.uMask, signal: source.ux, mode: source.uMode,
                            c0: c0Grid, scale: uScale(grid.dx), shape: shape)

    var p0Field: MLXArray? = nil
    if let p0 = source.p0 {
        var f = p0.asType(.float32)
        if options.smoothP0 { f = smooth1D(f) }
        p0Field = f
    }

    let fwd: (MLXArray) -> MLXArray = { MLXFFT.fft($0) }
    let inv: (MLXArray) -> MLXArray = { MLXFFT.ifft($0) }
    let absorb = makeAbsorption(medium: medium, c0: c0Grid, rho0: rho0Grid, k: grid.k,
                                fwd: fwd, inv: inv)
    let nonlinear = makeNonlinearity(medium: medium, rho0: rho0Grid)

    var p = MLXArray.zeros(shape, dtype: .float32)
    var ux = MLXArray.zeros(shape, dtype: .float32)
    var rhox = MLXArray.zeros(shape, dtype: .float32)

    let plan = RecordPlan(sensor.record)
    let sampler: SensorSampler? = sensor.mask.map { makeSensorSampler(mask: $0, grid: grid) }
    let collocX = plan.recordU ? collocationOp(grid.kxVec, spacing: grid.dx) : nil
    var pRec: [MLXArray] = [], uxRec: [MLXArray] = []

    for t in 0..<grid.nt {
        // Velocity update.
        let pk = MLXFFT.fft(p.asType(.complex64))
        let dpdx = MLXFFT.ifft(ddxPos * kappa * pk).realPart()
        ux = pmlXsg * (pmlXsg * ux - dtOverRhoX * dpdx)
        if let op = uOpX { ux = op.apply(ux, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }

        // Density update. nl_factor uses the previous step's density (before this update).
        let nlFactor = nonlinear?.factor(rhox)
        let duxdx = MLXFFT.ifft(ddxNeg * kappa * MLXFFT.fft(ux.asType(.complex64))).realPart()
        let massX = nlFactor.map { dtRho0 * duxdx * $0 } ?? (dtRho0 * duxdx)
        rhox = pmlX * (pmlX * rhox - massX)
        if let op = pOpX { rhox = op.apply(rhox, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }

        // Equation of state: p = c0^2 * (rho + absorption - dispersion + nonlinearity).
        var eosRho = rhox
        if let absorb { eosRho = eosRho + absorb.eosTerm(divU: duxdx, rho: rhox) }
        if let nonlinear { eosRho = eosRho + nonlinear.term(rhox) }
        p = c2Grid * eosRho

        // t=0 leapfrog override for an initial-pressure source.
        if t == 0, let f = p0Field {
            p = f
            rhox = f / c2Grid
            let pk0 = MLXFFT.fft(p.asType(.complex64))
            ux = (dtOverRhoX / 2) * MLXFFT.ifft(ddxPos * kappa * pk0).realPart()
        }

        if let sampler {
            if plan.recordP { pRec.append(sampler.sample(p)) }
            if plan.recordU {
                let uxC = MLXFFT.ifft(collocX! * MLXFFT.fft(ux.asType(.complex64))).realPart()
                let sx = sampler.sample(uxC)
                MLX.eval(sx)
                uxRec.append(sx)
            }
        }
        options.progress?(t, grid.nt)
        MLX.eval(p, ux, rhox)
    }

    return finalizeRecording(
        record: sensor.record,
        p: pRec.isEmpty ? nil : MLX.stacked(pRec, axis: 1),
        ux: uxRec.isEmpty ? nil : MLX.stacked(uxRec, axis: 1), uy: nil, uz: nil,
        pFinal: p, uxFinal: plan.recordUFinal ? ux : nil, uyFinal: nil, uzFinal: nil)
}

private func kspaceFirstOrder2D(
    grid: KWaveGrid,
    medium: KWaveMedium,
    source: KWaveSource,
    sensor: KWaveSensor,
    options: SimulationOptions
) -> SimulationOutput {
    precondition(grid.nt > 0, "call grid.makeTime(...) before running the solver")
    precondition(source.p0 != nil || source.pMask != nil || source.uMask != nil,
                 "2D solver requires source.p0 or a time-varying p/u source")

    let nx = grid.nx, ny = grid.ny
    let shape = [nx, ny]
    let ndim = 2
    let dt = grid.dt

    var c0Grid = medium.soundSpeed.asType(.float32)
    var rho0Grid = medium.density.asType(.float32)
    precondition(c0Grid.size == 1 || c0Grid.shape == shape, "soundSpeed must be scalar or [nx, ny]")
    precondition(rho0Grid.size == 1 || rho0Grid.shape == shape, "density must be scalar or [nx, ny]")
    if options.smoothC0 && c0Grid.size > 1 { c0Grid = smooth(c0Grid) }
    if options.smoothRho0 && rho0Grid.size > 1 { rho0Grid = smooth(rho0Grid) }
    let cRef = referenceSoundSpeed(c0Grid)

    // k-space operators: kappa = sinc(c_ref*dt*|k|/2); source_kappa = cos(c_ref*dt*|k|/2).
    let arg = grid.k * (cRef * dt / 2.0)
    let kappa = MLX.which(arg .== 0, MLXArray(Float(1)), MLX.sin(arg) / arg).asType(.float32)
    let sourceKappa = MLX.cos(arg).asType(.float32)

    let ddxPos = derivativeOperator(grid.kxVec, spacing: grid.dx, shift: +1).reshaped([nx, 1])
    let ddxNeg = derivativeOperator(grid.kxVec, spacing: grid.dx, shift: -1).reshaped([nx, 1])
    let ddyPos = derivativeOperator(grid.kyVec, spacing: grid.dy, shift: +1).reshaped([1, ny])
    let ddyNeg = derivativeOperator(grid.kyVec, spacing: grid.dy, shift: -1).reshaped([1, ny])

    let pmlSizeN: Int = { switch options.pmlSize { case let .uniform(s): return s } }()
    let pmlX = vectorToColumn(pmlProfile(n: nx, dx: grid.dx, dt: dt, c: cRef,
                                         pmlSize: pmlSizeN, pmlAlpha: options.pmlAlpha,
                                         staggered: false), length: nx, axis: 0, other: ny)
    let pmlY = vectorToColumn(pmlProfile(n: ny, dx: grid.dy, dt: dt, c: cRef,
                                         pmlSize: pmlSizeN, pmlAlpha: options.pmlAlpha,
                                         staggered: false), length: ny, axis: 1, other: nx)
    let pmlXsg = vectorToColumn(pmlProfile(n: nx, dx: grid.dx, dt: dt, c: cRef,
                                           pmlSize: pmlSizeN, pmlAlpha: options.pmlAlpha,
                                           staggered: true), length: nx, axis: 0, other: ny)
    let pmlYsg = vectorToColumn(pmlProfile(n: ny, dx: grid.dy, dt: dt, c: cRef,
                                           pmlSize: pmlSizeN, pmlAlpha: options.pmlAlpha,
                                           staggered: true), length: ny, axis: 1, other: nx)

    let c2Grid = c0Grid * c0Grid
    let dtRho0 = Float(dt) * rho0Grid
    let dtOverRhoX = Float(dt) / staggerDensity(rho0Grid, axis: 0)
    let dtOverRhoY = Float(dt) / staggerDensity(rho0Grid, axis: 1)

    // Source operators: per-point scale from the local sound speed.
    let n = Double(ndim)
    func pScale(_ d: Double) -> (Float) -> Float {
        if source.pMode == .dirichlet {
            return { c in Float(1.0 / (n * Double(c) * Double(c))) }
        }
        return { c in Float(2.0 * dt / (n * Double(c) * d)) }
    }
    func uScale(_ d: Double) -> (Float) -> Float {
        if source.uMode == .dirichlet { return { _ in Float(1) } }
        return { c in Float(2.0 * Double(c) * dt / d) }
    }
    let pOpX = makeSourceOp(mask: source.pMask, signal: source.p, mode: source.pMode,
                            c0: c0Grid, scale: pScale(grid.dx), shape: shape)
    let pOpY = makeSourceOp(mask: source.pMask, signal: source.p, mode: source.pMode,
                            c0: c0Grid, scale: pScale(grid.dy), shape: shape)
    let uOpX = makeSourceOp(mask: source.uMask, signal: source.ux, mode: source.uMode,
                            c0: c0Grid, scale: uScale(grid.dx), shape: shape)
    let uOpY = makeSourceOp(mask: source.uMask, signal: source.uy, mode: source.uMode,
                            c0: c0Grid, scale: uScale(grid.dy), shape: shape)

    var p0Field: MLXArray? = nil
    if let p0 = source.p0 {
        var f = p0.asType(.float32)
        if options.smoothP0 { f = smooth(f) }
        p0Field = f
    }

    let fwd: (MLXArray) -> MLXArray = { MLXFFT.fft2($0) }
    let inv: (MLXArray) -> MLXArray = { MLXFFT.ifft2($0) }
    let absorb = makeAbsorption(medium: medium, c0: c0Grid, rho0: rho0Grid, k: grid.k,
                                fwd: fwd, inv: inv)
    let nonlinear = makeNonlinearity(medium: medium, rho0: rho0Grid)

    var p = MLXArray.zeros(shape, dtype: .float32)
    var ux = MLXArray.zeros(shape, dtype: .float32)
    var uy = MLXArray.zeros(shape, dtype: .float32)
    var rhox = MLXArray.zeros(shape, dtype: .float32)
    var rhoy = MLXArray.zeros(shape, dtype: .float32)

    let plan = RecordPlan(sensor.record)
    let sampler: SensorSampler? = sensor.mask.map { makeSensorSampler(mask: $0, grid: grid) }
    let directivity = makeDirectivityFilter(sensor: sensor, grid: grid)
    let collocX = plan.recordU ? collocationOp(grid.kxVec, spacing: grid.dx).reshaped([nx, 1]) : nil
    let collocY = plan.recordU ? collocationOp(grid.kyVec, spacing: grid.dy).reshaped([1, ny]) : nil
    var pRec: [MLXArray] = [], uxRec: [MLXArray] = [], uyRec: [MLXArray] = []

    for t in 0..<grid.nt {
        // Velocity update.
        let pk = MLXFFT.fft2(p.asType(.complex64))
        let dpdx = MLXFFT.ifft2(ddxPos * kappa * pk).realPart()
        let dpdy = MLXFFT.ifft2(ddyPos * kappa * pk).realPart()
        ux = pmlXsg * (pmlXsg * ux - dtOverRhoX * dpdx)
        if let op = uOpX { ux = op.apply(ux, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }
        uy = pmlYsg * (pmlYsg * uy - dtOverRhoY * dpdy)
        if let op = uOpY { uy = op.apply(uy, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }

        // Density update. nl_factor uses the previous step's density (before this update).
        let nlFactor = nonlinear?.factor(rhox + rhoy)
        let duxdx = MLXFFT.ifft2(ddxNeg * kappa * MLXFFT.fft2(ux.asType(.complex64))).realPart()
        let duydy = MLXFFT.ifft2(ddyNeg * kappa * MLXFFT.fft2(uy.asType(.complex64))).realPart()
        let massX = nlFactor.map { dtRho0 * duxdx * $0 } ?? (dtRho0 * duxdx)
        let massY = nlFactor.map { dtRho0 * duydy * $0 } ?? (dtRho0 * duydy)
        rhox = pmlX * (pmlX * rhox - massX)
        if let op = pOpX { rhox = op.apply(rhox, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }
        rhoy = pmlY * (pmlY * rhoy - massY)
        if let op = pOpY { rhoy = op.apply(rhoy, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }

        // Equation of state: p = c0^2 * (rho + absorption - dispersion + nonlinearity).
        let rhoTotal = rhox + rhoy
        var eosRho = rhoTotal
        if let absorb { eosRho = eosRho + absorb.eosTerm(divU: duxdx + duydy, rho: rhoTotal) }
        if let nonlinear { eosRho = eosRho + nonlinear.term(rhoTotal) }
        p = c2Grid * eosRho

        // t=0 leapfrog override for an initial-pressure source.
        if t == 0, let f = p0Field {
            p = f
            rhox = f / (Float(ndim) * c2Grid)
            rhoy = f / (Float(ndim) * c2Grid)
            let pk0 = MLXFFT.fft2(p.asType(.complex64))
            ux = (dtOverRhoX / 2) * MLXFFT.ifft2(ddxPos * kappa * pk0).realPart()
            uy = (dtOverRhoY / 2) * MLXFFT.ifft2(ddyPos * kappa * pk0).realPart()
        }

        if let sampler {
            if plan.recordP {
                if let directivity {
                    let s = directivity.apply(pk: MLXFFT.fft2(p.asType(.complex64)))
                    MLX.eval(s)
                    pRec.append(s)
                } else {
                    pRec.append(sampler.sample(p))
                }
            }
            if plan.recordU {
                let uxC = MLXFFT.ifft2(collocX! * MLXFFT.fft2(ux.asType(.complex64))).realPart()
                let uyC = MLXFFT.ifft2(collocY! * MLXFFT.fft2(uy.asType(.complex64))).realPart()
                let sx = sampler.sample(uxC), sy = sampler.sample(uyC)
                MLX.eval(sx, sy)
                uxRec.append(sx); uyRec.append(sy)
            }
        }
        options.progress?(t, grid.nt)
        MLX.eval(p, ux, uy, rhox, rhoy)
    }

    return finalizeRecording(
        record: sensor.record,
        p: pRec.isEmpty ? nil : MLX.stacked(pRec, axis: 1),
        ux: uxRec.isEmpty ? nil : MLX.stacked(uxRec, axis: 1),
        uy: uyRec.isEmpty ? nil : MLX.stacked(uyRec, axis: 1), uz: nil,
        pFinal: p, uxFinal: plan.recordUFinal ? ux : nil,
        uyFinal: plan.recordUFinal ? uy : nil, uzFinal: nil)
}

private func kspaceFirstOrder3D(
    grid: KWaveGrid,
    medium: KWaveMedium,
    source: KWaveSource,
    sensor: KWaveSensor,
    options: SimulationOptions
) -> SimulationOutput {
    precondition(grid.nt > 0, "call grid.makeTime(...) before running the solver")
    precondition(source.p0 != nil || source.pMask != nil || source.uMask != nil,
                 "3D solver requires source.p0 or a time-varying p/u source")

    let nx = grid.nx, ny = grid.ny, nz = grid.nz
    let shape = [nx, ny, nz]
    let ndim = 3
    let dt = grid.dt

    var c0Grid = medium.soundSpeed.asType(.float32)
    var rho0Grid = medium.density.asType(.float32)
    precondition(c0Grid.size == 1 || c0Grid.shape == shape, "soundSpeed must be scalar or [nx, ny, nz]")
    precondition(rho0Grid.size == 1 || rho0Grid.shape == shape, "density must be scalar or [nx, ny, nz]")
    if options.smoothC0 && c0Grid.size > 1 { c0Grid = smooth3D(c0Grid) }
    if options.smoothRho0 && rho0Grid.size > 1 { rho0Grid = smooth3D(rho0Grid) }
    let cRef = referenceSoundSpeed(c0Grid)

    let arg = grid.k * (cRef * dt / 2.0)
    let kappa = MLX.which(arg .== 0, MLXArray(Float(1)), MLX.sin(arg) / arg).asType(.float32)
    let sourceKappa = MLX.cos(arg).asType(.float32)

    let ddxPos = derivativeOperator(grid.kxVec, spacing: grid.dx, shift: +1).reshaped([nx, 1, 1])
    let ddxNeg = derivativeOperator(grid.kxVec, spacing: grid.dx, shift: -1).reshaped([nx, 1, 1])
    let ddyPos = derivativeOperator(grid.kyVec, spacing: grid.dy, shift: +1).reshaped([1, ny, 1])
    let ddyNeg = derivativeOperator(grid.kyVec, spacing: grid.dy, shift: -1).reshaped([1, ny, 1])
    let ddzPos = derivativeOperator(grid.kzVec, spacing: grid.dz, shift: +1).reshaped([1, 1, nz])
    let ddzNeg = derivativeOperator(grid.kzVec, spacing: grid.dz, shift: -1).reshaped([1, 1, nz])

    let pmlSizeN: Int = { switch options.pmlSize { case let .uniform(s): return s } }()
    func pml(_ n: Int, _ d: Double, _ axis: Int, staggered: Bool) -> MLXArray {
        let v = MLXArray(converting: pmlProfile(n: n, dx: d, dt: dt, c: cRef,
                                                pmlSize: pmlSizeN, pmlAlpha: options.pmlAlpha,
                                                staggered: staggered)).asType(.float32)
        switch axis {
        case 0: return v.reshaped([n, 1, 1])
        case 1: return v.reshaped([1, n, 1])
        default: return v.reshaped([1, 1, n])
        }
    }
    let pmlX = pml(nx, grid.dx, 0, staggered: false)
    let pmlY = pml(ny, grid.dy, 1, staggered: false)
    let pmlZ = pml(nz, grid.dz, 2, staggered: false)
    let pmlXsg = pml(nx, grid.dx, 0, staggered: true)
    let pmlYsg = pml(ny, grid.dy, 1, staggered: true)
    let pmlZsg = pml(nz, grid.dz, 2, staggered: true)

    let c2Grid = c0Grid * c0Grid
    let dtRho0 = Float(dt) * rho0Grid
    let dtOverRhoX = Float(dt) / staggerDensity(rho0Grid, axis: 0)
    let dtOverRhoY = Float(dt) / staggerDensity(rho0Grid, axis: 1)
    let dtOverRhoZ = Float(dt) / staggerDensity(rho0Grid, axis: 2)

    let n = Double(ndim)
    func pScale(_ d: Double) -> (Float) -> Float {
        if source.pMode == .dirichlet {
            return { c in Float(1.0 / (n * Double(c) * Double(c))) }
        }
        return { c in Float(2.0 * dt / (n * Double(c) * d)) }
    }
    func uScale(_ d: Double) -> (Float) -> Float {
        if source.uMode == .dirichlet { return { _ in Float(1) } }
        return { c in Float(2.0 * Double(c) * dt / d) }
    }
    let pOpX = makeSourceOp(mask: source.pMask, signal: source.p, mode: source.pMode,
                            c0: c0Grid, scale: pScale(grid.dx), shape: shape)
    let pOpY = makeSourceOp(mask: source.pMask, signal: source.p, mode: source.pMode,
                            c0: c0Grid, scale: pScale(grid.dy), shape: shape)
    let pOpZ = makeSourceOp(mask: source.pMask, signal: source.p, mode: source.pMode,
                            c0: c0Grid, scale: pScale(grid.dz), shape: shape)
    let uOpX = makeSourceOp(mask: source.uMask, signal: source.ux, mode: source.uMode,
                            c0: c0Grid, scale: uScale(grid.dx), shape: shape)
    let uOpY = makeSourceOp(mask: source.uMask, signal: source.uy, mode: source.uMode,
                            c0: c0Grid, scale: uScale(grid.dy), shape: shape)
    let uOpZ = makeSourceOp(mask: source.uMask, signal: source.uz, mode: source.uMode,
                            c0: c0Grid, scale: uScale(grid.dz), shape: shape)

    var p0Field: MLXArray? = nil
    if let p0 = source.p0 {
        var f = p0.asType(.float32)
        if options.smoothP0 { f = smooth3D(f) }
        p0Field = f
    }

    let fwd: (MLXArray) -> MLXArray = { MLXFFT.fftn($0) }
    let inv: (MLXArray) -> MLXArray = { MLXFFT.ifftn($0) }
    let absorb = makeAbsorption(medium: medium, c0: c0Grid, rho0: rho0Grid, k: grid.k,
                                fwd: fwd, inv: inv)
    let nonlinear = makeNonlinearity(medium: medium, rho0: rho0Grid)

    var p = MLXArray.zeros(shape, dtype: .float32)
    var ux = MLXArray.zeros(shape, dtype: .float32)
    var uy = MLXArray.zeros(shape, dtype: .float32)
    var uz = MLXArray.zeros(shape, dtype: .float32)
    var rhox = MLXArray.zeros(shape, dtype: .float32)
    var rhoy = MLXArray.zeros(shape, dtype: .float32)
    var rhoz = MLXArray.zeros(shape, dtype: .float32)

    let plan = RecordPlan(sensor.record)
    let sampler: SensorSampler? = sensor.mask.map { makeSensorSampler(mask: $0, grid: grid) }
    let collocX = plan.recordU ? collocationOp(grid.kxVec, spacing: grid.dx).reshaped([nx, 1, 1]) : nil
    let collocY = plan.recordU ? collocationOp(grid.kyVec, spacing: grid.dy).reshaped([1, ny, 1]) : nil
    let collocZ = plan.recordU ? collocationOp(grid.kzVec, spacing: grid.dz).reshaped([1, 1, nz]) : nil
    var pRec: [MLXArray] = [], uxRec: [MLXArray] = [], uyRec: [MLXArray] = [], uzRec: [MLXArray] = []

    for t in 0..<grid.nt {
        // Velocity update.
        let pk = MLXFFT.fftn(p.asType(.complex64))
        let dpdx = MLXFFT.ifftn(ddxPos * kappa * pk).realPart()
        let dpdy = MLXFFT.ifftn(ddyPos * kappa * pk).realPart()
        let dpdz = MLXFFT.ifftn(ddzPos * kappa * pk).realPart()
        ux = pmlXsg * (pmlXsg * ux - dtOverRhoX * dpdx)
        if let op = uOpX { ux = op.apply(ux, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }
        uy = pmlYsg * (pmlYsg * uy - dtOverRhoY * dpdy)
        if let op = uOpY { uy = op.apply(uy, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }
        uz = pmlZsg * (pmlZsg * uz - dtOverRhoZ * dpdz)
        if let op = uOpZ { uz = op.apply(uz, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }

        // Density update. nl_factor uses the previous step's density (before this update).
        let nlFactor = nonlinear?.factor(rhox + rhoy + rhoz)
        let duxdx = MLXFFT.ifftn(ddxNeg * kappa * MLXFFT.fftn(ux.asType(.complex64))).realPart()
        let duydy = MLXFFT.ifftn(ddyNeg * kappa * MLXFFT.fftn(uy.asType(.complex64))).realPart()
        let duzdz = MLXFFT.ifftn(ddzNeg * kappa * MLXFFT.fftn(uz.asType(.complex64))).realPart()
        let massX = nlFactor.map { dtRho0 * duxdx * $0 } ?? (dtRho0 * duxdx)
        let massY = nlFactor.map { dtRho0 * duydy * $0 } ?? (dtRho0 * duydy)
        let massZ = nlFactor.map { dtRho0 * duzdz * $0 } ?? (dtRho0 * duzdz)
        rhox = pmlX * (pmlX * rhox - massX)
        if let op = pOpX { rhox = op.apply(rhox, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }
        rhoy = pmlY * (pmlY * rhoy - massY)
        if let op = pOpY { rhoy = op.apply(rhoy, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }
        rhoz = pmlZ * (pmlZ * rhoz - massZ)
        if let op = pOpZ { rhoz = op.apply(rhoz, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }

        // Equation of state: p = c0^2 * (rho + absorption - dispersion + nonlinearity).
        let rhoTotal = rhox + rhoy + rhoz
        var eosRho = rhoTotal
        if let absorb { eosRho = eosRho + absorb.eosTerm(divU: duxdx + duydy + duzdz, rho: rhoTotal) }
        if let nonlinear { eosRho = eosRho + nonlinear.term(rhoTotal) }
        p = c2Grid * eosRho

        if t == 0, let f = p0Field {
            p = f
            rhox = f / (Float(ndim) * c2Grid)
            rhoy = f / (Float(ndim) * c2Grid)
            rhoz = f / (Float(ndim) * c2Grid)
            let pk0 = MLXFFT.fftn(p.asType(.complex64))
            ux = (dtOverRhoX / 2) * MLXFFT.ifftn(ddxPos * kappa * pk0).realPart()
            uy = (dtOverRhoY / 2) * MLXFFT.ifftn(ddyPos * kappa * pk0).realPart()
            uz = (dtOverRhoZ / 2) * MLXFFT.ifftn(ddzPos * kappa * pk0).realPart()
        }

        if let sampler {
            if plan.recordP { pRec.append(sampler.sample(p)) }
            if plan.recordU {
                let uxC = MLXFFT.ifftn(collocX! * MLXFFT.fftn(ux.asType(.complex64))).realPart()
                let uyC = MLXFFT.ifftn(collocY! * MLXFFT.fftn(uy.asType(.complex64))).realPart()
                let uzC = MLXFFT.ifftn(collocZ! * MLXFFT.fftn(uz.asType(.complex64))).realPart()
                let sx = sampler.sample(uxC), sy = sampler.sample(uyC), sz = sampler.sample(uzC)
                MLX.eval(sx, sy, sz)
                uxRec.append(sx); uyRec.append(sy); uzRec.append(sz)
            }
        }
        options.progress?(t, grid.nt)
        MLX.eval(p, ux, uy, uz, rhox, rhoy, rhoz)
    }

    return finalizeRecording(
        record: sensor.record,
        p: pRec.isEmpty ? nil : MLX.stacked(pRec, axis: 1),
        ux: uxRec.isEmpty ? nil : MLX.stacked(uxRec, axis: 1),
        uy: uyRec.isEmpty ? nil : MLX.stacked(uyRec, axis: 1),
        uz: uzRec.isEmpty ? nil : MLX.stacked(uzRec, axis: 1),
        pFinal: p, uxFinal: plan.recordUFinal ? ux : nil,
        uyFinal: plan.recordUFinal ? uy : nil, uzFinal: plan.recordUFinal ? uz : nil)
}

/// i*k * exp(±i*k*Δ/2) staggered derivative multiplier (FFT order), as a complex64 vector.
func derivativeOperator(_ kVec: MLXArray, spacing: Double, shift: Int) -> MLXArray {
    let theta = kVec * (spacing / 2.0)
    let s = Float(shift)
    let phase = MLX.cos(theta) + (s * imagUnit()) * MLX.sin(theta)
    return imagUnit() * kVec * phase
}

/// Reshape a 1-D PML profile to broadcast along `axis` of a 2D grid.
private func vectorToColumn(_ values: [Double], length: Int, axis: Int, other: Int) -> MLXArray {
    let v = MLXArray(converting: values).asType(.float32)
    return axis == 0 ? v.reshaped([length, 1]) : v.reshaped([1, length])
}

/// Flattened indices of nonzero mask entries, as an Int32 MLXArray for gathering.
func flatNonzeroIndices(_ mask: MLXArray) -> MLXArray {
    let host = mask.reshaped([mask.size]).asArray(Float.self)
    let idx = host.enumerated().compactMap { $0.element != 0 ? Int32($0.offset) : nil }
    return MLXArray(idx)
}

/// Reference sound speed for the k-space correction (`kappa`) and the PML profile: `max(c0)`
/// (the scalar value itself when the medium is homogeneous). Matches k-Wave's default `c_ref`.
func referenceSoundSpeed(_ c0: MLXArray) -> Double {
    c0.size == 1 ? Double(c0.item(Float.self)) : Double(MLX.max(c0).item(Float.self))
}

/// Per-source-point sound-speed samples (the scalar broadcast to every point when `c0` is scalar).
func soundSpeedSamples(_ c0: MLXArray, at idx: MLXArray, count: Int) -> [Float] {
    if c0.size == 1 { return [Float](repeating: c0.item(Float.self), count: count) }
    return c0.reshaped([c0.size])[idx].asArray(Float.self)
}

/// Density interpolated to the staggered (+d/2) grid along `axis` by averaging neighbours and
/// replicating the trailing edge — k-Wave's linear-interp-to-half-point with edge fill. Returns a
/// scalar density unchanged (homogeneous media need no staggering).
func staggerDensity(_ rho0: MLXArray, axis: Int) -> MLXArray {
    guard rho0.size > 1 else { return rho0 }
    let m = MLX.swappedAxes(rho0, axis, 0)
    let n = m.dim(0)
    let avg = 0.5 * (m[0..<(n - 1)] + m[1..<n])
    let staggered = MLX.concatenated([avg, m[(n - 1)..<n]], axis: 0)
    return MLX.swappedAxes(staggered, axis, 0)
}

// MARK: - Power-law absorption / dispersion

/// `|k|^power` fractional-Laplacian multiplier in k-space, with the DC bin forced to 0 (so a
/// negative `power` doesn't produce `inf` there). Matches k-wave-python `_fractional_laplacian`.
private func fractionalLaplacian(_ k: MLXArray, power: Double) -> MLXArray {
    let kf = k.asType(.float32)
    let raised = MLX.pow(kf, Float(power))
    return MLX.which(kf .== 0, MLXArray(Float(0)), raised).asType(.float32)
}

/// Power-law / Stokes absorption and dispersion operators for the equation of state, mirroring
/// k-wave-python `_init_absorption` / `_init_dispersion`. The EOS gains
/// `+ absorption(div_u) - dispersion(rho)`; both terms are zero in the lossless case (`nil`).
private struct PowerLawAbsorption {
    let tau: MLXArray            // absorption coefficient on the grid.
    let nabla1: MLXArray?        // |k|^(y-2); nil for Stokes (direct multiply, no FFT round-trip).
    let eta: MLXArray?           // dispersion coefficient; nil when dispersion disabled.
    let nabla2: MLXArray?        // |k|^(y-1) dispersion fractional Laplacian.
    let rho0: MLXArray
    let fwd: (MLXArray) -> MLXArray
    let inv: (MLXArray) -> MLXArray

    /// Spectral apply of a real fractional-Laplacian operator: `Re(ifft(op * fft(f)))`.
    private func diff(_ f: MLXArray, _ op: MLXArray) -> MLXArray {
        inv(op * fwd(f.asType(.complex64))).realPart()
    }

    /// `absorption(div_u) - dispersion(rho)`, the additive EOS contribution.
    func eosTerm(divU: MLXArray, rho: MLXArray) -> MLXArray {
        let absorption: MLXArray
        if let nabla1 {
            absorption = tau * diff(rho0 * divU, nabla1)   // full power-law
        } else {
            absorption = tau * rho0 * divU                 // Stokes
        }
        guard let eta, let nabla2 else { return absorption }
        return absorption - eta * diff(rho, nabla2)
    }
}

/// Build the absorption/dispersion operators for the medium, or `nil` when lossless
/// (`alphaCoeff` absent or `alphaMode == .noAbsorption`). `c0`/`rho0` are the post-smoothing grids;
/// `k` is the wavenumber magnitude; `fwd`/`inv` are the n-D FFT pair for this dimensionality.
private func makeAbsorption(
    medium: KWaveMedium, c0: MLXArray, rho0: MLXArray, k: MLXArray,
    fwd: @escaping (MLXArray) -> MLXArray, inv: @escaping (MLXArray) -> MLXArray
) -> PowerLawAbsorption? {
    guard let alphaCoeff = medium.alphaCoeff, medium.alphaMode != .noAbsorption else { return nil }

    var y = medium.alphaPower ?? 1.5
    if medium.alphaMode == .stokes { y = 2.0 }
    // db2neper is linear in alpha, so scale the dB/(MHz^y cm) coefficient to Np/((rad/s)^y m).
    let alphaNp = alphaCoeff.asType(.float32) * Float(db2neper(1.0, y: y))
    let isStokes = medium.alphaMode == .stokes || abs(y - 2.0) < 1e-10

    let tau: MLXArray
    let nabla1: MLXArray?
    if isStokes {
        tau = -2 * alphaNp * c0
        nabla1 = nil
    } else {
        tau = -2 * alphaNp * MLX.pow(c0, Float(y - 1))
        nabla1 = fractionalLaplacian(k, power: y - 2)
    }

    var eta: MLXArray? = nil
    var nabla2: MLXArray? = nil
    if !isStokes && medium.alphaMode != .noDispersion {
        eta = 2 * alphaNp * MLX.pow(c0, Float(y)) * Float(tan(Double.pi * y / 2))
        nabla2 = fractionalLaplacian(k, power: y - 1)
    }

    return PowerLawAbsorption(tau: tau, nabla1: nabla1, eta: eta, nabla2: nabla2,
                              rho0: rho0, fwd: fwd, inv: inv)
}

// MARK: - Nonlinearity (B/A)

/// Quadratic (B/A) nonlinearity for the equation of state, mirroring k-wave-python
/// `_init_nonlinearity`. It adds `B/A · ρ²/(2ρ0)` to the EOS and scales the mass-conservation
/// source term by `nl_factor = (2·Σρ + ρ0)/ρ0` evaluated on the *previous* step's split densities.
private struct Nonlinearity {
    let bOnA: MLXArray
    let rho0: MLXArray

    /// `nl_factor`, from the previous step's total density `rhoTotalPrev` (sum of the split ρ's).
    func factor(_ rhoTotalPrev: MLXArray) -> MLXArray { (2 * rhoTotalPrev + rho0) / rho0 }

    /// Additive EOS term `B/A · ρ²/(2ρ0)`.
    func term(_ rhoTotal: MLXArray) -> MLXArray { bOnA * (rhoTotal * rhoTotal) / (2 * rho0) }
}

/// Build the B/A nonlinearity operator, or `nil` when `bOnA` is absent or all-zero (matching
/// k-wave-python's `_is_enabled`). The lossless path is unchanged in that case.
private func makeNonlinearity(medium: KWaveMedium, rho0: MLXArray) -> Nonlinearity? {
    guard let bOnA = medium.bOnA else { return nil }
    let bf = bOnA.asType(.float32)
    guard MLX.max(MLX.abs(bf)).item(Float.self) != 0 else { return nil }
    return Nonlinearity(bOnA: bf, rho0: rho0)
}

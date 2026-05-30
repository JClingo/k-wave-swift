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
    public var dtype: DTypePrecision = .float32
    public var device: DeviceKind = .gpu
    public var plotSim: Bool = false
    public var plotScale: PlotScale = .auto
    public var recordMovie: String? = nil
    public var saveToDisk: String? = nil

    public init() {}
}

/// Recorded simulation results.
public struct SimulationOutput {
    /// Pressure time series at sensor points, shape `[numSensorPoints, nt]`.
    public var p: MLXArray?
    /// Final pressure field over the whole grid.
    public var pFinal: MLXArray?
}

/// k-space pseudospectral first-order acoustic solver. Branches on `grid.dim`.
/// Supports the linear, lossless, homogeneous case with an initial-pressure source
/// and/or time-varying pressure / velocity sources (additive, additive-no-correction,
/// dirichlet modes).
public func kspaceFirstOrder(
    grid: KWaveGrid,
    medium: KWaveMedium,
    source: KWaveSource,
    sensor: KWaveSensor,
    options: SimulationOptions = .init()
) -> SimulationOutput {
    switch grid.dim {
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

/// Build a `SourceOp` for one field variable, or `nil` if the mask or signal is absent.
private func makeSourceOp(mask: MLXArray?, signal: MLXArray?, mode: SourceMode,
                          scalePerPoint: Float, shape: [Int]) -> SourceOp? {
    guard let mask, let signal else { return nil }
    let idx = flatNonzeroIndices(mask)
    let nSrc = idx.size
    guard nSrc > 0 else { return nil }
    return SourceOp(indices: idx, signal: signalRows(signal),
                    scale: [Float](repeating: scalePerPoint, count: nSrc),
                    mode: mode, nSrc: nSrc, size: shape.reduce(1, *), shape: shape)
}

private func kspaceFirstOrder2D(
    grid: KWaveGrid,
    medium: KWaveMedium,
    source: KWaveSource,
    sensor: KWaveSensor,
    options: SimulationOptions
) -> SimulationOutput {
    precondition(medium.isHomogeneous, "2D solver requires a homogeneous medium")
    precondition(grid.nt > 0, "call grid.makeTime(...) before running the solver")
    precondition(source.p0 != nil || source.pMask != nil || source.uMask != nil,
                 "2D solver requires source.p0 or a time-varying p/u source")

    let nx = grid.nx, ny = grid.ny
    let shape = [nx, ny]
    let dt = grid.dt
    let c0 = medium.soundSpeed.item(Float.self)
    let rho0 = medium.density.item(Float.self)

    // k-space operators: kappa = sinc(c0*dt*|k|/2); source_kappa = cos(c0*dt*|k|/2).
    let arg = grid.k * (Double(c0) * dt / 2.0)
    let kappa = MLX.which(arg .== 0, MLXArray(Float(1)), MLX.sin(arg) / arg).asType(.float32)
    let sourceKappa = MLX.cos(arg).asType(.float32)

    let ddxPos = derivativeOperator(grid.kxVec, spacing: grid.dx, shift: +1).reshaped([nx, 1])
    let ddxNeg = derivativeOperator(grid.kxVec, spacing: grid.dx, shift: -1).reshaped([nx, 1])
    let ddyPos = derivativeOperator(grid.kyVec, spacing: grid.dy, shift: +1).reshaped([1, ny])
    let ddyNeg = derivativeOperator(grid.kyVec, spacing: grid.dy, shift: -1).reshaped([1, ny])

    let pmlSizeN: Int = { switch options.pmlSize { case let .uniform(s): return s } }()
    let pmlX = vectorToColumn(pmlProfile(n: nx, dx: grid.dx, dt: dt, c: Double(c0),
                                         pmlSize: pmlSizeN, pmlAlpha: options.pmlAlpha,
                                         staggered: false), length: nx, axis: 0, other: ny)
    let pmlY = vectorToColumn(pmlProfile(n: ny, dx: grid.dy, dt: dt, c: Double(c0),
                                         pmlSize: pmlSizeN, pmlAlpha: options.pmlAlpha,
                                         staggered: false), length: ny, axis: 1, other: nx)
    let pmlXsg = vectorToColumn(pmlProfile(n: nx, dx: grid.dx, dt: dt, c: Double(c0),
                                           pmlSize: pmlSizeN, pmlAlpha: options.pmlAlpha,
                                           staggered: true), length: nx, axis: 0, other: ny)
    let pmlYsg = vectorToColumn(pmlProfile(n: ny, dx: grid.dy, dt: dt, c: Double(c0),
                                           pmlSize: pmlSizeN, pmlAlpha: options.pmlAlpha,
                                           staggered: true), length: ny, axis: 1, other: nx)

    let c2 = Float(c0 * c0)
    let dtOverRho = Float(dt) / rho0
    let dtRho = Float(dt) * rho0
    let ndim = 2

    // Source operators (homogeneous c0 ⇒ scalar per-point scale).
    let pScaleDirichlet = Float(1.0 / (Double(ndim) * Double(c0) * Double(c0)))
    let pScaleAddX = Float(2.0 * dt / (Double(ndim) * Double(c0) * grid.dx))
    let pScaleAddY = Float(2.0 * dt / (Double(ndim) * Double(c0) * grid.dy))
    let pOpX = makeSourceOp(mask: source.pMask, signal: source.p, mode: source.pMode,
                            scalePerPoint: source.pMode == .dirichlet ? pScaleDirichlet : pScaleAddX,
                            shape: shape)
    let pOpY = makeSourceOp(mask: source.pMask, signal: source.p, mode: source.pMode,
                            scalePerPoint: source.pMode == .dirichlet ? pScaleDirichlet : pScaleAddY,
                            shape: shape)
    let uScaleX = source.uMode == .dirichlet ? Float(1) : Float(2.0 * Double(c0) * dt / grid.dx)
    let uScaleY = source.uMode == .dirichlet ? Float(1) : Float(2.0 * Double(c0) * dt / grid.dy)
    let uOpX = makeSourceOp(mask: source.uMask, signal: source.ux, mode: source.uMode,
                            scalePerPoint: uScaleX, shape: shape)
    let uOpY = makeSourceOp(mask: source.uMask, signal: source.uy, mode: source.uMode,
                            scalePerPoint: uScaleY, shape: shape)

    var p0Field: MLXArray? = nil
    if let p0 = source.p0 {
        var f = p0.asType(.float32)
        if options.smoothP0 { f = smooth(f) }
        p0Field = f
    }

    let fwd: (MLXArray) -> MLXArray = { MLXFFT.fft2($0) }
    let inv: (MLXArray) -> MLXArray = { MLXFFT.ifft2($0) }

    var p = MLXArray.zeros(shape, dtype: .float32)
    var ux = MLXArray.zeros(shape, dtype: .float32)
    var uy = MLXArray.zeros(shape, dtype: .float32)
    var rhox = MLXArray.zeros(shape, dtype: .float32)
    var rhoy = MLXArray.zeros(shape, dtype: .float32)

    let sensorIndices: MLXArray? = sensor.mask.map { flatNonzeroIndices($0) }
    var recorded: [MLXArray] = []
    recorded.reserveCapacity(grid.nt)

    for t in 0..<grid.nt {
        // Velocity update.
        let pk = MLXFFT.fft2(p.asType(.complex64))
        let dpdx = MLXFFT.ifft2(ddxPos * kappa * pk).realPart()
        let dpdy = MLXFFT.ifft2(ddyPos * kappa * pk).realPart()
        ux = pmlXsg * (pmlXsg * ux - dtOverRho * dpdx)
        if let op = uOpX { ux = op.apply(ux, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }
        uy = pmlYsg * (pmlYsg * uy - dtOverRho * dpdy)
        if let op = uOpY { uy = op.apply(uy, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }

        // Density update.
        let duxdx = MLXFFT.ifft2(ddxNeg * kappa * MLXFFT.fft2(ux.asType(.complex64))).realPart()
        let duydy = MLXFFT.ifft2(ddyNeg * kappa * MLXFFT.fft2(uy.asType(.complex64))).realPart()
        rhox = pmlX * (pmlX * rhox - dtRho * duxdx)
        if let op = pOpX { rhox = op.apply(rhox, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }
        rhoy = pmlY * (pmlY * rhoy - dtRho * duydy)
        if let op = pOpY { rhoy = op.apply(rhoy, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }

        // Equation of state (linear, lossless).
        p = c2 * (rhox + rhoy)

        // t=0 leapfrog override for an initial-pressure source.
        if t == 0, let f = p0Field {
            p = f
            rhox = f / (2 * c2)
            rhoy = f / (2 * c2)
            let pk0 = MLXFFT.fft2(p.asType(.complex64))
            ux = (dtOverRho / 2) * MLXFFT.ifft2(ddxPos * kappa * pk0).realPart()
            uy = (dtOverRho / 2) * MLXFFT.ifft2(ddyPos * kappa * pk0).realPart()
        }

        if let idx = sensorIndices { recorded.append(p.reshaped([nx * ny])[idx]) }
        MLX.eval(p, ux, uy, rhox, rhoy)
    }

    var output = SimulationOutput()
    output.pFinal = p
    if !recorded.isEmpty { output.p = MLX.stacked(recorded, axis: 1) }
    return output
}

private func kspaceFirstOrder3D(
    grid: KWaveGrid,
    medium: KWaveMedium,
    source: KWaveSource,
    sensor: KWaveSensor,
    options: SimulationOptions
) -> SimulationOutput {
    precondition(medium.isHomogeneous, "3D solver requires a homogeneous medium")
    precondition(grid.nt > 0, "call grid.makeTime(...) before running the solver")
    precondition(source.p0 != nil || source.pMask != nil || source.uMask != nil,
                 "3D solver requires source.p0 or a time-varying p/u source")

    let nx = grid.nx, ny = grid.ny, nz = grid.nz
    let shape = [nx, ny, nz]
    let dt = grid.dt
    let c0 = medium.soundSpeed.item(Float.self)
    let rho0 = medium.density.item(Float.self)

    let arg = grid.k * (Double(c0) * dt / 2.0)
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
        let v = MLXArray(converting: pmlProfile(n: n, dx: d, dt: dt, c: Double(c0),
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

    let c2 = Float(c0 * c0)
    let dtOverRho = Float(dt) / rho0
    let dtRho = Float(dt) * rho0
    let ndim = 3

    let pScaleDirichlet = Float(1.0 / (Double(ndim) * Double(c0) * Double(c0)))
    let pScaleAddX = Float(2.0 * dt / (Double(ndim) * Double(c0) * grid.dx))
    let pScaleAddY = Float(2.0 * dt / (Double(ndim) * Double(c0) * grid.dy))
    let pScaleAddZ = Float(2.0 * dt / (Double(ndim) * Double(c0) * grid.dz))
    let pOpX = makeSourceOp(mask: source.pMask, signal: source.p, mode: source.pMode,
                            scalePerPoint: source.pMode == .dirichlet ? pScaleDirichlet : pScaleAddX,
                            shape: shape)
    let pOpY = makeSourceOp(mask: source.pMask, signal: source.p, mode: source.pMode,
                            scalePerPoint: source.pMode == .dirichlet ? pScaleDirichlet : pScaleAddY,
                            shape: shape)
    let pOpZ = makeSourceOp(mask: source.pMask, signal: source.p, mode: source.pMode,
                            scalePerPoint: source.pMode == .dirichlet ? pScaleDirichlet : pScaleAddZ,
                            shape: shape)
    let uScaleX = source.uMode == .dirichlet ? Float(1) : Float(2.0 * Double(c0) * dt / grid.dx)
    let uScaleY = source.uMode == .dirichlet ? Float(1) : Float(2.0 * Double(c0) * dt / grid.dy)
    let uScaleZ = source.uMode == .dirichlet ? Float(1) : Float(2.0 * Double(c0) * dt / grid.dz)
    let uOpX = makeSourceOp(mask: source.uMask, signal: source.ux, mode: source.uMode,
                            scalePerPoint: uScaleX, shape: shape)
    let uOpY = makeSourceOp(mask: source.uMask, signal: source.uy, mode: source.uMode,
                            scalePerPoint: uScaleY, shape: shape)
    let uOpZ = makeSourceOp(mask: source.uMask, signal: source.uz, mode: source.uMode,
                            scalePerPoint: uScaleZ, shape: shape)

    var p0Field: MLXArray? = nil
    if let p0 = source.p0 {
        var f = p0.asType(.float32)
        if options.smoothP0 { f = smooth3D(f) }
        p0Field = f
    }

    let fwd: (MLXArray) -> MLXArray = { MLXFFT.fftn($0) }
    let inv: (MLXArray) -> MLXArray = { MLXFFT.ifftn($0) }

    var p = MLXArray.zeros(shape, dtype: .float32)
    var ux = MLXArray.zeros(shape, dtype: .float32)
    var uy = MLXArray.zeros(shape, dtype: .float32)
    var uz = MLXArray.zeros(shape, dtype: .float32)
    var rhox = MLXArray.zeros(shape, dtype: .float32)
    var rhoy = MLXArray.zeros(shape, dtype: .float32)
    var rhoz = MLXArray.zeros(shape, dtype: .float32)

    let sensorIndices: MLXArray? = sensor.mask.map { flatNonzeroIndices($0) }
    var recorded: [MLXArray] = []
    recorded.reserveCapacity(grid.nt)

    for t in 0..<grid.nt {
        // Velocity update.
        let pk = MLXFFT.fftn(p.asType(.complex64))
        let dpdx = MLXFFT.ifftn(ddxPos * kappa * pk).realPart()
        let dpdy = MLXFFT.ifftn(ddyPos * kappa * pk).realPart()
        let dpdz = MLXFFT.ifftn(ddzPos * kappa * pk).realPart()
        ux = pmlXsg * (pmlXsg * ux - dtOverRho * dpdx)
        if let op = uOpX { ux = op.apply(ux, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }
        uy = pmlYsg * (pmlYsg * uy - dtOverRho * dpdy)
        if let op = uOpY { uy = op.apply(uy, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }
        uz = pmlZsg * (pmlZsg * uz - dtOverRho * dpdz)
        if let op = uOpZ { uz = op.apply(uz, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }

        // Density update.
        let duxdx = MLXFFT.ifftn(ddxNeg * kappa * MLXFFT.fftn(ux.asType(.complex64))).realPart()
        let duydy = MLXFFT.ifftn(ddyNeg * kappa * MLXFFT.fftn(uy.asType(.complex64))).realPart()
        let duzdz = MLXFFT.ifftn(ddzNeg * kappa * MLXFFT.fftn(uz.asType(.complex64))).realPart()
        rhox = pmlX * (pmlX * rhox - dtRho * duxdx)
        if let op = pOpX { rhox = op.apply(rhox, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }
        rhoy = pmlY * (pmlY * rhoy - dtRho * duydy)
        if let op = pOpY { rhoy = op.apply(rhoy, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }
        rhoz = pmlZ * (pmlZ * rhoz - dtRho * duzdz)
        if let op = pOpZ { rhoz = op.apply(rhoz, t: t, sourceKappa: sourceKappa, fwd: fwd, inv: inv) }

        // Equation of state (linear, lossless).
        p = c2 * (rhox + rhoy + rhoz)

        if t == 0, let f = p0Field {
            p = f
            rhox = f / (3 * c2)
            rhoy = f / (3 * c2)
            rhoz = f / (3 * c2)
            let pk0 = MLXFFT.fftn(p.asType(.complex64))
            ux = (dtOverRho / 2) * MLXFFT.ifftn(ddxPos * kappa * pk0).realPart()
            uy = (dtOverRho / 2) * MLXFFT.ifftn(ddyPos * kappa * pk0).realPart()
            uz = (dtOverRho / 2) * MLXFFT.ifftn(ddzPos * kappa * pk0).realPart()
        }

        if let idx = sensorIndices { recorded.append(p.reshaped([nx * ny * nz])[idx]) }
        MLX.eval(p, ux, uy, uz, rhox, rhoy, rhoz)
    }

    var output = SimulationOutput()
    output.pFinal = p
    if !recorded.isEmpty { output.p = MLX.stacked(recorded, axis: 1) }
    return output
}

/// i*k * exp(±i*k*Δ/2) staggered derivative multiplier (FFT order), as a complex64 vector.
private func derivativeOperator(_ kVec: MLXArray, spacing: Double, shift: Int) -> MLXArray {
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
private func flatNonzeroIndices(_ mask: MLXArray) -> MLXArray {
    let host = mask.reshaped([mask.size]).asArray(Float.self)
    let idx = host.enumerated().compactMap { $0.element != 0 ? Int32($0.offset) : nil }
    return MLXArray(idx)
}

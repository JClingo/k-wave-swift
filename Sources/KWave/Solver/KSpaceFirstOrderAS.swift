import Foundation
import MLX
import MLXFFT

/// Axisymmetric k-space pseudospectral solver (`kspaceFirstOrderAS`), WSWA-FFT radial symmetry —
/// a 1:1 port of MATLAB k-Wave's default variant. The grid is `[nx, ny]` with x the axial
/// dimension (FFT, periodic) and y the radial dimension (r ≥ 0). Radial symmetry is imposed by
/// mirroring each field into a 4×-expanded radial grid with its variable-specific symmetry
/// (p, ux: WSWA; uy: HAHS; uy/r: HSHA) and using ordinary Fourier transforms on the expansion.
///
/// The radial PML is one-sided (outer edge only); the equation of state gains the cylindrical
/// `uy/r` divergence term evaluated on the staggered radial grid (`r = dy/2, 3dy/2, …`, never 0).
///
/// Scope: linear, lossless; scalar or spatially varying `c0`/`rho0`; initial-pressure (p0)
/// and/or time-varying pressure sources (dirichlet, additive, additive-no-correction); binary
/// sensor mask recording `p`/`pFinal`. Velocity sources, absorption, and nonlinearity follow in
/// later slices.
public func kspaceFirstOrderAS(
    grid: KWaveGrid,
    medium: KWaveMedium,
    source: KWaveSource,
    sensor: KWaveSensor,
    options: SimulationOptions = .init()
) -> SimulationOutput {
    precondition(grid.dim == 2, "axisymmetric solver requires a 2D grid (x axial, y radial)")
    precondition(grid.nt > 0, "call grid.makeTime(...) before running the solver")
    precondition(source.p0 != nil || source.pMask != nil,
                 "axisymmetric solver requires source.p0 or a time-varying pressure source")
    precondition(source.uMask == nil, "velocity sources are not yet supported in the AS solver")

    let nx = grid.nx, ny = grid.ny
    let shape = [nx, ny]
    let dt = grid.dt

    var c0Grid = medium.soundSpeed.asType(.float32)
    var rho0Grid = medium.density.asType(.float32)
    precondition(c0Grid.size == 1 || c0Grid.shape == shape, "soundSpeed must be scalar or [nx, ny]")
    precondition(rho0Grid.size == 1 || rho0Grid.shape == shape, "density must be scalar or [nx, ny]")
    if options.smoothC0 && c0Grid.size > 1 { c0Grid = smooth(c0Grid) }
    if options.smoothRho0 && rho0Grid.size > 1 { rho0Grid = smooth(rho0Grid) }
    let cRef = referenceSoundSpeed(c0Grid)

    // Expanded grid for the WSWA mirroring (radial dimension ×4).
    let gridExp = KWaveGrid(nx: nx, dx: grid.dx, ny: 4 * ny, dy: grid.dy)

    // k-space operator on the expanded grid; axial derivative+shift; radial derivative and
    // half-grid shifts from the expanded ky (FFT order, rows broadcast along y).
    let arg = gridExp.k * (cRef * dt / 2.0)
    let kappa = MLX.which(arg .== 0, MLXArray(Float(1)), MLX.sin(arg) / arg).asType(.float32)

    let ddxPos = derivativeOperator(grid.kxVec, spacing: grid.dx, shift: +1).reshaped([nx, 1])
    let ddxNeg = derivativeOperator(grid.kxVec, spacing: grid.dx, shift: -1).reshaped([nx, 1])

    let nyE = 4 * ny
    let kyE = gridExp.kyVec.asType(.float32).reshaped([1, nyE])
    let ddyK = imagUnit() * kyE
    let thetaY = kyE * Float(grid.dy / 2)
    let yShiftPos = MLX.cos(thetaY) + imagUnit() * MLX.sin(thetaY)   // exp(+i·ky·dy/2)
    let yShiftNeg = MLX.cos(thetaY) - imagUnit() * MLX.sin(thetaY)   // exp(−i·ky·dy/2)

    // PMLs: axial standard two-sided; radial one-sided (outer edge only).
    let pmlSizeN: Int = { switch options.pmlSize { case let .uniform(s): return s } }()
    func pml(_ n: Int, _ d: Double, axis: Int, staggered: Bool, axisymmetric: Bool) -> MLXArray {
        let v = MLXArray(converting: pmlProfile(n: n, dx: d, dt: dt, c: cRef,
                                                pmlSize: pmlSizeN, pmlAlpha: options.pmlAlpha,
                                                staggered: staggered, axisymmetric: axisymmetric))
            .asType(.float32)
        return axis == 0 ? v.reshaped([n, 1]) : v.reshaped([1, n])
    }
    let pmlX = pml(nx, grid.dx, axis: 0, staggered: false, axisymmetric: false)
    let pmlXsg = pml(nx, grid.dx, axis: 0, staggered: true, axisymmetric: false)
    let pmlY = pml(ny, grid.dy, axis: 1, staggered: false, axisymmetric: true)
    let pmlYsg = pml(ny, grid.dy, axis: 1, staggered: true, axisymmetric: true)

    // Staggered radial coordinate r = dy/2, 3dy/2, … (never zero on the staggered grid).
    let invYsg = MLXArray((0..<ny).map { Float(1.0 / ((Double($0) + 0.5) * grid.dy)) })
        .reshaped([1, ny])

    let c2Grid = c0Grid * c0Grid
    let dtRho0 = Float(dt) * rho0Grid
    let dtOverRhoX = Float(dt) / staggerDensity(rho0Grid, axis: 0)
    let dtOverRhoY = Float(dt) / staggerDensity(rho0Grid, axis: 1)

    var p0Field: MLXArray? = nil
    if let p0 = source.p0 {
        var f = p0.asType(.float32)
        if options.smoothP0 { f = smooth(f) }
        p0Field = f
    }

    // Time-varying pressure source: injected as a mass source into BOTH rho splits with a single
    // scale (kspaceFirstOrder_scaleSourceTerms, uniform grid, N = 2 splits):
    //   dirichlet: p/(2·c0²)   additive: p·2·dt/(2·c0·dx).
    struct ASPressureSource {
        let indices: MLXArray       // Int32 flat indices, length nSrc.
        let signal: [[Float]]       // [rows][len]; 1 row broadcasts.
        let scale: [Float]          // per-point scale.
        let nSrc: Int
        func valuesAt(_ t: Int) -> MLXArray? {
            guard t < (signal.first?.count ?? 0) else { return nil }
            if signal.count == 1 {
                let base = signal[0][t]
                return MLXArray(scale.map { base * $0 })
            }
            return MLXArray((0..<nSrc).map { signal[$0][t] * scale[$0] })
        }
    }
    var pSource: ASPressureSource? = nil
    var sourceKappa: MLXArray? = nil
    if let mask = source.pMask, let sig = source.p {
        let idx = flatNonzeroIndices(mask)
        let nSrc = idx.size
        let cAt = soundSpeedSamples(c0Grid, at: idx, count: nSrc)
        let scale: [Float]
        if source.pMode == .dirichlet {
            scale = cAt.map { 1 / (2 * $0 * $0) }
        } else {
            scale = cAt.map { Float(2.0 * dt / (2.0 * Double($0) * grid.dx)) }
        }
        let rows: [[Float]]
        if sig.ndim <= 1 {
            rows = [sig.asArray(Float.self)]
        } else {
            let r = sig.dim(0), c = sig.dim(1)
            let flat = sig.reshaped([r * c]).asArray(Float.self)
            rows = (0..<r).map { Array(flat[$0 * c ..< ($0 + 1) * c]) }
        }
        pSource = ASPressureSource(indices: idx, signal: rows, scale: scale, nSrc: nSrc)
        if source.pMode == .additive {
            sourceKappa = MLX.cos(arg).asType(.float32)        // cos(c_ref·k_exp·dt/2), expanded.
        }
    }

    // Radial mirroring into the 4×-expanded grid. Reversed-index gathers are precomputed.
    let revAll = MLXArray((0..<ny).reversed().map { Int32($0) })       // fliplr.
    let revTail = MLXArray((1..<ny).reversed().map { Int32($0) })      // fliplr of columns 1...
    let zeroCol = MLXArray.zeros([nx, 1], dtype: .float32)
    func mirrorWSWA(_ w: MLXArray) -> MLXArray {
        let tail = MLX.take(w, revTail, axis: 1)
        return MLX.concatenated([w, zeroCol, -tail, -w, zeroCol, tail], axis: 1)
    }
    func mirrorHAHS(_ w: MLXArray) -> MLXArray {
        let flip = MLX.take(w, revAll, axis: 1)
        return MLX.concatenated([w, flip, -w, -flip], axis: 1)
    }
    func mirrorHSHA(_ w: MLXArray) -> MLXArray {
        let flip = MLX.take(w, revAll, axis: 1)
        return MLX.concatenated([w, -flip, -w, flip], axis: 1)
    }

    /// Pressure gradients on the staggered grids from the WSWA-mirrored field.
    func pressureGradients(_ p: MLXArray) -> (dpdx: MLXArray, dpdy: MLXArray) {
        let pk = kappa.asType(.complex64) * MLXFFT.fft2(mirrorWSWA(p).asType(.complex64))
        let dpdx = MLXFFT.ifft2(ddxPos * pk).realPart()[0..<nx, 0..<ny]
        let dpdy = MLXFFT.ifft2(ddyK * yShiftPos * pk).realPart()[0..<nx, 0..<ny]
        return (dpdx, dpdy)
    }

    var p = MLXArray.zeros(shape, dtype: .float32)
    var ux = MLXArray.zeros(shape, dtype: .float32)
    var uy = MLXArray.zeros(shape, dtype: .float32)
    var rhox = MLXArray.zeros(shape, dtype: .float32)
    var rhoy = MLXArray.zeros(shape, dtype: .float32)

    let plan = RecordPlan(sensor.record)
    let sampler: SensorSampler? = sensor.mask.map { makeSensorSampler(mask: $0, grid: grid) }
    var pRec: [MLXArray] = []

    for t in 0..<grid.nt {
        // Velocity update from the pressure gradients.
        let (dpdx, dpdy) = pressureGradients(p)
        ux = pmlXsg * (pmlXsg * ux - dtOverRhoX * dpdx)
        uy = pmlYsg * (pmlYsg * uy - dtOverRhoY * dpdy)

        // Velocity divergence: axial term plus the cylindrical radial term ∂uy/∂y + uy/r.
        let uxK = kappa.asType(.complex64) * MLXFFT.fft2(mirrorWSWA(ux).asType(.complex64))
        let duxdx = MLXFFT.ifft2(ddxNeg * uxK).realPart()[0..<nx, 0..<ny]
        let uyK = ddyK * MLXFFT.fft2(mirrorHAHS(uy).asType(.complex64))
            + MLXFFT.fft2(mirrorHSHA(invYsg * uy).asType(.complex64))
        let duydy = MLXFFT.ifft2(kappa.asType(.complex64) * yShiftNeg * uyK)
            .realPart()[0..<nx, 0..<ny]

        // Mass conservation.
        rhox = pmlX * (pmlX * rhox - dtRho0 * duxdx)
        rhoy = pmlY * (pmlY * rhoy - dtRho0 * duydy)

        // Pre-scaled pressure source injected identically into both rho splits.
        if let src = pSource, let vals = src.valuesAt(t) {
            switch source.pMode {
            case .dirichlet:
                var fx = rhox.reshaped([nx * ny]); fx[src.indices] = vals
                var fy = rhoy.reshaped([nx * ny]); fy[src.indices] = vals
                rhox = fx.reshaped(shape); rhoy = fy.reshaped(shape)
            case .additiveNoCorrection:
                let buf = MLXArray.zeros([nx * ny], dtype: .float32)
                buf[src.indices] = vals
                let mat = buf.reshaped(shape)
                rhox = rhox + mat; rhoy = rhoy + mat
            case .additive:
                let buf = MLXArray.zeros([nx * ny], dtype: .float32)
                buf[src.indices] = vals
                // k-space source correction on the WSWA-mirrored expansion.
                let spec = sourceKappa!.asType(.complex64)
                    * MLXFFT.fft2(mirrorWSWA(buf.reshaped(shape)).asType(.complex64))
                let mat = MLXFFT.ifft2(spec).realPart()[0..<nx, 0..<ny]
                rhox = rhox + mat; rhoy = rhoy + mat
            }
        }

        // Equation of state (linear, lossless).
        p = c2Grid * (rhox + rhoy)

        // t=0 leapfrog override for the initial-pressure source.
        if t == 0, let f = p0Field {
            p = f
            rhox = f / (2 * c2Grid)
            rhoy = f / (2 * c2Grid)
            let (dpdx0, dpdy0) = pressureGradients(p)
            ux = (dtOverRhoX / 2) * dpdx0
            uy = (dtOverRhoY / 2) * dpdy0
        }

        if let sampler, plan.recordP { pRec.append(sampler.sample(p)) }
        options.progress?(t, grid.nt)
        MLX.eval(p, ux, uy, rhox, rhoy)
    }

    var output = SimulationOutput()
    output.pFinal = p
    if !pRec.isEmpty { output.p = MLX.stacked(pRec, axis: 1) }
    return output
}

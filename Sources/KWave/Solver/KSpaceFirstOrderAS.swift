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
/// Scope: scalar or spatially varying `c0`/`rho0`; initial-pressure (p0) and/or time-varying
/// pressure sources (dirichlet, additive, additive-no-correction); Stokes absorption
/// (`alphaMode == .stokes` or `alphaPower == 2` — the only absorption model k-Wave supports
/// axisymmetrically); B/A nonlinearity; binary sensor mask recording `p`/`pFinal`. Velocity
/// sources follow in a later slice.
public func kspaceFirstOrderAS(
    grid: KWaveGrid,
    medium: KWaveMedium,
    source: KWaveSource,
    sensor: KWaveSensor,
    options: SimulationOptions = .init()
) -> SimulationOutput {
    precondition(grid.dim == 2, "axisymmetric solver requires a 2D grid (x axial, y radial)")
    precondition(grid.nt > 0, "call grid.makeTime(...) before running the solver")
    precondition(source.p0 != nil || source.pMask != nil || source.uMask != nil,
                 "axisymmetric solver requires source.p0 or a time-varying p/u source")

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

    // Stokes absorption (the only model supported axisymmetrically): tau = −2·αNp·c0 with
    // αNp = db2neper(alphaCoeff, 2). The EOS gains `+ tau·ρ0·(∇·u)`.
    var absorbTau: MLXArray? = nil
    if let alphaCoeff = medium.alphaCoeff, medium.alphaMode != .noAbsorption {
        let y = medium.alphaMode == .stokes ? 2.0 : (medium.alphaPower ?? 2.0)
        precondition(abs(y - 2.0) < 1e-10,
                     "axisymmetric absorption requires the Stokes model (alphaPower = 2)")
        absorbTau = -2 * alphaCoeff.asType(.float32) * Float(db2neper(1.0, y: 2.0)) * c0Grid
    }

    // B/A nonlinearity: nonlinear mass conservation uses dt·(2·Σρ + ρ0)·∇·u, and the EOS gains
    // `+ B/A·ρ²/(2ρ0)` (MATLAB kspaceFirstOrderAS, flags.nonlinear).
    let bOnA: MLXArray? = medium.bOnA.flatMap { b -> MLXArray? in
        let bf = b.asType(.float32)
        return MLX.max(MLX.abs(bf)).item(Float.self) != 0 ? bf : nil
    }

    var p0Field: MLXArray? = nil
    if let p0 = source.p0 {
        var f = p0.asType(.float32)
        if options.smoothP0 { f = smooth(f) }
        p0Field = f
    }

    // Time-varying sources. Scaling per kspaceFirstOrder_scaleSourceTerms (uniform grid):
    //   pressure (mass source into BOTH rho splits, N = 2): dirichlet p/(2·c0²),
    //   additive p·2·dt/(2·c0·dx); velocity: dirichlet unscaled, additive u·2·c0·dt/d (per axis).
    struct ASSource {
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
        /// The scaled source values scattered into a grid-shaped matrix, or `nil` once exhausted.
        func matrixAt(_ t: Int, shape: [Int]) -> MLXArray? {
            guard let vals = valuesAt(t) else { return nil }
            let buf = MLXArray.zeros([shape[0] * shape[1]], dtype: .float32)
            buf[indices] = vals
            return buf.reshaped(shape)
        }
    }
    func makeASSource(mask: MLXArray?, signal: MLXArray?, scale: (Float) -> Float) -> ASSource? {
        guard let mask, let signal else { return nil }
        let idx = flatNonzeroIndices(mask)
        let nSrc = idx.size
        guard nSrc > 0 else { return nil }
        let scales = soundSpeedSamples(c0Grid, at: idx, count: nSrc).map(scale)
        let rows: [[Float]]
        if signal.ndim <= 1 {
            rows = [signal.asArray(Float.self)]
        } else {
            let r = signal.dim(0), c = signal.dim(1)
            let flat = signal.reshaped([r * c]).asArray(Float.self)
            rows = (0..<r).map { Array(flat[$0 * c ..< ($0 + 1) * c]) }
        }
        return ASSource(indices: idx, signal: rows, scale: scales, nSrc: nSrc)
    }

    let pSource = makeASSource(mask: source.pMask, signal: source.p, scale: { c in
        source.pMode == .dirichlet ? 1 / (2 * c * c)
                                   : Float(2.0 * dt / (2.0 * Double(c) * grid.dx))
    })
    let uxSource = makeASSource(mask: source.uMask, signal: source.ux, scale: { c in
        source.uMode == .dirichlet ? 1 : Float(2.0 * Double(c) * dt / grid.dx)
    })
    let uySource = makeASSource(mask: source.uMask, signal: source.uy, scale: { c in
        source.uMode == .dirichlet ? 1 : Float(2.0 * Double(c) * dt / grid.dy)
    })
    let needsCorrection = (pSource != nil && source.pMode == .additive)
        || ((uxSource != nil || uySource != nil) && source.uMode == .additive)
    let sourceKappa: MLXArray? = needsCorrection
        ? MLX.cos(arg).asType(.float32) : nil                  // cos(c_ref·k_exp·dt/2), expanded.

    /// Additive k-space correction: mirror with the variable's symmetry, filter, trim.
    func correctSource(_ mat: MLXArray, mirror: (MLXArray) -> MLXArray) -> MLXArray {
        let spec = sourceKappa!.asType(.complex64) * MLXFFT.fft2(mirror(mat).asType(.complex64))
        return MLXFFT.ifft2(spec).realPart()[0..<nx, 0..<ny]
    }

    /// Apply a velocity source to `field` (dirichlet / additive / additive-no-correction).
    func applyVelocitySource(_ src: ASSource?, _ field: MLXArray, t: Int,
                             mirror: (MLXArray) -> MLXArray) -> MLXArray {
        guard let src else { return field }
        switch source.uMode {
        case .dirichlet:
            guard let vals = src.valuesAt(t) else { return field }
            let flat = field.reshaped([nx * ny])
            flat[src.indices] = vals
            return flat.reshaped(shape)
        case .additiveNoCorrection:
            guard let mat = src.matrixAt(t, shape: shape) else { return field }
            return field + mat
        case .additive:
            guard let mat = src.matrixAt(t, shape: shape) else { return field }
            return field + correctSource(mat, mirror: mirror)
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
        // Velocity update from the pressure gradients, plus velocity sources
        // (ux mirrors with WSWA symmetry, uy with HAHS — matching the field variables).
        let (dpdx, dpdy) = pressureGradients(p)
        ux = pmlXsg * (pmlXsg * ux - dtOverRhoX * dpdx)
        ux = applyVelocitySource(uxSource, ux, t: t, mirror: mirrorWSWA)
        uy = pmlYsg * (pmlYsg * uy - dtOverRhoY * dpdy)
        uy = applyVelocitySource(uySource, uy, t: t, mirror: mirrorHAHS)

        // Velocity divergence: axial term plus the cylindrical radial term ∂uy/∂y + uy/r.
        let uxK = kappa.asType(.complex64) * MLXFFT.fft2(mirrorWSWA(ux).asType(.complex64))
        let duxdx = MLXFFT.ifft2(ddxNeg * uxK).realPart()[0..<nx, 0..<ny]
        let uyK = ddyK * MLXFFT.fft2(mirrorHAHS(uy).asType(.complex64))
            + MLXFFT.fft2(mirrorHSHA(invYsg * uy).asType(.complex64))
        let duydy = MLXFFT.ifft2(kappa.asType(.complex64) * yShiftNeg * uyK)
            .realPart()[0..<nx, 0..<ny]

        // Mass conservation (nonlinear variant uses the previous step's total density).
        if let bOnA {
            let dtRhoTotal = Float(dt) * (2 * (rhox + rhoy) + rho0Grid)
            rhox = pmlX * (pmlX * rhox - dtRhoTotal * duxdx)
            rhoy = pmlY * (pmlY * rhoy - dtRhoTotal * duydy)
        } else {
            rhox = pmlX * (pmlX * rhox - dtRho0 * duxdx)
            rhoy = pmlY * (pmlY * rhoy - dtRho0 * duydy)
        }

        // Pre-scaled pressure source injected identically into both rho splits.
        if let src = pSource {
            switch source.pMode {
            case .dirichlet:
                if let vals = src.valuesAt(t) {
                    let fx = rhox.reshaped([nx * ny]); fx[src.indices] = vals
                    let fy = rhoy.reshaped([nx * ny]); fy[src.indices] = vals
                    rhox = fx.reshaped(shape); rhoy = fy.reshaped(shape)
                }
            case .additiveNoCorrection:
                if let mat = src.matrixAt(t, shape: shape) {
                    rhox = rhox + mat; rhoy = rhoy + mat
                }
            case .additive:
                if let mat = src.matrixAt(t, shape: shape) {
                    let corrected = correctSource(mat, mirror: mirrorWSWA)
                    rhox = rhox + corrected; rhoy = rhoy + corrected
                }
            }
        }

        // Equation of state: p = c0²·(ρ [+ τ·ρ0·∇·u] [+ B/A·ρ²/(2ρ0)]).
        let rhoTotal = rhox + rhoy
        var eosRho = rhoTotal
        if let absorbTau { eosRho = eosRho + absorbTau * rho0Grid * (duxdx + duydy) }
        if let bOnA { eosRho = eosRho + bOnA * (rhoTotal * rhoTotal) / (2 * rho0Grid) }
        p = c2Grid * eosRho

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

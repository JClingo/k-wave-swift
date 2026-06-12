import Foundation
import MLX
import MLXFFT

/// Smallest integer `p` with `2^p >= n` (k-Wave `nextpow2`).
private func nextPow2(_ n: Int) -> Int {
    max(0, Int(ceil(log2(Double(max(n, 1))))))
}

/// Continuous-wave (single-frequency) angular-spectrum projection, mirroring k-wave-python
/// `angular_spectrum_cw`. Projects a complex 2D source plane to one or more parallel planes at the
/// `zPos` offsets using the spectral propagator with optional angular restriction (Zeng & McGough,
/// JASA 2008).
///
/// Scope: lossless medium (scalar sound speed). Absorption, grid expansion, and data casting from
/// the MATLAB/Python API are not implemented.
///
/// - Parameters:
///   - inputPlane: `[Nx, Ny]` complex (or real) source pressure plane.
///   - dx: grid spacing [m].
///   - zPos: relative z-positions of the output planes [m].
///   - f0: source frequency [Hz].
///   - soundSpeed: medium sound speed [m/s].
///   - angularRestriction: apply the angular-restriction window (default true).
///   - fftLength: FFT size; `nil` selects `2^(nextpow2(max(Nx,Ny)) + 1)`.
///   - reverseProj: project in the reverse direction (phase-conjugate the input).
/// - Returns: `[Nx, Ny, Nz]` complex projected pressure.
public func angularSpectrumCW(
    inputPlane: MLXArray, dx: Double, zPos: [Double], f0: Double, soundSpeed: Double,
    angularRestriction: Bool = true, fftLength: Int? = nil, reverseProj: Bool = false
) -> MLXArray {
    precondition(inputPlane.ndim == 2, "inputPlane must be [Nx, Ny]")
    precondition(!zPos.isEmpty, "zPos must have at least one position")
    precondition(dx <= soundSpeed / (2 * f0), "f0 exceeds the maximum supported frequency c/(2·dx)")

    let nx = inputPlane.dim(0), ny = inputPlane.dim(1)
    var input = inputPlane.asType(.complex64)
    if reverseProj { input = MLX.conjugate(input) }

    let n = fftLength ?? (1 << (nextPow2(max(nx, ny)) + 1))

    // FFTW-ordered wavenumbers (centered + forced-zero + ifftshift ≡ 2π·fftfreq).
    let kvec = MLXArray(converting: wavenumberVector(n, d: dx))
    let kx = kvec.reshaped([1, n]), ky = kvec.reshaped([n, 1])
    let k = 2 * Double.pi * f0 / soundSpeed
    let kxy2 = kx * kx + ky * ky                 // broadcasts to [n, n].
    let sqrtKxy = MLX.sqrt(kxy2)

    let inputFFT = MLXFFT.fft2(input, s: [n, n])

    var slices: [MLXArray] = []
    slices.reserveCapacity(zPos.count)
    for z in zPos {
        if z == 0 { slices.append(input); continue }
        let h = spectralPropagator(k: k, z: z, kxy2: kxy2, sqrtKxy: sqrtKxy,
                                   n: n, dx: dx, angularRestriction: angularRestriction)
        let step = MLXFFT.ifft2(inputFFT * h, s: [n, n])
        slices.append(step[0..<nx, 0..<ny])
    }
    return MLX.stacked(slices, axis: 2)
}

/// Spectral propagator `H = conj(exp(i·z·kz))` with `kz = sqrt(k² − (kx²+ky²))` (real for
/// propagating components, imaginary for evanescent), optionally windowed by the angular-restriction
/// threshold `kc = k·sqrt(D²/2 / (D²/2 + z²))` (Zeng & McGough Eq. 10).
private func spectralPropagator(
    k: Double, z: Double, kxy2: MLXArray, sqrtKxy: MLXArray,
    n: Int, dx: Double, angularRestriction: Bool, alphaNp: Double = 0
) -> MLXArray {
    let arg = Float(k * k) - kxy2
    let kzRe = MLX.sqrt(MLX.maximum(arg, MLXArray(Float(0))))
    let kzIm = MLX.sqrt(MLX.maximum(-arg, MLXArray(Float(0))))

    // H = exp(-z·kzIm)·(cos(z·kzRe) − i·sin(z·kzRe)).
    let decay = MLX.exp(Float(-z) * kzIm)
    let angle = Float(z) * kzRe
    var hRe = decay * MLX.cos(angle)
    var hIm = -decay * MLX.sin(angle)

    // Attenuation (Zeng & McGough Eq. 11): H ·= exp(−αNp·z·k/kz). kz is real for propagating
    // components (a real decay factor) and imaginary for evanescent ones, where k/kz = −i·k/kzIm
    // makes the factor a pure phase. Applied before the angular restriction, matching k-Wave.
    if alphaNp != 0 {
        let azk = Float(alphaNp * z * k)
        let propagating = arg .> MLXArray(Float(0))
        let fRePropag = MLX.exp(-azk / kzRe)
        let theta = azk / kzIm
        let fRe = MLX.which(propagating, fRePropag, MLX.cos(theta))
        let fIm = MLX.which(propagating, MLXArray(Float(0)), MLX.sin(theta))
        let newRe = hRe * fRe - hIm * fIm
        let newIm = hRe * fIm + hIm * fRe
        hRe = newRe
        hIm = newIm
    }

    if angularRestriction {
        let d = Double(n - 1) * dx
        let kc = k * (0.5 * d * d / (0.5 * d * d + z * z)).squareRoot()
        let mask = sqrtKxy .> MLXArray(Float(kc))
        let zero = MLXArray(Float(0))
        hRe = MLX.which(mask, zero, hRe)
        hIm = MLX.which(mask, zero, hIm)
    }
    return hRe.asType(.complex64) + imagUnit() * hIm.asType(.complex64)
}

/// Broadband angular-spectrum projection, mirroring k-wave-python `angular_spectrum`. Projects a 2D
/// plane of *time series* `[Nx, Ny, Nt]` to parallel planes at the `zPos` offsets: the series are
/// decomposed into frequency components, each component is propagated with the spectral propagator
/// (with angular restriction), and the time series is rebuilt from the (conjugate-symmetric)
/// spectrum. Returns the maximum pressure over time at each plane, and optionally the full series.
///
/// Supports power-law absorption (`alphaCoeff`/`alphaPower`), grid expansion, and reverse
/// projection; data casting from the MATLAB/Python API is not applicable.
///
/// Note: k-wave-python's even-`Nt` double-sided spectrum rebuild drops one bin (`[1:-2]`, an
/// upstream off-by-one vs MATLAB's `2:end-1`); this port uses the correct MATLAB formula, so for
/// even `Nt` results agree with MATLAB, not the buggy Python path.
///
/// - Parameters:
///   - inputPlane: `[Nx, Ny, Nt]` real pressure time series over the input plane [Pa].
///   - dx: spatial step [m]; dt: temporal step [s]; zPos: projection offsets [m].
///   - soundSpeed: medium sound speed [m/s].
///   - angularRestriction: apply the angular-restriction window (default true).
///   - gridExpansion: zero-pad the plane by this many points per side before computation.
///   - fftLength: 2D FFT size; `nil` selects `2^(nextpow2(max(Nx,Ny)) + 1)`.
///   - recordTimeSeries: also return the projected time series.
/// - Returns: `pressureMax` `[Nx, Ny, Nz]`, and `pressureTime` `[Nx, Ny, Nt, Nz]` when requested.
public func angularSpectrum(
    inputPlane: MLXArray, dx: Double, dt: Double, zPos: [Double], soundSpeed: Double,
    alphaCoeff: Double? = nil, alphaPower: Double? = nil,
    angularRestriction: Bool = true, gridExpansion: Int = 0, fftLength: Int? = nil,
    recordTimeSeries: Bool = false, reverseProj: Bool = false
) -> (pressureMax: MLXArray, pressureTime: MLXArray?) {
    precondition(inputPlane.ndim == 3, "inputPlane must be [Nx, Ny, Nt]")
    precondition(!zPos.isEmpty, "zPos must have at least one position")
    precondition((alphaCoeff == nil) == (alphaPower == nil),
                 "alphaCoeff and alphaPower must be provided together")
    precondition(soundSpeed * dt / dx <= 1,
                 "temporal sampling cannot resolve the maximum spatial frequency (CFL > 1)")

    var input = inputPlane.asType(.float32)
    if reverseProj {
        // Project backwards: reverse the time signals, project, and reverse the outputs again.
        let nt0 = input.dim(2)
        input = MLX.take(input, MLXArray((0..<nt0).reversed().map { Int32($0) }), axis: 2)
    }
    if gridExpansion > 0 {
        input = MLX.padded(input, widths: [.init((gridExpansion, gridExpansion)),
                                           .init((gridExpansion, gridExpansion)),
                                           .init((0, 0))])
    }
    let nx = input.dim(0), ny = input.dim(1), nt = input.dim(2)

    let n = fftLength ?? (1 << (nextPow2(max(nx, ny)) + 1))
    let kvec = MLXArray(converting: wavenumberVector(n, d: dx))
    let kxy2 = kvec.reshaped([1, n]) * kvec.reshaped([1, n])
        + kvec.reshaped([n, 1]) * kvec.reshaped([n, 1])
    let sqrtKxy = MLX.sqrt(kxy2)

    // Single-sided spectrum of each time series: nUnique = ceil((Nt+1)/2) bins at f = i/(dt·Nt).
    let spec = MLXFFT.fft(input.asType(.complex64), axis: 2)
    let nUnique = (nt + 2) / 2
    let fVec = (0..<nUnique).map { Double($0) / (dt * Double(nt)) }
    let nProp = fVec.filter { $0 < soundSpeed / (2 * dx) }.count

    var maxSlices: [MLXArray] = []
    var timeSlices: [MLXArray] = []
    let zeroPlane = MLXArray.zeros([nx, ny], dtype: .complex64)

    for z in zPos {
        if z == 0 {
            maxSlices.append(MLX.max(input, axis: 2))
            if recordTimeSeries { timeSlices.append(input) }
            continue
        }

        // Propagate each frequency component below the spatial Nyquist; others stay zero.
        var freqPlanes = [MLXArray](repeating: zeroPlane, count: nUnique)
        for f in 0..<nProp {
            let k = 2 * Double.pi * fVec[f] / soundSpeed
            // αNp = db2neper(α, y)·ω^y converts dB/(MHz^y cm) to Np/m at this frequency.
            var alphaNp = 0.0
            if let alphaCoeff, let alphaPower {
                alphaNp = db2neper(alphaCoeff, y: alphaPower)
                    * pow(2 * Double.pi * fVec[f], alphaPower)
            }
            let h = spectralPropagator(k: k, z: z, kxy2: kxy2, sqrtKxy: sqrtKxy,
                                       n: n, dx: dx, angularRestriction: angularRestriction,
                                       alphaNp: alphaNp)
            let plane = spec[0..<nx, 0..<ny, f..<(f + 1)].reshaped([nx, ny])
            let step = MLXFFT.ifft2(MLXFFT.fft2(plane, s: [n, n]) * h, s: [n, n])[0..<nx, 0..<ny]

            // Retarded-time phase shift exp(i·2πf·z/c).
            let phi = 2 * Double.pi * fVec[f] * z / soundSpeed
            let ret = MLXArray(Float(cos(phi))).asType(.complex64)
                + imagUnit() * MLXArray(Float(sin(phi))).asType(.complex64)
            freqPlanes[f] = step * ret
            MLX.eval(freqPlanes[f])
        }

        // Double-sided conjugate-symmetric spectrum (correct MATLAB formula for both parities).
        var sided = freqPlanes
        let mirrorEnd = nt % 2 == 1 ? nUnique : nUnique - 1
        for f in stride(from: mirrorEnd - 1, through: 1, by: -1) {
            sided.append(MLX.conjugate(freqPlanes[f]))
        }
        let series = MLXFFT.ifft(MLX.stacked(sided, axis: 2), axis: 2).realPart()
        maxSlices.append(MLX.max(series, axis: 2))
        if recordTimeSeries { timeSlices.append(series) }
    }

    var pressureMax = MLX.stacked(maxSlices, axis: 2)
    var pressureTime: MLXArray? = recordTimeSeries ? MLX.stacked(timeSlices, axis: 3) : nil
    if gridExpansion > 0 {
        let g = gridExpansion
        pressureMax = pressureMax[g..<(nx - g), g..<(ny - g), 0..<zPos.count]
        pressureTime = pressureTime.map { $0[g..<(nx - g), g..<(ny - g), 0..<nt, 0..<zPos.count] }
    }
    if reverseProj {
        // Mirror k-wave-python: pressure_max flips over the z axis, the time series over time.
        let revZ = MLXArray((0..<zPos.count).reversed().map { Int32($0) })
        pressureMax = MLX.take(pressureMax, revZ, axis: 2)
        let revT = MLXArray((0..<nt).reversed().map { Int32($0) })
        pressureTime = pressureTime.map { MLX.take($0, revT, axis: 2) }
    }
    return (pressureMax, pressureTime)
}

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
    let kvec = MLXArray(wavenumberVector(n, d: dx)).asType(.float32)
    let kx = kvec.reshaped([1, n]), ky = kvec.reshaped([n, 1])
    let k = 2 * Double.pi * f0 / soundSpeed
    let kxy2 = kx * kx + ky * ky                 // broadcasts to [n, n].
    let arg = Float(k * k) - kxy2

    // kz = sqrt(k² − (kx²+ky²)): real for propagating, imaginary for evanescent.
    let kzRe = MLX.sqrt(MLX.maximum(arg, MLXArray(Float(0))))
    let kzIm = MLX.sqrt(MLX.maximum(-arg, MLXArray(Float(0))))
    let sqrtKxy = MLX.sqrt(kxy2)

    let inputFFT = MLXFFT.fft2(input, s: [n, n])

    var slices: [MLXArray] = []
    slices.reserveCapacity(zPos.count)
    for z in zPos {
        if z == 0 { slices.append(input); continue }

        // H = conj(exp(i·z·kz)) = exp(-z·kzIm)·(cos(z·kzRe) − i·sin(z·kzRe)).
        let decay = MLX.exp(Float(-z) * kzIm)
        let angle = Float(z) * kzRe
        var hRe = decay * MLX.cos(angle)
        var hIm = -decay * MLX.sin(angle)

        if angularRestriction {
            let d = Double(n - 1) * dx
            let kc = k * (0.5 * d * d / (0.5 * d * d + z * z)).squareRoot()
            let mask = sqrtKxy .> MLXArray(Float(kc))
            let zero = MLXArray(Float(0))
            hRe = MLX.which(mask, zero, hRe)
            hIm = MLX.which(mask, zero, hIm)
        }

        let h = hRe.asType(.complex64) + imagUnit() * hIm.asType(.complex64)
        let step = MLXFFT.ifft2(inputFFT * h, s: [n, n])
        slices.append(step[0..<nx, 0..<ny])
    }
    return MLX.stacked(slices, axis: 2)
}

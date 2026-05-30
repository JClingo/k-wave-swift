import Foundation
import MLX
import MLXFFT

/// Imaginary unit as a complex64 scalar.
func imagUnit() -> MLXArray { MLXArray(real: 0, imaginary: 1) }

/// Spectral first derivative along `axis`: d/dx f = ifft( i*k * fft(f) ).
///
/// `kVec` is the 1-D angular wavenumber vector in FFT order, reshaped to broadcast along `axis`.
/// Returns a real array (the imaginary residue from round-off is discarded).
func spectralDerivative(_ field: MLXArray, kVec: MLXArray, axis: Int) -> MLXArray {
    let spectrum = MLXFFT.fft(field.asType(.complex64), axis: axis)
    let deriv = MLXFFT.ifft(imagUnit() * kVec * spectrum, axis: axis)
    return deriv.realPart()
}

/// Staggered-grid spectral derivative operators used by the first-order solver.
///
/// On a staggered grid, the gradient that maps a quantity from the regular grid to the
/// half-step-shifted grid (and back) uses `i*k * exp(±i*k*Δ/2)`. `forward` shifts by `+Δ/2`,
/// `backward` shifts by `-Δ/2`. Both are complex multipliers in FFT order, shaped to broadcast
/// along their axis.
struct StaggeredDerivative {
    let forward: MLXArray   // i*k * exp(+i*k*Δ/2)
    let backward: MLXArray  // i*k * exp(-i*k*Δ/2)
    let axis: Int

    init(kVec: MLXArray, spacing: Double, axis: Int) {
        let theta = kVec * (spacing / 2.0)
        let phasePos = MLX.cos(theta) + imagUnit() * MLX.sin(theta)
        let phaseNeg = MLX.cos(theta) - imagUnit() * MLX.sin(theta)
        let ik = imagUnit() * kVec
        self.forward = ik * phasePos
        self.backward = ik * phaseNeg
        self.axis = axis
    }

    /// Apply the forward (+Δ/2) staggered derivative to a real field.
    func applyForward(_ field: MLXArray) -> MLXArray {
        let spectrum = MLXFFT.fft(field.asType(.complex64), axis: axis)
        return MLXFFT.ifft(forward * spectrum, axis: axis).realPart()
    }

    /// Apply the backward (-Δ/2) staggered derivative to a real field.
    func applyBackward(_ field: MLXArray) -> MLXArray {
        let spectrum = MLXFFT.fft(field.asType(.complex64), axis: axis)
        return MLXFFT.ifft(backward * spectrum, axis: axis).realPart()
    }
}

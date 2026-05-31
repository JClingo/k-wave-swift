import Foundation
import MLX
import MLXFFT

/// Resample `data` along `axis` by a non-dimensional `shift` (in samples) using a Fourier
/// interpolant, mirroring k-Wave `fourierShift` / k-wave-python `phase_shift_interpolate`.
///
/// `shift = 0.5` moves a staggered-grid quantity onto the regular grid; `shift = 1` is a full
/// grid point. The shift is applied as `exp(2πi·f_k·shift)` in the frequency domain, where `f_k`
/// are the `fftfreq` bins of the chosen axis, and the real part of the inverse transform is
/// returned (matching the negate-then-`scipy.ndimage.fourier_shift` convention).
public func fourierShift(_ data: MLXArray, shift: Double, axis: Int = -1) -> MLXArray {
    let ndim = max(data.ndim, 1)
    let ax = axis < 0 ? ndim + axis : axis
    precondition(ax >= 0 && ax < data.ndim, "fourierShift: axis \(axis) out of range")
    let n = data.dim(ax)

    // fftfreq(n): f_k = k/n for k < (n+1)/2, else (k-n)/n. Multiplier = exp(2πi·f·shift).
    let half = (n + 1) / 2
    let twoPiShift = 2.0 * Double.pi * shift
    let re = (0..<n).map { k -> Float in
        let f = Double(k < half ? k : k - n) / Double(n)
        return Float(Foundation.cos(twoPiShift * f))
    }
    let im = (0..<n).map { k -> Float in
        let f = Double(k < half ? k : k - n) / Double(n)
        return Float(Foundation.sin(twoPiShift * f))
    }
    var multShape = [Int](repeating: 1, count: data.ndim)
    multShape[ax] = n
    let mult = (MLXArray(re) + imagUnit() * MLXArray(im)).reshaped(multShape)

    let spec = MLXFFT.fft(data.asType(.complex64), axis: ax)
    return MLXFFT.ifft(mult * spec, axis: ax).realPart()
}

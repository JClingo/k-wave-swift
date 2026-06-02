import Foundation
import MLX

/// Variation of sound speed with frequency for a power-law-absorbing medium, via the Kramers-Kronig
/// relation (`att = a0·wʸ`), mirroring k-wave-python `power_law_kramers_kronig`.
///
/// Power-law parameters must be in Nepers/((rad/s)ʸ·m); frequencies in rad/s. The variation is given
/// about the sound speed `c0` at reference frequency `w0`.
///
/// - Parameters:
///   - w: frequency array [rad/s].
///   - w0: reference frequency [rad/s]; c0: sound speed at `w0` [m/s].
///   - a0: power-law coefficient [Np/((rad/s)ʸ·m)]; y: exponent, expected in (0, 3).
/// - Returns: sound speed at each `w` [m/s]. Returns `c0` everywhere when `y` is outside (0, 3).
public func powerLawKramersKronig(w: MLXArray, w0: Double, c0: Double, a0: Double, y: Double) -> MLXArray {
    let wf = w.asType(.float32)
    if y <= 0 || y >= 3 {
        return MLXArray.ones(wf.shape, dtype: .float32) * Float(c0)
    }
    if y == 1 {
        // c_kk = 1 / (1/c0 − 2·a0·ln(w/w0)/π)
        let denom = Float(1 / c0) - Float(2 * a0 / Double.pi) * MLX.log(wf / Float(w0))
        return Float(1) / denom
    }
    // c_kk = 1 / (1/c0 + a0·tan(yπ/2)·(w^(y−1) − w0^(y−1)))
    let coeff = Float(a0 * tan(y * Double.pi / 2))
    let denom = Float(1 / c0) + coeff * (MLX.pow(wf, Float(y - 1)) - Float(pow(w0, y - 1)))
    return Float(1) / denom
}

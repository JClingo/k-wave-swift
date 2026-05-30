import Foundation
import MLX

/// Discrete FFT sample frequencies (cycles per unit), matching numpy.fft.fftfreq.
/// Returns angular wavenumbers when scaled by 2*pi: k = 2*pi*fftfreq(n, d).
func fftFreq(_ n: Int, d: Double) -> [Double] {
    let val = 1.0 / (Double(n) * d)
    var result = [Double](repeating: 0, count: n)
    let cutoff = (n - 1) / 2 + 1   // number of non-negative frequencies
    for i in 0..<cutoff { result[i] = Double(i) * val }
    var j = -(n / 2)
    for i in cutoff..<n { result[i] = Double(j) * val; j += 1 }
    return result
}

/// Angular wavenumber vector in FFT order: 2*pi*fftfreq(n, d).
func wavenumberVector(_ n: Int, d: Double) -> [Double] {
    fftFreq(n, d: d).map { 2.0 * Double.pi * $0 }
}

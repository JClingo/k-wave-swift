import Foundation
import MLX
import MLXFFT

/// FFT window for `spect` / `extractAmpPhase`.
public enum SpectrumWindow: Sendable { case rectangular, hanning }

private func nextPow2(_ n: Int) -> Int { max(0, Int(ceil(log2(Double(max(n, 1)))))) }

/// Window samples and coherent gain `mean(win)`, mirroring k-wave-python `get_win`.
private func windowSamples(_ type: SpectrumWindow, _ n: Int, symmetric: Bool) -> ([Double], Double) {
    let win: [Double]
    switch type {
    case .rectangular:
        win = [Double](repeating: 1, count: n)
    case .hanning:
        // Non-symmetric windows are computed as the symmetric window of length n+1, first n samples.
        let m = symmetric ? n : n + 1
        win = (0..<n).map { 0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(m - 1)) }
    }
    return (win, win.reduce(0, +) / Double(n))
}

/// Single-sided amplitude/phase spectrum of a 1-D signal, mirroring k-wave-python `spect`.
///
/// - Returns: `(f, amp, phase)` — frequency axis [Hz], single-sided amplitude, phase [rad].
public func spect(
    _ signal: MLXArray, fs: Double, fftLen: Int = 0, powerTwo: Bool = false,
    window: SpectrumWindow = .rectangular, unwrapPhase: Bool = false
) -> (f: [Double], amp: [Double], phase: [Double]) {
    let host = signal.reshaped([signal.size]).asArray(Float.self).map { Double($0) }
    let n = host.count
    precondition(n > 1, "Input signal cannot be scalar.")

    let (win, cg) = windowSamples(window, n, symmetric: false)
    let windowed = zip(host, win).map { $0 * $1 }

    var len = fftLen
    if len <= 0 || len < n { len = powerTwo ? (1 << nextPow2(n)) : n }

    let spec = MLXFFT.fft(MLXArray(windowed.map { Float($0) }).asType(.complex64), n: len)
    let re = spec.realPart().asArray(Float.self)
    let im = spec.imaginaryPart().asArray(Float.self)

    let denom = Double(n) * cg + 1e-10
    let numUnique = (len + 1 + 1) / 2          // ceil((len+1)/2)
    let nyquistUnique = len % 2 == 0           // even len has a unique Nyquist bin

    var amp = [Double](repeating: 0, count: numUnique)
    var phase = [Double](repeating: 0, count: numUnique)
    var f = [Double](repeating: 0, count: numUnique)
    for k in 0..<numUnique {
        let a = (Double(re[k]) * Double(re[k]) + Double(im[k]) * Double(im[k])).squareRoot() / denom
        let isUnique = k == 0 || (nyquistUnique && k == numUnique - 1)
        amp[k] = isUnique ? a : 2 * a
        phase[k] = atan2(Double(im[k]), Double(re[k]))
        f[k] = Double(k) * fs / Double(len)
    }

    if unwrapPhase {
        for k in 1..<numUnique {
            var d = phase[k] - phase[k - 1]
            while d > .pi { d -= 2 * .pi }
            while d < -.pi { d += 2 * .pi }
            phase[k] = phase[k - 1] + d
        }
    }
    return (f, amp, phase)
}

/// Extract amplitude and phase of a 1-D time signal at the frequency closest to `sourceFreq`, via a
/// windowed, zero-padded FFT. Mirrors k-wave-python `extract_amp_phase` (1-D).
///
/// - Returns: `(amplitude, phase [rad], frequency [Hz])` at the nearest bin to `sourceFreq`.
public func extractAmpPhase(
    _ data: MLXArray, fs: Double, sourceFreq: Double, fftPadding: Int = 3, window: SpectrumWindow = .hanning
) -> (amp: Double, phase: Double, freq: Double) {
    let host = data.reshaped([data.size]).asArray(Float.self).map { Double($0) }
    let n = host.count
    let (win, cg) = windowSamples(window, n, symmetric: true)
    let windowed = zip(host, win).map { $0 * $1 }

    // spect applies a rectangular window (no-op) and scales by the signal length.
    let (f, ampRaw, phase) = spect(MLXArray(windowed.map { Float($0) }), fs: fs,
                                   fftLen: fftPadding * n, window: .rectangular)
    let amp = ampRaw.map { $0 / cg }

    var idx = 0, best = Double.greatestFiniteMagnitude
    for (i, fi) in f.enumerated() where abs(fi - sourceFreq) < best {
        best = abs(fi - sourceFreq); idx = i
    }
    return (amp[idx], phase[idx], f[idx])
}

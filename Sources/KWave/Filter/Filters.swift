import Foundation
import MLX
import MLXFFT

/// Window function types.
public enum WindowType: Sendable {
    case hann
    case blackman
}

/// Generate a symmetric window of length `n` (host array), matching common DSP definitions.
public func getWin(_ n: Int, type: WindowType) -> [Double] {
    guard n > 1 else { return [Double](repeating: 1, count: max(n, 0)) }
    let denom = Double(n - 1)
    return (0..<n).map { i in
        let x = Double(i) / denom
        switch type {
        case .hann:
            return 0.5 - 0.5 * cos(2 * .pi * x)
        case .blackman:
            return 0.42 - 0.5 * cos(2 * .pi * x) + 0.08 * cos(4 * .pi * x)
        }
    }
}

/// Apply a frequency-domain double-sided Gaussian band-pass filter to a 1-D signal.
///
/// Mirrors k-wave-python `gaussian_filter`: centre frequency `frequency` [Hz], `bandwidth` in
/// percent, sampling frequency `fs` [Hz]. Returns the real filtered signal.
public func gaussianFilter(
    _ signal: [Double], fs: Double, frequency: Double, bandwidth: Double
) -> [Double] {
    let n = signal.count
    guard n > 1 else { return signal }

    // Frequency axis (fftshift order: -N/2 .. N/2-1) scaled by fs/N.
    let f: [Double]
    if n % 2 == 0 {
        f = (0..<n).map { (Double($0) - Double(n) / 2) * fs / Double(n) }
    } else {
        f = (0..<n).map { (Double($0) - Double(n - 1) / 2) * fs / Double(n) }
    }

    let variance = pow(bandwidth / 100 * frequency / (2 * (2 * log(2.0)).squareRoot()), 2)
    let gPos = gaussian(f, magnitude: 1, mean: frequency, variance: variance)
    let gNeg = gaussian(f, magnitude: 1, mean: -frequency, variance: variance)
    let gfilter = zip(gPos, gNeg).map { Swift.max($0, $1) }   // shifted (peak-at-centre) order

    // signal -> fft -> fftshift -> *filter -> ifftshift -> ifft -> real.
    let spec = MLXFFT.fft(MLXArray(signal).asType(.complex64))
    let shifted = fftShift1D(spec)
    let filt = MLXArray(gfilter.map { Float($0) }).asType(.complex64)
    let out = MLXFFT.ifft(ifftShift1D(filt * shifted)).realPart()
    return out.asArray(Float.self).map(Double.init)
}

/// FIR filter type for `applyFilter`.
public enum FilterType: Sendable {
    case lowPass
    case highPass
    case bandPass
}

/// Unnormalised sinc: sin(x)/x with the removable singularity at 0 returning 1.
private func sincRaw(_ x: Double) -> Double { x == 0 ? 1 : sin(x) / x }

/// Modified Bessel function of the first kind, order 0 (series expansion).
private func besselI0(_ x: Double) -> Double {
    var sum = 1.0, term = 1.0
    let y = (x / 2) * (x / 2)
    var k = 1.0
    while true {
        term *= y / (k * k)
        sum += term
        if term < sum * 1e-16 { break }
        k += 1
        if k > 1000 { break }
    }
    return sum
}

/// FIR (Kaiser-window) causal/zero-phase filtering, mirroring k-wave-python `apply_filter`.
///
/// - Parameters:
///   - signal: input time series.
///   - fs: sampling frequency [Hz].
///   - cutoff: cut-off frequency [Hz] (single value; ignored — pass via `bandCutoff` for band-pass).
///   - type: `.lowPass`, `.highPass`, or `.bandPass`.
///   - bandCutoff: `(low, high)` cut-offs, required for `.bandPass`.
public func applyFilter(
    _ signal: [Double], fs: Double, cutoff: Double, type: FilterType,
    bandCutoff: (Double, Double)? = nil,
    zeroPhase: Bool = false, transitionWidth: Double = 0.1, stopBandAtten: Double = 60
) -> [Double] {
    if type == .bandPass {
        guard let (lo, hi) = bandCutoff else {
            preconditionFailure("bandPass requires bandCutoff (low, high)")
        }
        let lp = applyFilter(signal, fs: fs, cutoff: hi, type: .lowPass,
                             zeroPhase: zeroPhase, transitionWidth: transitionWidth,
                             stopBandAtten: stopBandAtten)
        return applyFilter(lp, fs: fs, cutoff: lo, type: .highPass,
                           zeroPhase: zeroPhase, transitionWidth: transitionWidth,
                           stopBandAtten: stopBandAtten)
    }

    let highPass = (type == .highPass)
    var cutoffF = highPass ? (fs / 2 - cutoff) : cutoff
    var sba = stopBandAtten
    if zeroPhase { sba /= 2 }

    let nOrder = Int(((sba - 7.95) / (2.285 * (transitionWidth * .pi))).rounded(.up))
    let fc = cutoffF / fs
    _ = cutoffF

    // Ideal impulse response (sinc), n = -N/2 .. N/2-1.
    var h = [Double](repeating: 0, count: nOrder)
    for i in 0..<nOrder {
        let nVal = -Double(nOrder) / 2 + Double(i)
        h[i] = 2 * fc * sincRaw(2 * .pi * fc * nVal)
    }

    // Kaiser window.
    let beta: Double
    if sba > 50 { beta = 0.1102 * (sba - 8.7) }
    else if sba >= 21 { beta = 0.5842 * pow(sba - 21, 0.4) + 0.07886 * (sba - 21) }
    else { beta = 0 }
    let i0Beta = besselI0(.pi * beta)
    var hw = [Double](repeating: 0, count: nOrder)
    for m in 0..<nOrder {
        let arg = 2 * Double(m) / Double(nOrder) - 1
        let w = besselI0(.pi * beta * (1 - arg * arg).squareRoot()) / i0Beta
        hw[m] = w * h[m]
    }
    // k-wave-python high-pass "modification" is `(-1 * ones) ** arange`, but `**` binds tighter
    // than `*`, so `ones ** arange == ones` and the whole factor collapses to a constant -1.
    // Replicate that exactly (i.e. negate every tap) for bit-parity with the reference.
    if highPass {
        for j in 0..<nOrder { hw[j] = -hw[j] }
    }

    // Zero-pad front by N, FIR filter, optional zero-phase reverse pass, then trim the N zeros.
    let L = signal.count
    var padded = [Double](repeating: 0, count: nOrder + L)
    for i in 0..<L { padded[nOrder + i] = signal[i] }
    var filtered = firFilter(hw, padded)
    if zeroPhase {
        let reversed = Array(filtered.reversed())
        filtered = Array(firFilter(hw, reversed).reversed())
    }
    return Array(filtered[nOrder...])
}

/// Causal FIR filtering: y[i] = Σ_k b[k]·x[i-k] (scipy `lfilter` with denominator 1).
private func firFilter(_ b: [Double], _ x: [Double]) -> [Double] {
    var y = [Double](repeating: 0, count: x.count)
    for i in 0..<x.count {
        var acc = 0.0
        let kMax = min(b.count - 1, i)
        for k in 0...kMax { acc += b[k] * x[i - k] }
        y[i] = acc
    }
    return y
}

/// Low-pass filter a source time series using the Kaiser windowing method, mirroring k-wave-python
/// `filter_time_series` (default `ppw = 3`, causal). Returns the filtered signal.
public func filterTimeSeries(
    grid: KWaveGrid, medium: KWaveMedium, signal: [Double],
    ppw: Double = 3, stopBandAtten: Double = 60, transitionWidth: Double = 0.1,
    zeroPhase: Bool = false
) -> [Double] {
    precondition(grid.dt > 0, "grid.dt must be set (call makeTime/setTime)")
    let fs = 1 / grid.dt
    let c0 = MLX.min(medium.soundSpeed).item(Float.self)
    // k_max_all = min over axes of max(|k_vec|) (the lowest per-axis Nyquist), not the diagonal max.
    func axisMax(_ v: MLXArray) -> Double { Double(MLX.max(MLX.abs(v)).item(Float.self)) }
    var perAxis = [axisMax(grid.kxVec)]
    if grid.dim >= 2 { perAxis.append(axisMax(grid.kyVec)) }
    if grid.dim == 3 { perAxis.append(axisMax(grid.kzVec)) }
    let kMax = perAxis.min() ?? axisMax(grid.kxVec)
    let fMax = kMax * Double(c0) / (2 * .pi)
    let cutoff = 2 * fMax / ppw
    guard ppw != 0 else { return signal }
    return applyFilter(signal, fs: fs, cutoff: cutoff, type: .lowPass,
                       zeroPhase: zeroPhase, transitionWidth: transitionWidth,
                       stopBandAtten: stopBandAtten)
}

/// 1-D fftshift (move zero-frequency component to the centre).
private func fftShift1D(_ a: MLXArray) -> MLXArray {
    let n = a.dim(0)
    return MLX.roll(a, shift: n / 2, axis: 0)
}

/// 1-D ifftshift (inverse of `fftShift1D`).
private func ifftShift1D(_ a: MLXArray) -> MLXArray {
    let n = a.dim(0)
    return MLX.roll(a, shift: (n + 1) / 2, axis: 0)
}

/// Symmetric 1-D Blackman window of length `n` (param 0.16: coeffs 0.42, 0.5, 0.08).
private func blackman1D(_ n: Int) -> [Double] {
    guard n > 1 else { return [Double](repeating: 1, count: max(n, 0)) }
    let denom = Double(n - 1)
    return (0..<n).map { i in
        let x = 2 * Double.pi * Double(i) / denom
        return 0.42 - 0.5 * cos(x) + 0.08 * cos(2 * x)
    }
}

/// 2-D rotationally-symmetric window (k-Wave `getWin(..., 'Rotation', true)`).
///
/// For even grid sizes k-Wave builds a non-symmetric window: each dimension is enlarged by 1,
/// the radial profile is taken from a symmetric 1-D window of length `L = max(enlarged)`, and the
/// result is trimmed back to `[nx, ny]`. The radial axis has unit spacing because
/// `radius = (L-1)/2` over `L` samples, so interpolation is a simple linear lookup.
private func rotationWindow2D(nx: Int, ny: Int) -> [[Double]] {
    let symX = nx % 2 != 0, symY = ny % 2 != 0
    let exX = nx + (symX ? 0 : 1)   // enlarged sizes
    let exY = ny + (symY ? 0 : 1)
    let L = max(exX, exY)
    let winLin = blackman1D(L)
    let radius = Double(L - 1) / 2.0

    func axis(_ count: Int) -> [Double] {
        guard count > 1 else { return [0] }
        return (0..<count).map { -radius + 2 * radius * Double($0) / Double(count - 1) }
    }
    let xs = axis(exX), ys = axis(exY)

    func interp(_ r: Double) -> Double {
        let rr = min(r, radius)
        let pos = rr + radius           // unit-spaced axis from -radius
        let i0 = min(Int(pos.rounded(.down)), L - 2)
        let frac = pos - Double(i0)
        return winLin[i0] * (1 - frac) + winLin[i0 + 1] * frac
    }

    // Build enlarged window, then trim to [nx, ny].
    var win = [[Double]](repeating: [Double](repeating: 0, count: ny), count: nx)
    for i in 0..<nx {
        for j in 0..<ny {
            let r = (xs[i] * xs[i] + ys[j] * ys[j]).squareRoot()
            win[i][j] = abs(interp(r))
        }
    }
    return win
}

/// Shared radial Blackman profile + axis sampling for the rotation window (any dimension).
private func rotationProfile(sizes: [Int]) -> (winLin: [Double], radius: Double, axes: [[Double]]) {
    let enlarged = sizes.map { $0 % 2 != 0 ? $0 : $0 + 1 }
    let L = enlarged.max() ?? 1
    let winLin = blackman1D(L)
    let radius = Double(L - 1) / 2.0
    let axes = enlarged.map { count -> [Double] in
        guard count > 1 else { return [0] }
        return (0..<count).map { -radius + 2 * radius * Double($0) / Double(count - 1) }
    }
    return (winLin, radius, axes)
}

private func interpRadial(_ r: Double, winLin: [Double], radius: Double) -> Double {
    let pos = min(r, radius) + radius           // unit-spaced axis from -radius
    let i0 = min(Int(pos.rounded(.down)), winLin.count - 2)
    let frac = pos - Double(i0)
    return abs(winLin[i0] * (1 - frac) + winLin[i0 + 1] * frac)
}

/// Smooth a 1D field with a Blackman window applied in k-space, rescaling to preserve the peak
/// magnitude. The 1-D companion to `smooth`/`smooth3D` (which no-op on 1-D input).
public func smooth1D(_ field: MLXArray) -> MLXArray {
    guard field.ndim == 1 else { return field }
    let nx = field.dim(0)
    let (winLin, radius, axes) = rotationProfile(sizes: [nx])
    let xs = axes[0]

    let sx = (nx + 1) / 2
    var flat = [Float](repeating: 0, count: nx)
    for i in 0..<nx {
        let w = Float(interpRadial(abs(xs[i]), winLin: winLin, radius: radius))
        flat[(i + sx) % nx] = w   // ifftshift inline (peak-at-centre → index-0).
    }
    let winA = MLXArray(flat).asType(.complex64)
    let smoothed = MLXFFT.ifft(winA * MLXFFT.fft(field.asType(.complex64))).realPart()

    let srcMax = MLX.max(MLX.abs(field)).item(Float.self)
    let outMax = MLX.max(MLX.abs(smoothed)).item(Float.self)
    let scale = outMax > 0 ? srcMax / outMax : 1
    return smoothed * scale
}

/// Smooth a 3D field with a rotationally-symmetric Blackman window (k-Wave `smooth`, restore_max).
public func smooth3D(_ field: MLXArray) -> MLXArray {
    guard field.ndim == 3 else { return field }
    let nx = field.dim(0), ny = field.dim(1), nz = field.dim(2)
    let (winLin, radius, axes) = rotationProfile(sizes: [nx, ny, nz])
    let xs = axes[0], ys = axes[1], zs = axes[2]

    let sx = (nx + 1) / 2, sy = (ny + 1) / 2, sz = (nz + 1) / 2
    var flat = [Float](repeating: 0, count: nx * ny * nz)
    for i in 0..<nx {
        for j in 0..<ny {
            for k in 0..<nz {
                let r = (xs[i] * xs[i] + ys[j] * ys[j] + zs[k] * zs[k]).squareRoot()
                let w = Float(interpRadial(r, winLin: winLin, radius: radius))
                // ifftshift inline (peak-at-centre → index-0).
                let di = (i + sx) % nx, dj = (j + sy) % ny, dk = (k + sz) % nz
                flat[(di * ny + dj) * nz + dk] = w
            }
        }
    }
    let winA = MLXArray(flat).reshaped([nx, ny, nz]).asType(.complex64)
    let smoothed = MLXFFT.ifftn(winA * MLXFFT.fftn(field.asType(.complex64))).realPart()

    let srcMax = MLX.max(MLX.abs(field)).item(Float.self)
    let outMax = MLX.max(MLX.abs(smoothed)).item(Float.self)
    let scale = outMax > 0 ? srcMax / outMax : 1
    return smoothed * scale
}

/// Smooth a 2D field with a rotationally-symmetric Blackman window applied in k-space, rescaling
/// to preserve the peak magnitude. Matches k-Wave / k-wave-python `smooth(a, restore_max=True)`.
public func smooth(_ field: MLXArray) -> MLXArray {
    guard field.ndim == 2 else { return field }
    let nx = field.dim(0), ny = field.dim(1)
    let win = rotationWindow2D(nx: nx, ny: ny)

    // ifftshift the window (peak-at-centre → index-0) and flatten in row-major order.
    let sx = (nx + 1) / 2, sy = (ny + 1) / 2
    var flat = [Float](repeating: 0, count: nx * ny)
    for i in 0..<nx {
        for j in 0..<ny {
            flat[i * ny + j] = Float(win[(i + sx) % nx][(j + sy) % ny])
        }
    }
    let winA = MLXArray(flat).reshaped([nx, ny]).asType(.complex64)

    let spectrum = MLXFFT.fft2(field.asType(.complex64))
    let smoothed = MLXFFT.ifft2(winA * spectrum).realPart()

    let srcMax = MLX.max(MLX.abs(field)).item(Float.self)
    let outMax = MLX.max(MLX.abs(smoothed)).item(Float.self)
    let scale = outMax > 0 ? srcMax / outMax : 1
    return smoothed * scale
}

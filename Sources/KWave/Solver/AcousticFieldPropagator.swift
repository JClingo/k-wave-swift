import Foundation
import MLX
import MLXFFT

/// Largest prime factor (1 for n ≤ 1).
private func largestPrimeFactor(_ n: Int) -> Int {
    var m = n, best = 1, d = 2
    while d * d <= m {
        while m % d == 0 { best = max(best, d); m /= d }
        d += 1
    }
    return max(best, m > 1 ? m : 1)
}

/// Smallest grid size in `[n, n + searchRange]` with the smallest largest-prime-factor
/// (MATLAB `getOptimalGridSize` inside acousticFieldPropagator).
private func optimalGridDim(_ n: Int, searchRange: Int) -> Int {
    var bestN = n, bestFac = Int.max
    for i in 0...searchRange where largestPrimeFactor(n + i) < bestFac {
        bestFac = largestPrimeFactor(n + i)
        bestN = n + i
    }
    return bestN
}

/// Exact single-frequency (CW) field propagation in a lossless homogeneous medium, a 1:1 port of
/// MATLAB k-Wave `acousticFieldPropagator` (Treeby, Budisky, Wise, Jaros & Cox). The source plane
/// is given as amplitude and phase; the steady-state complex field is computed with a single
/// n-dimensional FFT using the exact Green's-function propagator, evaluated at time `t` (long
/// enough by default for the field to establish over the whole grid).
///
/// The propagator is built in double precision on the host (the rational expression suffers
/// catastrophic cancellation near its singular wavenumbers in single precision); the FFTs run in
/// MLX complex64.
///
/// - Parameters:
///   - ampIn: source amplitude over the grid (`[Nx]`, `[Nx, Ny]`, or `[Nx, Ny, Nz]`) [Pa].
///   - phaseIn: source phase [rad] — a single value or one per grid point.
///   - dx: isotropic grid spacing [m]; f0: source frequency [Hz]; c0: sound speed [m/s].
///   - time: evaluation time [s]; `nil` selects `timeExpansionFactor·dx·√(Σ N²)/c0`.
///   - expandedGridSize: FFT grid size; `nil` expands by `gridExpansionFactor·t·c0/dx` points and
///     rounds each dimension up to a small-prime-factor size within `gridSearchRange`.
///   - useRamp: use the propagator derived for a cosine-ramped source onset (k-Wave default);
///     `false` uses the abrupt-onset variant.
/// - Returns: complex `pressure`, plus its `amp` and `phase`, each shaped like `ampIn`.
public func acousticFieldPropagator(
    ampIn: MLXArray, phaseIn: MLXArray, dx: Double, f0: Double, c0: Double,
    time: Double? = nil, timeExpansionFactor: Double = 1.5,
    gridExpansionFactor: Double = 1.1, gridSearchRange: Int = 50,
    expandedGridSize: [Int]? = nil, useRamp: Bool = true
) -> (pressure: MLXArray, amp: MLXArray, phase: MLXArray) {
    let sz = ampIn.shape
    let ndim = sz.count
    precondition((1...3).contains(ndim), "ampIn must be 1, 2, or 3 dimensional")
    precondition(phaseIn.size == 1 || phaseIn.shape == sz,
                 "phaseIn must be scalar or match ampIn")
    let w0 = 2 * Double.pi * f0
    precondition(w0 / c0 <= .pi / dx + 1e-12,
                 "frequency exceeds the maximum supported c0/(2·dx)")

    let t = time ?? (timeExpansionFactor * dx
        * Double(sz.reduce(0) { $0 + $1 * $1 }).squareRoot() / c0)

    let szEx: [Int]
    if let expandedGridSize {
        precondition(expandedGridSize.count == ndim, "expandedGridSize must match ampIn rank")
        szEx = expandedGridSize
    } else {
        let expansion = Int(ceil(gridExpansionFactor * t * c0 / dx))
        szEx = sz.map { optimalGridDim($0 + expansion, searchRange: gridSearchRange) }
    }

    // |k| over the expanded grid in FFT-natural order, in double precision on the host.
    let axes = szEx.map { wavenumberVector($0, d: dx) }
    let total = szEx.reduce(1, *)
    var kMag = [Double](repeating: 0, count: total)
    var kMax = 0.0
    for flat in 0..<total {
        var rem = flat, sq = 0.0
        for axis in stride(from: ndim - 1, through: 0, by: -1) {
            let i = rem % szEx[axis]
            rem /= szEx[axis]
            sq += axes[axis][i] * axes[axis][i]
        }
        let k = sq.squareRoot()
        kMag[flat] = k
        kMax = max(kMax, k)
    }

    // Propagator per bin (complex, double). Singular wavenumbers are patched with their limits.
    let eps = kMax * 2.220446049250313e-16 * 10          // MATLAB eps·10 scaled by max(k).
    let (cwt, swt) = (cos(w0 * t), sin(w0 * t))          // e^{i·w0·t} parts.
    var propRe = [Float](repeating: 0, count: total)
    var propIm = [Float](repeating: 0, count: total)
    for flat in 0..<total {
        let k = kMag[flat]
        var re: Double, im: Double
        if k == 0 {
            if useRamp {
                // −i·(1 + 3·e^{iw0t}) / (3w0) = (3·sin(w0t) − i·(1 + 3·cos(w0t))) / (3w0)
                re = swt / w0
                im = -(1 + 3 * cwt) / (3 * w0)
            } else {
                // (i − i·e^{iw0t}) / w0 = (swt + i(1 − cwt)) / w0
                re = swt / w0
                im = (1 - cwt) / w0
            }
        } else if useRamp {
            if abs(k - w0 / c0) < eps {
                // (−i − 15·e^{2iw0t}·(i + 2π − 2w0t)) / (e^{iw0t}·60w0)
                let a = 2 * Double.pi - 2 * w0 * t
                let (n2Re, n2Im) = (cos(2 * w0 * t), sin(2 * w0 * t))
                let numRe = -15 * (n2Re * a - n2Im)
                let numIm = -1 - 15 * (n2Re + n2Im * a)
                // divide by e^{iw0t}·60w0 → multiply by e^{−iw0t}/(60w0).
                re = (numRe * cwt + numIm * swt) / (60 * w0)
                im = (numIm * cwt - numRe * swt) / (60 * w0)
            } else if abs(k - w0 / (2 * c0)) < eps {
                // −(16i·e^{iw0t} + 3π·e^{iw0t/2}) / (12w0)
                let (hRe, hIm) = (cos(w0 * t / 2), sin(w0 * t / 2))
                re = -(-16 * swt + 3 * .pi * hRe) / (12 * w0)
                im = -(16 * cwt + 3 * .pi * hIm) / (12 * w0)
            } else if abs(k - 3 * w0 / (2 * c0)) < eps {
                // (16i·e^{iw0t} − 5π·e^{3iw0t/2}) / (20w0)
                let (qRe, qIm) = (cos(3 * w0 * t / 2), sin(3 * w0 * t / 2))
                re = (-16 * swt - 5 * .pi * qRe) / (20 * w0)
                im = (16 * cwt - 5 * .pi * qIm) / (20 * w0)
            } else {
                let ck = c0 * k
                let ck2 = ck * ck, w2 = w0 * w0
                let cosA = cos(ck * t), cosB = cos(ck * (t - 2 * .pi / w0))
                let sinA = sin(ck * t), sinB = sin(ck * (t - 2 * .pi / w0))
                let den = -32 * pow(ck, 6) + 112 * pow(ck, 4) * w2
                    - 98 * ck2 * w2 * w2 + 18 * pow(w0, 6)
                // −2i·e^{iw0t}·w0·(16c⁴k⁴ − 40c²k²w0² + 9w0⁴)
                let g = w0 * (16 * ck2 * ck2 - 40 * ck2 * w2 + 9 * w2 * w2)
                var numRe = 2 * g * swt
                var numIm = -2 * g * cwt
                // −3i·w0³·(4c²k² + w0²)·(cos A + cos B)
                let h = 3 * pow(w0, 3) * (4 * ck2 + w2)
                numIm -= h * (cosA + cosB)
                // + ck·w0²·(4c²k² + 11w0²)·(sin A + sin B)
                numRe += ck * w2 * (4 * ck2 + 11 * w2) * (sinA + sinB)
                re = numRe / den
                im = numIm / den
            }
        } else {
            if abs(k - w0 / c0) < eps {
                // (w0t·e^{iw0t} + sin(w0t)) / (2w0)
                re = (w0 * t * cwt + swt) / (2 * w0)
                im = w0 * t * swt / (2 * w0)
            } else {
                let ck = c0 * k
                let den = pow(ck, 3) - ck * w0 * w0
                // (i·w0·ck·(e^{iw0t} − cos(ckt)) + w0²·sin(ckt)) / den + sin(ckt)/ck
                let dRe = cwt - cos(ck * t)
                re = (-w0 * ck * swt + w0 * w0 * sin(ck * t)) / den + sin(ck * t) / ck
                im = w0 * ck * dRe / den
            }
        }
        propRe[flat] = Float(re)
        propIm[flat] = Float(im)
    }
    let propagator = MLXArray(propRe).reshaped(szEx).asType(.complex64)
        + imagUnit() * MLXArray(propIm).reshaped(szEx).asType(.complex64)

    // Zero-pad the complex source amp·e^{iφ} into the corner of the expanded grid.
    let phase = phaseIn.size == 1
        ? MLXArray.ones(sz, dtype: .float32) * phaseIn.asType(.float32)
        : phaseIn.asType(.float32)
    let amp = ampIn.asType(.float32)
    let srcComplex = amp.asType(.complex64) * (MLX.cos(phase).asType(.complex64)
        + imagUnit() * MLX.sin(phase).asType(.complex64))
    let padded = MLX.padded(srcComplex,
                            widths: sz.enumerated().map { .init((0, szEx[$0.offset] - $0.element)) })

    var pressure = MLXFFT.ifftn(propagator * MLXFFT.fftn(padded))
    switch ndim {
    case 1: pressure = pressure[0..<sz[0]]
    case 2: pressure = pressure[0..<sz[0], 0..<sz[1]]
    default: pressure = pressure[0..<sz[0], 0..<sz[1], 0..<sz[2]]
    }
    pressure = pressure * Float(2 * c0 / dx)

    let outRe = pressure.realPart(), outIm = pressure.imaginaryPart()
    let ampOut = MLX.sqrt(outRe * outRe + outIm * outIm)
    let phaseOut = MLX.atan2(outIm, outRe)
    return (pressure, ampOut, phaseOut)
}

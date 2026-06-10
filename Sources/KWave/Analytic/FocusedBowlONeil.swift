import Foundation
import MLX

/// Double-precision complex on-axis pressure of a focused bowl (O'Neil), at `axial` positions.
/// Returns interleaved-free `(re, im)`. Shared by `focusedBowlONeil` and `focusedAnnulusONeil`.
private func bowlAxialComplex(
    radius: Double, diameter: Double, velocity: Double, k: Double, soundSpeed: Double, density: Double,
    axial: [Double]
) -> (re: [Double], im: [Double]) {
    let h = radius - (radius * radius - (diameter / 2) * (diameter / 2)).squareRoot()
    let rcv = density * soundSpeed * velocity
    let eps = Double.ulpOfOne
    var re = [Double](repeating: 0, count: axial.count)
    var im = [Double](repeating: 0, count: axial.count)
    for (i, x) in axial.enumerated() {
        let B = ((x - h) * (x - h) + (diameter / 2) * (diameter / 2)).squareRoot()
        let M = (B + x) / 2
        let P = abs(x - radius) < eps ? k * h : 2 / (1 - x / radius) * sin(k * (B - x) / 2)
        re[i] = rcv * P * sin(k * M)        // ρcv·P·i·exp(-ikM) = ρcv·P·(sin kM + i·cos kM)
        im[i] = rcv * P * cos(k * M)
    }
    return (re, im)
}

/// O'Neil's analytic solution for the steady-state pressure amplitude radiated by a focused bowl
/// transducer, mirroring k-wave-python `focused_bowl_oneil`. Used as an analytic reference for
/// validating focused-transducer simulations.
///
/// - Parameters:
///   - radius: radius of curvature [m]; diameter: aperture diameter [m].
///   - velocity: normal surface velocity [m/s]; frequency: drive frequency [Hz].
///   - soundSpeed: medium sound speed [m/s]; density: medium density [kg/m³].
///   - axialPositions: positions along the beam axis (0 = transducer surface) [m]; `nil` to skip.
///   - lateralPositions: lateral positions through the geometric focus (0 = axis) [m]; `nil` to skip.
/// - Returns: `axial` magnitude [Pa], `lateral` magnitude [Pa], and complex axial pressure (t=0).
public func focusedBowlONeil(
    radius: Double, diameter: Double, velocity: Double, frequency: Double,
    soundSpeed: Double, density: Double,
    axialPositions: [Double]? = nil, lateralPositions: [Double]? = nil
) -> (axial: MLXArray?, lateral: MLXArray?, axialComplex: MLXArray?) {
    let k = 2 * Double.pi * frequency / soundSpeed          // wavenumber.
    let h = radius - (radius * radius - (diameter / 2) * (diameter / 2)).squareRoot()  // rim height.
    let rcv = density * soundSpeed * velocity

    var axial: MLXArray?
    var axialComplex: MLXArray?
    if let x = axialPositions {
        let (re, im) = bowlAxialComplex(radius: radius, diameter: diameter, velocity: velocity,
                                        k: k, soundSpeed: soundSpeed, density: density, axial: x)
        var magHost = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count { magHost[i] = Float((re[i] * re[i] + im[i] * im[i]).squareRoot()) }
        axial = MLXArray(magHost)
        let reA = MLXArray(re.map { Float($0) }).asType(.complex64)
        let imA = MLXArray(im.map { Float($0) }).asType(.complex64)
        axialComplex = reA + imA * imagUnit()
    }

    var lateral: MLXArray?
    if let y = lateralPositions {
        var lat = [Float](repeating: 0, count: y.count)
        for (i, yi) in y.enumerated() {
            if yi == 0 {
                lat[i] = Float(rcv * k * h)
            } else {
                let Z = k * yi * diameter / (2 * radius)
                lat[i] = Float(2 * rcv * k * h * j1(Z) / Z)   // J₁ Bessel.
            }
        }
        lateral = MLXArray(lat)
    }

    return (axial, lateral, axialComplex)
}

/// O'Neil's analytic on-axis pressure for a focused *annular* array transducer, mirroring
/// k-wave-python `focused_annulus_oneil`. Each annular element is the difference of two bowl fields
/// (outer minus inner aperture), phase-shifted, summed; the returned amplitude is `|Σ p_el|`.
///
/// - Parameters:
///   - radius: shared radius of curvature [m].
///   - diameters: `[2, numElements]` — row 0 inner, row 1 outer aperture diameter [m].
///   - amplitude: per-element normal surface velocity [m/s]; phase: per-element phase [rad].
///   - axialPositions: positions along the beam axis (0 = transducer surface) [m].
/// - Returns: axial pressure amplitude `[numAxial]` [Pa].
public func focusedAnnulusONeil(
    radius: Double, diameters: MLXArray, amplitude: [Double], phase: [Double],
    frequency: Double, soundSpeed: Double, density: Double, axialPositions: [Double]
) -> MLXArray {
    precondition(diameters.ndim == 2 && diameters.dim(0) == 2, "diameters must be [2, numElements]")
    let n = diameters.dim(1)
    precondition(amplitude.count == n && phase.count == n, "amplitude/phase must match numElements")
    let dh = diameters.reshaped([2 * n]).asArray(Float.self)
    func dia(_ row: Int, _ el: Int) -> Double { Double(dh[row * n + el]) }

    let k = 2 * Double.pi * frequency / soundSpeed
    let m = axialPositions.count
    var sumRe = [Double](repeating: 0, count: m)
    var sumIm = [Double](repeating: 0, count: m)

    for el in 0..<n {
        let outer = bowlAxialComplex(radius: radius, diameter: dia(1, el), velocity: amplitude[el],
                                     k: k, soundSpeed: soundSpeed, density: density, axial: axialPositions)
        let inner: (re: [Double], im: [Double]) = dia(0, el) == 0
            ? (re: [Double](repeating: 0, count: m), im: [Double](repeating: 0, count: m))
            : bowlAxialComplex(radius: radius, diameter: dia(0, el), velocity: amplitude[el],
                               k: k, soundSpeed: soundSpeed, density: density, axial: axialPositions)
        for i in 0..<m {
            let re = outer.re[i] - inner.re[i], im = outer.im[i] - inner.im[i]
            let mag = (re * re + im * im).squareRoot()
            let ang = atan2(im, re) + phase[el]      // |p_el|·exp(i(angle(p_el)+phase))
            sumRe[i] += mag * cos(ang)
            sumIm[i] += mag * sin(ang)
        }
    }
    return MLXArray((0..<m).map { Float((sumRe[$0] * sumRe[$0] + sumIm[$0] * sumIm[$0]).squareRoot()) })
}

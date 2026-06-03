import Foundation
import MLX

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
    let eps = Double.ulpOfOne

    var axial: MLXArray?
    var axialComplex: MLXArray?
    if let x = axialPositions {
        var mag = [Float](repeating: 0, count: x.count)
        var cplx = [Float](repeating: 0, count: 2 * x.count)   // interleaved re, im.
        for (i, xi) in x.enumerated() {
            let B = ((xi - h) * (xi - h) + (diameter / 2) * (diameter / 2)).squareRoot()
            let d = B - xi
            let M = (B + xi) / 2
            let P = abs(xi - radius) < eps ? k * h : 2 / (1 - xi / radius) * sin(k * d / 2)
            mag[i] = Float(rcv * abs(P))
            // complex = ρcv·P·i·exp(-ikM) = ρcv·P·(sin(kM) + i·cos(kM))
            cplx[2 * i] = Float(rcv * P * sin(k * M))
            cplx[2 * i + 1] = Float(rcv * P * cos(k * M))
        }
        axial = MLXArray(mag)
        let re = MLXArray(stride(from: 0, to: 2 * x.count, by: 2).map { cplx[$0] })
        let im = MLXArray(stride(from: 1, to: 2 * x.count, by: 2).map { cplx[$0] })
        axialComplex = re.asType(.complex64) + im.asType(.complex64) * imagUnit()
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

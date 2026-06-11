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

// MARK: - fitPowerLawParams

/// Faithful port of scipy `fmin` (Nelder-Mead, non-adaptive: ρ=1, χ=2, ψ=0.5, σ=0.5).
/// Initial simplex: x0 plus a 5% per-component perturbation (0.00025 where the component is 0).
/// Terminates when the simplex spread is below `xatol` AND the f-value spread is below `fatol`,
/// or after `maxiter` iterations / `maxfun` evaluations.
private func nelderMead(
    _ f: ([Double]) -> Double, x0: [Double],
    xatol: Double = 1e-4, fatol: Double = 1e-4
) -> [Double] {
    let n = x0.count
    let maxiter = n * 200, maxfun = n * 200
    let rho = 1.0, chi = 2.0, psi = 0.5, sigma = 0.5
    var fcalls = 0
    func eval(_ x: [Double]) -> Double { fcalls += 1; return f(x) }

    var sim = [x0]
    for k in 0..<n {
        var y = x0
        y[k] = y[k] != 0 ? y[k] * 1.05 : 0.00025
        sim.append(y)
    }
    var fsim = sim.map(eval)

    func sortSimplex() {
        let order = (0...n).sorted { fsim[$0] < fsim[$1] }
        sim = order.map { sim[$0] }
        fsim = order.map { fsim[$0] }
    }
    sortSimplex()

    var iterations = 1
    while iterations < maxiter && fcalls < maxfun {
        let xSpread = (1...n).map { j in (0..<n).map { abs(sim[j][$0] - sim[0][$0]) }.max()! }.max()!
        let fSpread = (1...n).map { abs(fsim[$0] - fsim[0]) }.max()!
        if xSpread <= xatol && fSpread <= fatol { break }

        // Centroid of all but the worst point.
        var xbar = [Double](repeating: 0, count: n)
        for j in 0..<n { for k in 0..<n { xbar[k] += sim[j][k] / Double(n) } }
        func blend(_ a: Double, _ b: Double) -> [Double] {
            (0..<n).map { a * xbar[$0] + b * sim[n][$0] }
        }

        let xr = blend(1 + rho, -rho)
        let fxr = eval(xr)
        if fxr < fsim[0] {
            let xe = blend(1 + rho * chi, -rho * chi)
            let fxe = eval(xe)
            if fxe < fxr { sim[n] = xe; fsim[n] = fxe } else { sim[n] = xr; fsim[n] = fxr }
        } else if fxr < fsim[n - 1] {
            sim[n] = xr; fsim[n] = fxr
        } else {
            var shrink = false
            if fxr < fsim[n] {
                let xc = blend(1 + psi * rho, -psi * rho)        // outside contraction
                let fxc = eval(xc)
                if fxc <= fxr { sim[n] = xc; fsim[n] = fxc } else { shrink = true }
            } else {
                let xcc = blend(1 - psi, psi)                    // inside contraction
                let fxcc = eval(xcc)
                if fxcc < fsim[n] { sim[n] = xcc; fsim[n] = fxcc } else { shrink = true }
            }
            if shrink {
                for j in 1...n {
                    sim[j] = (0..<n).map { sim[0][$0] + sigma * (sim[j][$0] - sim[0][$0]) }
                    fsim[j] = eval(sim[j])
                }
            }
        }
        sortSimplex()
        iterations += 1
    }
    return sim[0]
}

/// Calculate the absorption parameters (`alphaCoeff`, `alphaPower`) that should be passed to the
/// simulation functions so the *actual* absorption of the fractional-Laplacian wave equation follows
/// the desired power law `a = a0·f^y` over `fMin`…`fMax`. Mirrors k-wave-python
/// `fit_power_law_params` (200 linearly spaced frequencies, scipy-`fmin` Nelder-Mead fit).
///
/// - Parameters:
///   - a0: desired power-law coefficient [dB/(MHz^y cm)]; y: desired exponent.
///   - c0: sound speed [m/s]; fMin/fMax: frequency range of the fit [Hz].
/// - Returns: fitted `(a0, y)` to use as `medium.alphaCoeff` / `medium.alphaPower`.
public func fitPowerLawParams(
    a0: Double, y: Double, c0: Double, fMin: Double, fMax: Double
) -> (a0: Double, y: Double) {
    let nPoints = 200
    let angularFrequencyScale = 2.0 * Double.pi
    let frequencyStep = (fMax - fMin) / Double(nPoints - 1)
    var w = [Double]()
    w.reserveCapacity(nPoints)
    for i in 0..<nPoints {
        let frequency = fMin + frequencyStep * Double(i)
        w.append(angularFrequencyScale * frequency)
    }
    let a0Np = db2neper(a0, y: y)
    let desired = w.map { a0Np * pow($0, y) }

    func absorptionError(_ trial: [Double]) -> Double {
        let (a0t, yt) = (trial[0], trial[1])
        let tanTerm = tan(Double.pi * yt / 2)
        var sum = 0.0
        for (i, wi) in w.enumerated() {
            let actual = a0t * pow(wi, yt) / (1 - (yt + 1) * a0t * c0 * tanTerm * pow(wi, yt - 1))
            let d = desired[i] - actual
            sum += d * d
        }
        return sum.squareRoot()
    }

    let fit = nelderMead(absorptionError, x0: [a0Np, y])
    return (neper2db(fit[0], y: fit[1]), fit[1])
}

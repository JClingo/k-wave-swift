import Foundation
import MLX

/// Build a 1-D PML absorption profile of length `n` (values in (0, 1]; 1 in the interior),
/// matching MATLAB k-Wave's `getPML`.
///
/// - Parameters:
///   - n: axis length.
///   - dx: grid spacing.
///   - dt: time step.
///   - c: reference sound speed.
///   - pmlSize: PML thickness in grid points.
///   - pmlAlpha: PML absorption coefficient.
///   - staggered: true for the staggered (half-grid) variant used by velocity updates.
///   - axisymmetric: skip the left-side ramp (the radial axis of the axisymmetric solver has a
///     PML at the outer edge only).
func pmlProfile(
    n: Int, dx: Double, dt: Double, c: Double,
    pmlSize: Int, pmlAlpha: Double, staggered: Bool, axisymmetric: Bool = false
) -> [Double] {
    // Match k-wave-python `get_pml`: left and right profiles are computed independently (the
    // staggered grid is shifted +half a point in the + direction, so the sides are NOT mirrors).
    let power = 4.0
    let a = pmlAlpha * (c / dx)
    let m = Double(pmlSize)
    var pml = [Double](repeating: 1.0, count: n)
    for k in 0..<min(pmlSize, n) {
        let x = Double(k + 1)                                  // x = 1..pml_size
        let posLeft: Double, posRight: Double
        if staggered {
            posLeft = (m + 0.5 - x) / m                        // ((x+0.5)-pml-1)/(0-pml)
            posRight = (x + 0.5) / m
        } else {
            posLeft = (m + 1 - x) / m                          // (x-pml-1)/(0-pml)
            posRight = x / m
        }
        if !axisymmetric {
            pml[k] = exp(-a * dt / 2.0 * pow(posLeft, power))  // left, indices 0..pml-1
        }
        pml[n - pmlSize + k] = exp(-a * dt / 2.0 * pow(posRight, power)) // right
    }
    return pml
}

/// Optimal PML size per dimension: prefer a thickness whose padded grid (`n + 2*pmlSize`) has the
/// smallest largest-prime-factor, which gives the most efficient FFT size. Mirrors MATLAB k-Wave
/// `getOptimalPMLSize`.
public func getOptimalPMLSize(_ gridSize: [Int], pmlRange: ClosedRange<Int> = 10...60) -> [Int] {
    gridSize.map { optimalPMLForDim($0, range: pmlRange) }
}

/// Largest prime factor of `n` (1 for n <= 1). Smaller is better for FFT efficiency.
private func maxPrimeFactor(_ n: Int) -> Int {
    guard n > 1 else { return 1 }
    var maxFactor = 1
    var m = n
    var d = 2
    while d * d <= m {
        while m % d == 0 {
            maxFactor = max(maxFactor, d)
            m /= d
        }
        d += 1
    }
    if m > 1 { maxFactor = max(maxFactor, m) }
    return maxFactor
}

private func optimalPMLForDim(_ n: Int, range: ClosedRange<Int>) -> Int {
    var bestPML = range.lowerBound
    var bestScore = Int.max
    for pmlSize in range {
        let score = maxPrimeFactor(n + 2 * pmlSize)
        if score < bestScore {
            bestScore = score
            bestPML = pmlSize
        }
    }
    return bestPML
}

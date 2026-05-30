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
func pmlProfile(
    n: Int, dx: Double, dt: Double, c: Double,
    pmlSize: Int, pmlAlpha: Double, staggered: Bool
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
        pml[k] = exp(-a * dt / 2.0 * pow(posLeft, power))      // left, indices 0..pml-1
        pml[n - pmlSize + k] = exp(-a * dt / 2.0 * pow(posRight, power)) // right
    }
    return pml
}

/// Heuristic optimal PML size: prefer a thickness whose padded grid has small prime factors.
/// Phase-1 placeholder — returns the requested default.
public func getOptimalPMLSize(_ gridSize: [Int], defaultSize: Int = 20) -> Int {
    defaultSize
}

import Foundation
import MLX

/// Exact band-limited interpolation of an off-grid delta on the centered, periodic grid axis,
/// mirroring k-wave-python `get_delta_bli` (a periodic sinc / Dirichlet kernel, with a Nyquist
/// sinusoid added for even `n` to preserve conjugate symmetry).
private func deltaBLI(_ n: Int, _ d: Double, _ x0: Double) -> [Float] {
    let coords = centeredAxis(n, d)
    let isEven = n % 2 == 0
    let nD = Double(n)
    return coords.map { x -> Float in
        let dx = x - x0
        var f: Double
        if dx == 0 {
            f = 1
        } else if isEven {
            f = sin(.pi * dx / d) / (nD * tan(.pi * dx / (nD * d)))
        } else {
            f = sin(.pi * dx / d) / (nD * sin(.pi * dx / (nD * d)))
        }
        if isEven {
            f -= sin(.pi * x0 / d) / nD * sin(.pi * x / d)   // Nyquist correction.
        }
        return Float(f)
    }
}

/// Distribute off-grid source points onto the grid using a band-limited (sinc) interpolant,
/// mirroring k-wave-python `off_grid_points` with `bli_type="exact"` (the
/// exact, full-grid evaluation). Each point contributes a separable sinc, scaled and summed.
///
/// This is the off-grid spreading primitive underlying `kWaveArray` source/sensor weighting.
///
/// - Parameters:
///   - points: off-grid coordinates `[dim, numPoints]` [m] (centered-axis convention).
///   - scale: per-point weight; `nil` = all ones, or a single value applied to all points.
/// - Returns: grid-shaped source mask (`[Nx]`, `[Nx, Ny]`, or `[Nx, Ny, Nz]`).
public func offGridPoints(grid: KWaveGrid, points: MLXArray, scale: [Double]? = nil) -> MLXArray {
    precondition(points.ndim == 2 && points.dim(0) == grid.dim,
                 "points must be [dim, numPoints]")
    let dim = grid.dim
    let nPts = points.dim(1)
    let host = points.reshaped([dim * nPts]).asArray(Float.self)
    func coord(_ axis: Int, _ p: Int) -> Double { Double(host[axis * nPts + p]) }

    let weights: [Double]
    if let scale {
        precondition(scale.count == 1 || scale.count == nPts, "scale must be scalar or per-point")
        weights = scale.count == 1 ? [Double](repeating: scale[0], count: nPts) : scale
    } else {
        weights = [Double](repeating: 1, count: nPts)
    }

    let shape = grid.size
    var mask = MLXArray.zeros(shape, dtype: .float32)
    for p in 0..<nPts {
        let s = Float(weights[p])
        let vx = MLXArray(deltaBLI(grid.nx, grid.dx, coord(0, p)))
        switch dim {
        case 1:
            mask = mask + s * vx
        case 2:
            let vy = MLXArray(deltaBLI(grid.ny, grid.dy, coord(1, p)))
            mask = mask + s * (vx.reshaped([grid.nx, 1]) * vy.reshaped([1, grid.ny]))
        default:
            let vy = MLXArray(deltaBLI(grid.ny, grid.dy, coord(1, p)))
            let vz = MLXArray(deltaBLI(grid.nz, grid.dz, coord(2, p)))
            mask = mask + s * (vx.reshaped([grid.nx, 1, 1])
                               * vy.reshaped([1, grid.ny, 1]) * vz.reshaped([1, 1, grid.nz]))
        }
    }
    return mask
}

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

/// Unnormalized sinc `sin(x)/x` (==1 at 0), matching k-wave-python `sinc` (= `np.sinc(x/π)`).
private func sincU(_ x: Double) -> Double { x == 0 ? 1 : sin(x) / x }

/// Grid-node offsets of the truncated-BLI "star" around a point, ported from k-Wave MATLAB
/// `tolStar.m`: all per-axis offsets in `[-d, d]` (with `d = ceil(1/(π·tol))`) whose product of
/// magnitudes is `≤ d`; collapsed to the zero-offset plane in any axis where the point lies on a
/// grid node (the sinc is exactly zero at the other nodes); shifted to the nearest node and
/// clipped to the grid.
private func tolStarNodes(
    tolerance: Double, grid: KWaveGrid, point: [Double]
) -> [[Int]] {
    let dim = grid.dim
    let decay = Int(ceil(1 / (.pi * tolerance)))
    let onGridThreshold = grid.dx * 1e-3
    let dims = grid.size
    let spacing = grid.spacing

    // Per-axis nearest node and on-grid flag.
    var closest = [Int](repeating: 0, count: dim)
    var onGrid = [Bool](repeating: false, count: dim)
    for axis in 0..<dim {
        let axisCoords = centeredAxis(dims[axis], spacing[axis])
        var best = 0, bestD = Double.greatestFiniteMagnitude
        for (i, c) in axisCoords.enumerated() where abs(c - point[axis]) < bestD {
            bestD = abs(c - point[axis]); best = i
        }
        closest[axis] = best
        onGrid[axis] = bestD < onGridThreshold
    }

    // Canonical star offsets, with on-grid axes collapsed to offset 0.
    let axisRange = (-decay...decay).map { $0 }
    var offsets: [[Int]] = [[]]
    for axis in 0..<dim {
        let choices = onGrid[axis] ? [0] : axisRange
        offsets = offsets.flatMap { prefix in choices.map { prefix + [$0] } }
    }
    return offsets.compactMap { off -> [Int]? in
        // Star shape: |i·j·k| ≤ decay (no filter in 1D; zero offsets always pass).
        guard off.count == 1 || abs(off.reduce(1, *)) <= decay else { return nil }
        var node = [Int](repeating: 0, count: dim)
        for axis in 0..<dim {
            let idx = closest[axis] + off[axis]
            guard idx >= 0 && idx < dims[axis] else { return nil }
            node[axis] = idx
        }
        return node
    }
}

/// Distribute off-grid source points onto the grid using a band-limited (sinc) interpolant,
/// mirroring k-wave-python `off_grid_points`. With `bliTolerance == 0` this is the exact periodic
/// BLI (`bli_type="exact"`, full-grid evaluation); with `bliTolerance > 0` (k-Wave's default is
/// 0.1) each point contributes a plain truncated sinc evaluated only on its `tolStar` neighbour
/// nodes — much sparser, and the path `kWaveArray` uses.
///
/// This is the off-grid spreading primitive underlying `kWaveArray` source/sensor weighting.
///
/// - Parameters:
///   - points: off-grid coordinates `[dim, numPoints]` [m] (centered-axis convention).
///   - scale: per-point weight; `nil` = all ones, or a single value applied to all points.
///   - bliTolerance: relative BLI truncation tolerance in (0, 1), or 0 for the exact BLI.
/// - Returns: grid-shaped source mask (`[Nx]`, `[Nx, Ny]`, or `[Nx, Ny, Nz]`).
public func offGridPoints(grid: KWaveGrid, points: MLXArray, scale: [Double]? = nil,
                          bliTolerance: Double = 0) -> MLXArray {
    precondition(bliTolerance >= 0 && bliTolerance < 1, "bliTolerance must be in [0, 1)")
    if bliTolerance > 0 {
        return offGridPointsTruncated(grid: grid, points: points, scale: scale,
                                      tolerance: bliTolerance)
    }
    return offGridPointsExact(grid: grid, points: points, scale: scale)
}

/// Binary mask of the tolStar neighbour nodes of a set of off-grid points (k-wave-python
/// `off_grid_points(..., mask_only=True)` with a nonzero tolerance). Marks every star node,
/// including sinc zero-crossings.
func offGridPointsMask(grid: KWaveGrid, points: MLXArray, bliTolerance: Double) -> [Bool] {
    precondition(bliTolerance > 0 && bliTolerance < 1, "mask requires a tolerance in (0, 1)")
    precondition(points.ndim == 2 && points.dim(0) == grid.dim, "points must be [dim, numPoints]")
    let dim = grid.dim
    let nPts = points.dim(1)
    let host = points.reshaped([dim * nPts]).asArray(Float.self)
    let dims = grid.size
    var mask = [Bool](repeating: false, count: dims.reduce(1, *))
    for p in 0..<nPts {
        let point = (0..<dim).map { Double(host[$0 * nPts + p]) }
        for node in tolStarNodes(tolerance: bliTolerance, grid: grid, point: point) {
            var flat = 0
            for axis in 0..<dim { flat = flat * dims[axis] + node[axis] }
            mask[flat] = true
        }
    }
    return mask
}

/// Truncated-sinc evaluation on tolStar neighbour nodes, accumulated on the host.
private func offGridPointsTruncated(
    grid: KWaveGrid, points: MLXArray, scale: [Double]?, tolerance: Double
) -> MLXArray {
    precondition(points.ndim == 2 && points.dim(0) == grid.dim, "points must be [dim, numPoints]")
    let dim = grid.dim
    let nPts = points.dim(1)
    let host = points.reshaped([dim * nPts]).asArray(Float.self)
    let weights = normalizedScale(scale, nPts)
    let dims = grid.size
    let spacing = grid.spacing
    let axes = (0..<dim).map { centeredAxis(dims[$0], spacing[$0]) }

    var mask = [Float](repeating: 0, count: dims.reduce(1, *))
    for p in 0..<nPts {
        let point = (0..<dim).map { Double(host[$0 * nPts + p]) }
        for node in tolStarNodes(tolerance: tolerance, grid: grid, point: point) {
            var w = weights[p]
            var flat = 0
            for axis in 0..<dim {
                w *= sincU(.pi / spacing[axis] * (axes[axis][node[axis]] - point[axis]))
                flat = flat * dims[axis] + node[axis]
            }
            mask[flat] += Float(w)
        }
    }
    return MLXArray(mask).reshaped(dims)
}

private func normalizedScale(_ scale: [Double]?, _ nPts: Int) -> [Double] {
    guard let scale else { return [Double](repeating: 1, count: nPts) }
    precondition(scale.count == 1 || scale.count == nPts, "scale must be scalar or per-point")
    return scale.count == 1 ? [Double](repeating: scale[0], count: nPts) : scale
}

/// Exact (periodic) BLI evaluated over the full grid.
private func offGridPointsExact(grid: KWaveGrid, points: MLXArray, scale: [Double]?) -> MLXArray {
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

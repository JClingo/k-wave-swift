import Foundation
import MLX

/// Centered physical coordinates of a grid axis: `(0..<n).map { (i - n/2) * d }`, matching k-Wave's
/// `kgrid.x`/`y`/`z` and the Cartesian-sensor coordinate convention.
func centeredAxis(_ n: Int, _ d: Double) -> [Double] {
    let half = n / 2
    return (0..<n).map { Double($0 - half) * d }
}

/// Cartesian coordinates of the nonzero points of a binary grid mask, in ascending C-flat-index
/// order. Returns `coords` shaped `[dim][nPoints]` (mirroring k-wave-python `grid2cart`) and the
/// flat indices of those points. `mask` is grid-shaped.
func grid2cart(grid: KWaveGrid, mask: MLXArray) -> (coords: [[Double]], flatIndices: [Int]) {
    let host = mask.reshaped([mask.size]).asArray(Float.self)
    let flatIndices = host.enumerated().compactMap { $0.element != 0 ? $0.offset : nil }
    let axes = (0..<grid.dim).map { centeredAxis(grid.size[$0], grid.spacing[$0]) }
    let dims = grid.size
    var coords = [[Double]](repeating: [Double](repeating: 0, count: flatIndices.count), count: grid.dim)
    for (p, flat) in flatIndices.enumerated() {
        // Decode the C-order flat index into per-axis grid indices.
        var rem = flat
        for axis in stride(from: grid.dim - 1, through: 0, by: -1) {
            let idx = rem % dims[axis]
            rem /= dims[axis]
            coords[axis][p] = axes[axis][idx]
        }
    }
    return (coords, flatIndices)
}

/// Binary grid mask of the nearest grid nodes to a set of Cartesian points, mirroring the primary
/// output of k-wave-python `cart2grid` (2D/3D). Each point maps to node `round(coord/d) + N/2`
/// (banker's rounding, matching numpy `round`); points sharing a node collapse to one.
///
/// The `order_index`/`reorder_index` outputs are not ported (they carry upstream F/C-order and
/// 0/1-based quirks); use `grid2cart` for the reverse mapping with consistent ordering.
///
/// - Parameter cartData: `[dim, numPoints]` Cartesian coordinates [m].
/// - Returns: grid-shaped binary mask (`1` at occupied nodes).
public func cart2grid(grid: KWaveGrid, cartData: MLXArray) -> MLXArray {
    precondition(grid.dim == 2 || grid.dim == 3, "cart2grid supports 2D and 3D")
    precondition(cartData.ndim == 2 && cartData.dim(0) == grid.dim, "cartData must be [dim, numPoints]")
    let dim = grid.dim
    let nPts = cartData.dim(1)
    let host = cartData.reshaped([dim * nPts]).asArray(Float.self)
    func coord(_ axis: Int, _ p: Int) -> Double { Double(host[axis * nPts + p]) }
    let dims = grid.size
    let spacing = grid.spacing

    var mask = [Float](repeating: 0, count: dims.reduce(1, *))
    for p in 0..<nPts {
        var flat = 0
        for axis in 0..<dim {
            let idx = Int((coord(axis, p) / spacing[axis]).rounded(.toNearestOrEven)) + dims[axis] / 2
            precondition(idx >= 0 && idx < dims[axis], "Cartesian point outside grid (axis \(axis))")
            flat = flat * dims[axis] + idx
        }
        mask[flat] = 1
    }
    return MLXArray(mask).reshaped(dims)
}

/// Interpolation mode for `interpCartData`.
public enum CartInterp: Sendable { case nearest }

/// Resample time-series data recorded over a set of Cartesian sensor points onto the points of a
/// binary sensor mask, by nearest-neighbour interpolation. Mirrors k-wave-python `interp_cart_data`
/// (`interp="nearest"`, the default; the two-point linear mode is not implemented).
///
/// - Parameters:
///   - cartSensorData: `[numCartPoints, nt]` time series over `cartSensorMask`.
///   - cartSensorMask: `[dim, numCartPoints]` Cartesian sensor coordinates.
///   - binarySensorMask: grid-shaped binary mask whose points receive the interpolated series.
/// - Returns: `[numBinaryPoints, nt]` time series at the binary-mask points (ascending flat order).
public func interpCartData(
    grid: KWaveGrid,
    cartSensorData: MLXArray,
    cartSensorMask: MLXArray,
    binarySensorMask: MLXArray,
    interp: CartInterp = .nearest
) -> MLXArray {
    precondition(cartSensorMask.ndim == 2 && cartSensorMask.dim(0) == grid.dim,
                 "cartSensorMask must be [dim, numCartPoints]")
    let nCart = cartSensorMask.dim(1)
    precondition(cartSensorData.dim(0) == nCart, "cartSensorData rows must match cartSensorMask points")

    // Cartesian coords of the binary-mask points, and the Cartesian sensor coords (host).
    let (bsmCoords, _) = grid2cart(grid: grid, mask: binarySensorMask)
    let nBinary = bsmCoords.first?.count ?? 0
    let cartHost = cartSensorMask.reshaped([grid.dim * nCart]).asArray(Float.self)
    func cartCoord(_ axis: Int, _ i: Int) -> Double { Double(cartHost[axis * nCart + i]) }

    // Nearest Cartesian point for each binary point (squared Euclidean distance).
    var nearest = [Int32](repeating: 0, count: nBinary)
    for j in 0..<nBinary {
        var best = Double.greatestFiniteMagnitude, bestI = 0
        for i in 0..<nCart {
            var dist = 0.0
            for axis in 0..<grid.dim {
                let delta = bsmCoords[axis][j] - cartCoord(axis, i)
                dist += delta * delta
            }
            if dist < best { best = dist; bestI = i }
        }
        nearest[j] = Int32(bestI)
    }
    return cartSensorData[MLXArray(nearest)]
}

import Foundation
import MLX

/// k-Wave spatial coordinate vector for an axis: `x_i = (i - floor(N/2)) * d` (0-based index).
public func gridCoordinates(_ n: Int, spacing d: Double) -> [Double] {
    let off = n / 2
    return (0..<n).map { Double($0 - off) * d }
}

/// Enlarge a 2D matrix by padding each side. With `edgeVal == nil` the outer edge values are
/// replicated (k-Wave default); otherwise the constant `edgeVal` is used. Mirrors `expandMatrix`.
public func expandMatrix(
    _ m: MLXArray, padX: (Int, Int), padY: (Int, Int), edgeVal: Float? = nil
) -> MLXArray {
    precondition(m.ndim == 2, "expandMatrix currently supports 2D")
    if let v = edgeVal {
        return MLX.padded(m, widths: [.init((padX.0, padX.1)), .init((padY.0, padY.1))],
                          value: MLXArray(v))
    }
    let nx = m.dim(0), ny = m.dim(1)
    // Replicate top/bottom rows, then left/right columns.
    var rows: [MLXArray] = []
    if padX.0 > 0 { rows.append(MLX.broadcast(m[0..<1, 0...], to: [padX.0, ny])) }
    rows.append(m)
    if padX.1 > 0 { rows.append(MLX.broadcast(m[(nx - 1)..<nx, 0...], to: [padX.1, ny])) }
    let vExpanded = rows.count == 1 ? m : MLX.concatenated(rows, axis: 0)

    let nx2 = vExpanded.dim(0)
    var cols: [MLXArray] = []
    if padY.0 > 0 { cols.append(MLX.broadcast(vExpanded[0..., 0..<1], to: [nx2, padY.0])) }
    cols.append(vExpanded)
    if padY.1 > 0 { cols.append(MLX.broadcast(vExpanded[0..., (ny - 1)..<ny], to: [nx2, padY.1])) }
    return cols.count == 1 ? vExpanded : MLX.concatenated(cols, axis: 1)
}

/// Map 2D Cartesian sensor points onto a binary grid via nearest-neighbour (k-Wave `cart2grid`).
///
/// - Returns: `(mask, reorderIndex)` — a `[nx, ny]` 0/1 mask and the order in which the masked
///   points appear in row-major (C-order) traversal, so recorded data can be re-sorted to the
///   original Cartesian point order.
public func cart2grid2D(
    grid: KWaveGrid, points: [(x: Double, y: Double)]
) -> (mask: MLXArray, reorderIndex: [Int]) {
    let offX = grid.nx / 2, offY = grid.ny / 2
    var occupant = [Int](repeating: -1, count: grid.nx * grid.ny)
    for (n, p) in points.enumerated() {
        let ix = Int((p.x / grid.dx).rounded()) + offX
        let iy = Int((p.y / grid.dy).rounded()) + offY
        precondition(ix >= 0 && ix < grid.nx && iy >= 0 && iy < grid.ny,
                     "Cartesian point \(n) lies outside the grid")
        occupant[ix * grid.ny + iy] = n
    }
    var mask = [Float](repeating: 0, count: grid.nx * grid.ny)
    var reorder: [Int] = []
    for flat in 0..<occupant.count where occupant[flat] >= 0 {
        mask[flat] = 1
        reorder.append(occupant[flat])
    }
    return (MLXArray(mask).reshaped([grid.nx, grid.ny]), reorder)
}

/// Cartesian coordinates of the nonzero points of a 2D binary grid (k-Wave `grid2cart`).
///
/// - Returns: `(points, orderIndex)` where `points` are `(x, y)` in metres and `orderIndex` are the
///   flattened (row-major) grid indices of those points.
public func grid2cart2D(grid: KWaveGrid, mask: MLXArray) -> (points: [(x: Double, y: Double)], orderIndex: [Int]) {
    let host = mask.reshaped([mask.size]).asArray(Float.self)
    let xs = gridCoordinates(grid.nx, spacing: grid.dx)
    let ys = gridCoordinates(grid.ny, spacing: grid.dy)
    var points: [(x: Double, y: Double)] = []
    var order: [Int] = []
    for flat in 0..<host.count where host[flat] != 0 {
        points.append((xs[flat / grid.ny], ys[flat % grid.ny]))
        order.append(flat)
    }
    return (points, order)
}

import Foundation
import MLX

/// Cartesian coordinates of points evenly distributed on a circle or arc, mirroring k-Wave
/// `makeCartCircle`. Returns a `[2, numPoints]` array (row 0 = x, row 1 = y) suitable for use as a
/// Cartesian sensor/source mask.
///
/// - Parameters:
///   - radius: circle/arc radius [m].
///   - numPoints: number of points.
///   - center: `(x, y)` centre [m].
///   - arcAngle: arc angle [rad]; `2π` (default) gives a full circle.
public func makeCartCircle(
    radius: Double, numPoints: Int, center: (x: Double, y: Double) = (0, 0),
    arcAngle: Double = 2 * .pi
) -> MLXArray {
    precondition(numPoints > 0, "numPoints must be positive")
    let fullCircle = arcAngle == 2 * .pi
    let nSteps = Double(fullCircle ? numPoints : numPoints - 1)

    var data = [Float](repeating: 0, count: 2 * numPoints)
    for i in 0..<numPoints {
        let angle = Double(i) * arcAngle / nSteps + .pi / 2
        data[i] = Float(radius * cos(angle) + center.x)
        data[numPoints + i] = Float(radius * sin(-angle) + center.y)
    }
    return MLXArray(data).reshaped([2, numPoints])
}

/// Cartesian coordinates of points evenly distributed over a (optionally rotated) rectangle in 2D,
/// mirroring k-Wave `makeCartRect`. Returns a `[2, npts]` array where `npts = ceil(sqrt(numPoints·
/// Lx/Ly)) · ceil(numPoints / that)` (k-Wave rounds up to a full grid).
///
/// - Parameters:
///   - center: `(x, y)` centre of the rectangle [m].
///   - lx: height along x before rotation [m].
///   - ly: width along y before rotation [m].
///   - theta: rotation [degrees]; `nil` = no rotation.
///   - numPoints: approximate number of points (actual count rounds up to a full grid).
public func makeCartRect(
    center: (x: Double, y: Double), lx: Double, ly: Double, theta: Double? = nil, numPoints: Int
) -> MLXArray {
    precondition(numPoints > 0 && lx > 0 && ly > 0, "lx, ly, numPoints must be positive")
    let nptsX = Int(ceil((Double(numPoints) * lx / ly).squareRoot()))
    let nptsY = Int(ceil(Double(numPoints) / Double(nptsX)))
    let npts = nptsX * nptsY
    let dx = 2.0 / Double(nptsX), dy = 2.0 / Double(nptsY)

    // Scale (Lx/2, Ly/2) then rotate; A = R·S.
    let (c, s) = theta.map { (cos($0 * .pi / 180), sin($0 * .pi / 180)) } ?? (1, 0)
    let sx = lx / 2, sy = ly / 2
    let a00 = c * sx, a01 = -s * sy
    let a10 = s * sx, a11 = c * sy

    func linspace(_ lo: Double, _ hi: Double, _ n: Int) -> [Double] {
        n == 1 ? [lo] : (0..<n).map { lo + (hi - lo) * Double($0) / Double(n - 1) }
    }
    let pX = linspace(-1 + dx / 2, 1 - dx / 2, nptsX)
    let pY = linspace(-1 + dy / 2, 1 - dy / 2, nptsY)

    var data = [Float](repeating: 0, count: 2 * npts)
    for i in 0..<nptsX {
        for j in 0..<nptsY {
            let idx = i * nptsY + j          // C-order flatten of meshgrid(indexing='ij').
            let px = pX[i], py = pY[j]
            data[idx] = Float(a00 * px + a01 * py + center.x)
            data[npts + idx] = Float(a10 * px + a11 * py + center.y)
        }
    }
    return MLXArray(data).reshaped([2, npts])
}

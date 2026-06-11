import Foundation
import MLX

/// Binary 2D arc on a grid, mirroring k-Wave `makeArc` (finite radius). The arc lies on the circle
/// of curvature `radius` passing through `arcPos` (its midpoint), oriented so the focus is at
/// `focusPos`; points subtending more than `asin(diameter/2/radius)` from the midpoint are removed.
///
/// Positions are 0-based grid indices (consistent with `makeCircle`). `diameter` must be a positive
/// odd number of grid points and `≤ 2·radius`. The infinite-radius (straight-line) case is not
/// supported.
public func makeArc(
    nx: Int, ny: Int, arcPos: (x: Int, y: Int), radius: Double, diameter: Int, focusPos: (x: Int, y: Int)
) -> MLXArray {
    precondition(radius.isFinite && radius > 0, "radius must be positive and finite")
    precondition(diameter > 0 && diameter % 2 == 1, "diameter must be a positive odd number of grid points")
    precondition(Double(diameter) <= 2 * radius, "diameter must be ≤ 2·radius")
    precondition((arcPos.x, arcPos.y) != (focusPos.x, focusPos.y), "focusPos must differ from arcPos")

    let (ax, ay) = arcPos, (fx, fy) = focusPos
    let halfArcAngle = asin(Double(diameter) / 2 / radius)

    // Centre of the circle of curvature (banker's rounding to match Python `round`).
    let distCF = (Double((ax - fx) * (ax - fx) + (ay - fy) * (ay - fy))).squareRoot()
    let cx = Int((radius / distCF * Double(fx - ax) + Double(ax)).rounded(.toNearestOrEven))
    let cy = Int((radius / distCF * Double(fy - ay) + Double(ay)).rounded(.toNearestOrEven))

    var host = makeCircle(nx: nx, ny: ny, cx: cx, cy: cy, radius: radius)
        .reshaped([nx * ny]).asArray(Float.self)

    // Remove circle points subtending more than the half-arc angle from the midpoint.
    let v1x = Double(ax - cx), v1y = Double(ay - cy)
    let l1 = (v1x * v1x + v1y * v1y).squareRoot()
    for px in 0..<nx {
        for py in 0..<ny where host[px * ny + py] != 0 {
            let v2x = Double(px - cx), v2y = Double(py - cy)
            let l2 = (v2x * v2x + v2y * v2y).squareRoot()
            let theta = acos((v1x * v2x + v1y * v2y) / (l1 * l2))
            if theta > halfArcAngle { host[px * ny + py] = 0 }
        }
    }
    return MLXArray(host).reshaped([nx, ny])
}

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

/// Cartesian coordinates of points evenly distributed over an arc, mirroring k-Wave `makeCartArc`.
/// The arc midpoint is `arcPos`, oriented so the focus lies at `focusPos`; the opening is `diameter`
/// and the radius of curvature is `radius` (use `.infinity` for a straight line). Returns a
/// `[2, numPoints]` array. The first/last points are offset inward by half the angular spacing.
public func makeCartArc(
    arcPos: (x: Double, y: Double), radius: Double, diameter: Double,
    focusPos: (x: Double, y: Double), numPoints: Int
) -> MLXArray {
    precondition(numPoints > 0 && diameter > 0, "numPoints and diameter must be positive")
    precondition((arcPos.x, arcPos.y) != (focusPos.x, focusPos.y), "focusPos must differ from arcPos")
    let r = radius.isInfinite ? 1e10 * diameter : radius
    precondition(r > 0 && diameter <= 2 * r, "diameter must be ≤ 2·radius")

    let varphiMax = asin(diameter / (2 * r))
    let dvarphi = 2 * varphiMax / Double(numPoints)
    let t0 = -varphiMax + dvarphi / 2, t1 = varphiMax - dvarphi / 2

    // compute_linear_transform2D: rotate canonical arc (back on +y axis) to the beam orientation.
    let bx = focusPos.x - arcPos.x, by = focusPos.y - arcPos.y
    let bn = (bx * bx + by * by).squareRoot()
    let bvx = bx / bn, bvy = by / bn
    let theta = atan2(bvy, bvx) - atan2(-1, 0)   // canonical beam_vec0 = [0, -1].
    let cT = cos(theta), sT = sin(theta)
    let offX = arcPos.x + r * bvx, offY = arcPos.y + r * bvy

    var data = [Float](repeating: 0, count: 2 * numPoints)
    for i in 0..<numPoints {
        let t = numPoints == 1 ? t0 : t0 + (t1 - t0) * Double(i) / Double(numPoints - 1)
        let p0x = r * sin(t), p0y = r * cos(t)
        data[i] = Float(cT * p0x - sT * p0y + offX)
        data[numPoints + i] = Float(sT * p0x + cT * p0y + offY)
    }
    return MLXArray(data).reshaped([2, numPoints])
}

/// Cartesian coordinates of points distributed over a sphere via the Golden Section Spiral,
/// mirroring k-Wave `makeCartSphere`. Returns a `[3, numPoints]` array (rows x, y, z).
public func makeCartSphere(
    radius: Double, numPoints: Int, center: (x: Double, y: Double, z: Double) = (0, 0, 0)
) -> MLXArray {
    precondition(numPoints > 0, "numPoints must be positive")
    let inc = Double.pi * (3 - 5.0.squareRoot())
    let off = 2.0 / Double(numPoints)

    var data = [Float](repeating: 0, count: 3 * numPoints)
    for k in 0..<numPoints {
        let y = Double(k) * off - 1 + off / 2
        let r = (1 - y * y).squareRoot()
        let phi = Double(k) * inc
        data[k] = Float(radius * cos(phi) * r + center.x)
        data[numPoints + k] = Float(radius * y + center.y)
        data[2 * numPoints + k] = Float(radius * sin(phi) * r + center.z)
    }
    return MLXArray(data).reshaped([3, numPoints])
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

// MARK: - Disc and bowl point samplers

/// Rotation matrix (row-major 3×3) mapping the canonical direction `[0, 0, −1]` onto the unit
/// vector from `pos1` to `pos2` (Rodrigues' formula), plus that direction — mirroring k-wave-python
/// `compute_rotation_between_vectors`. Returns identity for coincident points and `−I` for the
/// anti-parallel case.
func rotationToBeamAxis(from pos1: [Double], to pos2: [Double]) -> (r: [Double], direction: [Double]) {
    let d = zip(pos2, pos1).map(-)
    let mag = (d.map { $0 * $0 }.reduce(0, +)).squareRoot()
    let identity: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    if mag < .ulpOfOne { return (identity, [0, 0, 0]) }
    let dir = d.map { $0 / mag }

    let ref: [Double] = [0, 0, -1]
    let axis = [ref[1] * dir[2] - ref[2] * dir[1],
                ref[2] * dir[0] - ref[0] * dir[2],
                ref[0] * dir[1] - ref[1] * dir[0]]
    let axisNorm = (axis.map { $0 * $0 }.reduce(0, +)).squareRoot()
    if axisNorm <= .ulpOfOne {
        let dot = zip(ref, dir).map(*).reduce(0, +)
        return (dot > 0 ? identity : identity.map { -$0 }, dir)
    }
    let u = axis.map { $0 / axisNorm }
    let angle = acos(max(-1, min(1, zip(ref, dir).map(*).reduce(0, +))))
    let c = cos(angle), s = sin(angle), t = 1 - c
    let r = [t * u[0] * u[0] + c,        t * u[0] * u[1] - s * u[2], t * u[0] * u[2] + s * u[1],
             t * u[0] * u[1] + s * u[2], t * u[1] * u[1] + c,        t * u[1] * u[2] - s * u[0],
             t * u[0] * u[2] - s * u[1], t * u[1] * u[2] + s * u[0], t * u[2] * u[2] + c]
    return (r, dir)
}

/// Cartesian points covering a disc using concentric rings (k-Wave `makeCartDisc`, the default
/// non-spiral sampling: `PACKING_NUMBER = 7` points per unit ring index). In 2D the disc lies in
/// the plane; in 3D it is rotated so its normal points at `focusPos`. The returned point count is
/// determined by the ring packing, not `numPoints` (which sets the ring budget).
public func makeCartDisc(
    discPos: [Double], radius: Double, focusPos: [Double]? = nil, numPoints: Int
) -> MLXArray {
    precondition(discPos.count == 2 || discPos.count == 3, "discPos must be 2D or 3D")
    precondition(radius > 0, "radius must be positive")
    let dim = discPos.count

    let numRadial = Int(ceil((Double(numPoints) / .pi).squareRoot()))
    let dRadial = numRadial > 1 ? radius / Double(numRadial - 1) : .infinity
    let rOf: (Int) -> Double = { k in
        numRadial > 1 ? Double(k) * (radius - dRadial / 2) / Double(numRadial - 1) : 0
    }
    var n = 1
    for k in 2...max(numRadial, 2) where k <= numRadial { n += (k - 1) * 7 }

    var px = [Double](repeating: 0, count: n)
    var py = [Double](repeating: 0, count: n)
    var idx = 1
    if numRadial >= 2 {
        for k in 2...numRadial {
            let nTheta = (k - 1) * 7
            for i in 0..<nTheta {
                let theta = Double(i) * 2 * .pi / Double(nTheta)
                px[idx] = rOf(k - 1) * cos(theta)
                py[idx] = rOf(k - 1) * sin(theta)
                idx += 1
            }
        }
    }

    var data = [Float](repeating: 0, count: dim * n)
    if dim == 2 {
        for i in 0..<n {
            data[i] = Float(px[i] + discPos[0])
            data[n + i] = Float(py[i] + discPos[1])
        }
    } else {
        // Rotate the in-plane points so the disc normal points at the focus, then translate.
        let r: [Double]
        if let focusPos {
            r = rotationToBeamAxis(from: discPos, to: focusPos).r
        } else {
            r = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        }
        for i in 0..<n {
            let p = [px[i], py[i], 0.0]
            for row in 0..<3 {
                let v = r[row * 3] * p[0] + r[row * 3 + 1] * p[1] + r[row * 3 + 2] * p[2]
                data[row * n + i] = Float(v + discPos[row])
            }
        }
    }
    return MLXArray(data).reshaped([dim, n])
}

/// Cartesian points covering a focused bowl using a golden-angle spiral over the spherical cap
/// (k-Wave `makeCartBowl`). The bowl's rear-surface midpoint is `bowlPos`; the beam axis points at
/// `focusPos`. Returns `[3, numPoints]`.
public func makeCartBowl(
    bowlPos: [Double], radius: Double, diameter: Double, focusPos: [Double], numPoints: Int
) -> MLXArray {
    precondition(bowlPos.count == 3 && focusPos.count == 3, "bowlPos/focusPos must be 3D")
    precondition(radius > 0 && diameter > 0 && diameter <= 2 * radius, "invalid bowl geometry")
    precondition(bowlPos != focusPos, "focusPos must differ from bowlPos")

    let goldenAngle = 2.39996322972865332223155550663361385312499901105811504
    let varphiMax = asin(diameter / (2 * radius))
    let C = 2 * Double.pi * (1 - cos(varphiMax)) / Double(numPoints - 1)

    let (r, dir) = rotationToBeamAxis(from: bowlPos, to: focusPos)
    let b = (0..<3).map { bowlPos[$0] + radius * dir[$0] }   // centre of the sphere of curvature.

    var data = [Float](repeating: 0, count: 3 * numPoints)
    for i in 0..<numPoints {
        let t = Double(i)
        let theta = goldenAngle * t
        let varphi = acos(1 - C * t / (2 * .pi))
        let p = [radius * cos(theta) * sin(varphi),
                 radius * sin(theta) * sin(varphi),
                 radius * cos(varphi)]
        for row in 0..<3 {
            let v = r[row * 3] * p[0] + r[row * 3 + 1] * p[1] + r[row * 3 + 2] * p[2]
            data[row * numPoints + i] = Float(v + b[row])
        }
    }
    return MLXArray(data).reshaped([3, numPoints])
}

/// Cartesian points covering a spherical segment (annular section of a focused bowl), mirroring
/// k-Wave `makeCartSphericalSegment`: a golden-angle spiral over the cap band between the inner
/// and outer aperture diameters. `numPointsInner` forces point positions to align with the inner
/// elements of a contiguous annular array (0 = standalone segment).
public func makeCartSphericalSegment(
    bowlPos: [Double], radius: Double, innerDiameter: Double, outerDiameter: Double,
    focusPos: [Double], numPoints: Int, numPointsInner: Int = 0
) -> MLXArray {
    precondition(bowlPos.count == 3 && focusPos.count == 3, "bowlPos/focusPos must be 3D")
    precondition(radius > 0 && radius.isFinite, "radius must be positive and finite")
    precondition(innerDiameter >= 0 && innerDiameter < outerDiameter, "need inner < outer diameter")
    precondition(outerDiameter <= 2 * radius, "outer diameter must be ≤ 2·radius")
    precondition(bowlPos != focusPos, "focusPos must differ from bowlPos")

    let goldenAngle = 2.39996322972865332223155550663361385312499901105811504
    let varphiMin = asin(innerDiameter / (2 * radius))
    let varphiMax = asin(outerDiameter / (2 * radius))

    let c: Double, tStart: Double, tEnd: Double
    if numPointsInner > 0 {
        c = (1 - cos(varphiMax)) / Double(numPoints + numPointsInner - 1)
        tStart = ceil((1 - cos(varphiMin)) / c)
        tEnd = Double(numPointsInner + numPoints - 1)
    } else {
        c = (1 - cos(varphiMax)) / Double(numPoints - 1)
        tStart = ceil((1 - cos(varphiMin)) / c)
        tEnd = Double(numPoints - 1)
    }

    let (r, dir) = rotationToBeamAxis(from: bowlPos, to: focusPos)
    let b = (0..<3).map { bowlPos[$0] + radius * dir[$0] }

    var data = [Float](repeating: 0, count: 3 * numPoints)
    for i in 0..<numPoints {
        let t = numPoints == 1 ? tStart
            : tStart + (tEnd - tStart) * Double(i) / Double(numPoints - 1)
        let theta = goldenAngle * t
        let varphi = acos(1 - c * t)
        let p = [radius * cos(theta) * sin(varphi),
                 radius * sin(theta) * sin(varphi),
                 radius * cos(varphi)]
        for row in 0..<3 {
            let v = r[row * 3] * p[0] + r[row * 3 + 1] * p[1] + r[row * 3 + 2] * p[2]
            data[row * numPoints + i] = Float(v + b[row])
        }
    }
    return MLXArray(data).reshaped([3, numPoints])
}

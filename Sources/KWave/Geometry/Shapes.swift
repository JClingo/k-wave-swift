import Foundation
import MLX

/// Default grid centre index (0-based), matching MATLAB k-Wave's `floor(N/2)+1` (1-based).
@inline(__always)
func defaultCenter(_ n: Int) -> Int { n / 2 }

/// Filled circle (disc) binary mask in 2D.
///
/// - Parameters:
///   - nx, ny: grid size.
///   - cx, cy: centre indices (0-based). Pass `nil` for the grid centre.
///   - radius: disc radius in grid points.
public func makeDisc(nx: Int, ny: Int, cx: Int? = nil, cy: Int? = nil, radius: Double) -> MLXArray {
    let centerX = cx ?? defaultCenter(nx)
    let centerY = cy ?? defaultCenter(ny)
    let xs = MLXArray(0..<nx).asType(.float32).reshaped([nx, 1])
    let ys = MLXArray(0..<ny).asType(.float32).reshaped([1, ny])
    let dx2 = (xs - Float(centerX)) * (xs - Float(centerX))
    let dy2 = (ys - Float(centerY)) * (ys - Float(centerY))
    let r2 = MLX.broadcast(dx2, to: [nx, ny]) + MLX.broadcast(dy2, to: [nx, ny])
    return (r2 .<= Float(radius * radius)).asType(.float32)
}

/// Circle outline (ring) binary mask in 2D via the midpoint circle algorithm.
///
/// Mirrors MATLAB / k-wave-python `makeCircle`. A single grid point is the centre, so the diameter
/// is always odd. Points outside the grid are clipped. Indices are 0-based; pass `nil` for centre.
///
/// - Parameters:
///   - arcAngle: optional arc extent in radians (default full circle, `2*pi`).
public func makeCircle(
    nx: Int, ny: Int, cx: Int? = nil, cy: Int? = nil,
    radius: Double, arcAngle: Double? = nil
) -> MLXArray {
    let arc = min(max(arcAngle ?? (2 * .pi), 0), 2 * .pi)
    let r = Int(radius.rounded())
    // Work in 1-based coordinates to mirror the reference exactly.
    let cx1 = (cx ?? defaultCenter(nx)) + 1
    let cy1 = (cy ?? defaultCenter(ny)) + 1

    var circle = [Float](repeating: 0, count: nx * ny)
    func set(_ px: Int, _ py: Int) {
        guard (atan2(Double(px - cx1), Double(py - cy1)) + .pi) <= arc else { return }
        guard px >= 1, px <= nx, py >= 1, py <= ny else { return }
        circle[(px - 1) * ny + (py - 1)] = 1
    }

    var x = 0, y = r, d = 1 - r
    set(cx1, cy1 - y)
    // Cardinal points: (cx, cy+y), (cx+y, cy), (cx-y, cy).
    set(cx1, cy1 + y); set(cx1 + y, cy1); set(cx1 - y, cy1)
    while x < (y - 1) {
        x += 1
        if d < 0 { d += 2 * x + 1 } else { y -= 1; d += 2 * (x - y + 1) }
        let px = [x + cx1, y + cx1, y + cx1, x + cx1, -x + cx1, -y + cx1, -y + cx1, -x + cx1]
        let py = [y + cy1, x + cy1, -x + cy1, -y + cy1, -y + cy1, -x + cy1, x + cy1, y + cy1]
        for i in 0..<8 { set(px[i], py[i]) }
    }
    return MLXArray(circle).reshaped([nx, ny])
}

/// Filled sphere (ball) binary mask in 3D.
public func makeBall(
    nx: Int, ny: Int, nz: Int,
    cx: Int? = nil, cy: Int? = nil, cz: Int? = nil,
    radius: Double
) -> MLXArray {
    let centerX = cx ?? defaultCenter(nx)
    let centerY = cy ?? defaultCenter(ny)
    let centerZ = cz ?? defaultCenter(nz)
    let xs = MLXArray(0..<nx).asType(.float32).reshaped([nx, 1, 1])
    let ys = MLXArray(0..<ny).asType(.float32).reshaped([1, ny, 1])
    let zs = MLXArray(0..<nz).asType(.float32).reshaped([1, 1, nz])
    let dx2 = MLX.broadcast((xs - Float(centerX)) * (xs - Float(centerX)), to: [nx, ny, nz])
    let dy2 = MLX.broadcast((ys - Float(centerY)) * (ys - Float(centerY)), to: [nx, ny, nz])
    let dz2 = MLX.broadcast((zs - Float(centerZ)) * (zs - Float(centerZ)), to: [nx, ny, nz])
    return ((dx2 + dy2 + dz2) .<= Float(radius * radius)).asType(.float32)
}

/// Spherical shell (sphere outline) binary mask in 3D.
///
/// Mirrors MATLAB / k-wave-python `makeSphere`: a guide circle in the x–y plane gives the swept
/// radius for each x-slice, and each slice is drawn as a `makeCircle` outline plus a line-by-line
/// fill, with the previous slice's points removed so the shell stays one voxel thick. The grid is
/// assumed cubic (as in the reference). Centre is the grid centre; the diameter is always odd.
public func makeSphere(nx: Int, ny: Int, nz: Int, radius: Double) -> MLXArray {
    // 1-based centres, matching the reference (floor(N/2)+1).
    let cx = nx / 2 + 1, cy = ny / 2 + 1, cz = nz / 2 + 1

    // Guide circle in (flipped) x–y plane: shape (ny, nx), element (a,b) at a*nx + b.
    let guide = makeCircleHost(rows: ny, cols: nx, crow: cy, ccol: cx, radius: radius)

    var sphere = [Float](repeating: 0, count: nx * ny * nz)   // (nx, ny, nz), x*ny*nz + y*nz + z
    func setSlice(_ x: Int, _ plane: [Float]) {               // plane is (ny, nz): y*nz + z
        let base = x * ny * nz
        for i in 0..<(ny * nz) { sphere[base + i] = plane[i] }
    }
    func getSlice(_ x: Int) -> [Float] {
        let base = x * ny * nz
        return Array(sphere[base ..< base + ny * nz])
    }

    let centerpoints = Array(stride(from: cx - Int(radius.rounded()), through: cx, by: 1))
    // reflection_offset = [len, len-1, ..., 2]
    let count = centerpoints.count

    var prevCircle = [Float](repeating: 0, count: ny * nz)
    for idx in 0..<count {
        let cp = centerpoints[idx]                            // 1-based x slice
        // Column of the guide circle at x = cp (over the ny axis).
        var maxIdx = 0, minNonzero = Int.max
        for a in 0..<ny {
            if guide[a * nx + (cp - 1)] != 0 {
                let ri = a + 1
                if ri > maxIdx { maxIdx = ri }
                if ri < minNonzero { minNonzero = ri }
            }
        }
        let sweptRadius = minNonzero == Int.max ? 0.0 : Double(maxIdx - minNonzero) / 2.0
        let sweptInt = roundHalfEven(sweptRadius)

        // Outline circle in the (ny, nz) plane.
        let circle = makeCircleHost(rows: ny, cols: nz, crow: cy, ccol: cz, radius: Double(sweptInt))
        var fill = [Float](repeating: 0, count: ny * nz)      // y*nz + z

        // Fill the circle line by line over z columns within [cz - r, cz + r].
        let zLo = cz - sweptInt, zHi = cz + sweptInt
        for fc in zLo...max(zLo, zHi) where fc >= 1 && fc <= nz {
            var maxR = 0, minR = Int.max, num = 0
            for y in 0..<ny {
                if circle[y * nz + (fc - 1)] != 0 {
                    let ri = y + 1
                    if ri > maxR { maxR = ri }
                    if ri < minR { minR = ri }
                    num += 1
                }
            }
            guard num > 0 else { continue }
            let start = minR, stop = maxR
            if start != stop && (stop - start) >= num {
                let lo = start + num / 2          // 1-based inclusive
                let hi = stop - num / 2           // 1-based inclusive
                if lo <= hi {
                    for y in (lo - 1)...(hi - 1) { fill[y * nz + (fc - 1)] = 1 }
                }
            }
        }

        var slice = [Float](repeating: 0, count: ny * nz)
        if idx == 0 {
            for i in 0..<(ny * nz) { slice[i] = circle[i] + fill[i] }
            prevCircle = slice
        } else {
            var prevAlt = [Float](repeating: 0, count: ny * nz)
            for i in 0..<(ny * nz) {
                prevAlt[i] = circle[i] + fill[i]
                let f = max(fill[i] - prevCircle[i], 0)
                slice[i] = circle[i] + f
            }
            prevCircle = prevAlt
        }
        setSlice(cp - 1, slice)

        // Mirror to the far hemisphere: sphere[cx + (count - idx) - 2] = current slice.
        if idx != count - 1 {
            let mirrorX = cx + (count - idx) - 2          // 0-based
            if mirrorX >= 0 && mirrorX < nx { setSlice(mirrorX, getSlice(cp - 1)) }
        }
    }
    return MLXArray(sphere).reshaped([nx, ny, nz])
}

/// Banker's rounding (round-half-to-even), matching NumPy's `round` used by the reference.
private func roundHalfEven(_ x: Double) -> Int {
    let f = x.rounded(.toNearestOrEven)
    return Int(f)
}

/// Host-array `makeCircle` returning a row-major `[Float]` of length `rows*cols`, element (i,j) at
/// `i*cols + j`. `crow`/`ccol` are 1-based centres (matching the reference `make_circle`).
private func makeCircleHost(rows: Int, cols: Int, crow: Int, ccol: Int, radius: Double) -> [Float] {
    let r = Int(radius.rounded())
    var circle = [Float](repeating: 0, count: rows * cols)
    func set(_ px: Int, _ py: Int) {
        guard px >= 1, px <= rows, py >= 1, py <= cols else { return }
        circle[(px - 1) * cols + (py - 1)] = 1
    }
    var x = 0, y = r, d = 1 - r
    set(crow, ccol - y)
    set(crow, ccol + y); set(crow + y, ccol); set(crow - y, ccol)
    while x < (y - 1) {
        x += 1
        if d < 0 { d += 2 * x + 1 } else { y -= 1; d += 2 * (x - y + 1) }
        let px = [x + crow, y + crow, y + crow, x + crow, -x + crow, -y + crow, -y + crow, -x + crow]
        let py = [y + ccol, x + ccol, -x + ccol, -y + ccol, -y + ccol, -x + ccol, x + ccol, y + ccol]
        for i in 0..<8 { set(px[i], py[i]) }
    }
    return circle
}

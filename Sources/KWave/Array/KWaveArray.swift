import Foundation
import MLX

/// Off-grid transducer array (mirrors k-wave-python `kWaveArray`, 2D elements). Each element is
/// described in physical (Cartesian) coordinates; its surface is covered with integration points
/// (`upsamplingRate` points per grid cell of element measure), each spread onto the grid with a
/// band-limited interpolant (`offGridPoints`), giving grid weights that drive or sample the
/// element off-grid.
///
/// Scope: 2D `arc`, `line`, and `rect` elements (the types whose Cartesian samplers are ported).
/// Disc and bowl elements (3D) follow with `makeCartDisc`/`makeCartBowl`.
public final class KWaveArray {
    private enum Geometry {
        case arc(position: (Double, Double), radius: Double, diameter: Double, focus: (Double, Double))
        case line(start: [Double], end: [Double])
        case rect(position: (Double, Double), lx: Double, ly: Double, theta: Double)
    }
    private struct ArrayElement {
        let geometry: Geometry
        let dim: Int          // 1 = line-like (arc, line), 2 = area (rect).
        let measure: Double   // length [m] or area [m²].
    }

    public let bliTolerance: Double
    public let upsamplingRate: Int
    private var elements: [ArrayElement] = []

    public var numberElements: Int { elements.count }

    public init(bliTolerance: Double = 0.1, upsamplingRate: Int = 10) {
        precondition(bliTolerance >= 0 && bliTolerance < 1, "bliTolerance must be in [0, 1)")
        self.bliTolerance = bliTolerance
        self.upsamplingRate = upsamplingRate
    }

    /// Add a 2D arc element (radius of curvature, aperture diameter, focus orientation).
    public func addArcElement(position: (Double, Double), radius: Double, diameter: Double,
                              focusPos: (Double, Double)) {
        let varphiMax = asin(diameter / (2 * radius))
        elements.append(ArrayElement(geometry: .arc(position: position, radius: radius,
                                                    diameter: diameter, focus: focusPos),
                                     dim: 1, measure: 2 * radius * varphiMax))
    }

    /// Add a 2D line element between two physical points.
    public func addLineElement(start: [Double], end: [Double]) {
        precondition(start.count == 2 && end.count == 2, "line elements are supported in 2D")
        let length = (zip(start, end).map { ($1 - $0) * ($1 - $0) }.reduce(0, +)).squareRoot()
        elements.append(ArrayElement(geometry: .line(start: start, end: end),
                                     dim: 1, measure: length))
    }

    /// Add a 2D rectangular element (`lx`×`ly`, rotated by `theta` degrees).
    public func addRectElement(position: (Double, Double), lx: Double, ly: Double, theta: Double) {
        elements.append(ArrayElement(geometry: .rect(position: position, lx: lx, ly: ly, theta: theta),
                                     dim: 2, measure: lx * ly))
    }

    /// Grid weights for one element: integration points spread with the BLI and scaled so the
    /// weights integrate to the element measure in grid units.
    public func elementGridWeights(grid: KWaveGrid, element: Int) -> MLXArray {
        precondition(grid.dim == 2, "this slice supports 2D grids")
        precondition(element >= 0 && element < elements.count, "element index out of range")
        let el = elements[element]

        // Measure in grid cells (assumes dx == dy), and the integration-point budget.
        let mGrid = el.measure / pow(grid.dx, Double(el.dim))
        let mIntRequested = Int(ceil(mGrid * Double(upsamplingRate)))

        let points: MLXArray
        switch el.geometry {
        case let .arc(position, radius, diameter, focus):
            points = makeCartArc(arcPos: position, radius: radius, diameter: diameter,
                                 focusPos: focus, numPoints: mIntRequested)
        case let .line(start, end):
            // Uniform points along the line, inset half a spacing from each end.
            let d = zip(start, end).map { ($1 - $0) / Double(mIntRequested) }
            var data = [Float](repeating: 0, count: 2 * mIntRequested)
            for i in 0..<mIntRequested {
                let f = mIntRequested == 1 ? 0.0 : Double(i) / Double(mIntRequested - 1)
                for axis in 0..<2 {
                    let lo = start[axis] + d[axis] / 2, hi = end[axis] - d[axis] / 2
                    data[axis * mIntRequested + i] = Float(lo + (hi - lo) * f)
                }
            }
            points = MLXArray(data).reshaped([2, mIntRequested])
        case let .rect(position, lx, ly, theta):
            points = makeCartRect(center: position, lx: lx, ly: ly, theta: theta,
                                  numPoints: mIntRequested)
        }

        // Scale uses the actual generated point count (some samplers round up to a full grid),
        // computed BEFORE trimming — matching k-Wave's order.
        let scale = mGrid / Double(points.dim(1))
        let trimmed = trimCartPoints(grid: grid, points: points)
        return offGridPoints(grid: grid, points: trimmed, scale: [scale],
                             bliTolerance: bliTolerance)
    }

    /// Summed grid weights over all elements (k-Wave `get_array_grid_weights`).
    public func arrayGridWeights(grid: KWaveGrid) -> MLXArray {
        precondition(!elements.isEmpty, "Cannot call method on an array with zero elements.")
        var weights = MLXArray.zeros(grid.size, dtype: .float32)
        for i in 0..<elements.count {
            weights = weights + elementGridWeights(grid: grid, element: i)
        }
        return weights
    }
}

/// Drop Cartesian points outside the grid bounds (k-wave-python `trim_cart_points`).
func trimCartPoints(grid: KWaveGrid, points: MLXArray) -> MLXArray {
    let dim = grid.dim
    let n = points.dim(1)
    let host = points.reshaped([dim * n]).asArray(Float.self)
    let dims = grid.size
    let spacing = grid.spacing
    let bounds = (0..<dim).map { axis -> (Float, Float) in
        let coords = centeredAxis(dims[axis], spacing[axis])
        return (Float(coords.first!), Float(coords.last!))
    }
    let keep = (0..<n).filter { p in
        (0..<dim).allSatisfy { axis in
            let v = host[axis * n + p]
            return v >= bounds[axis].0 && v <= bounds[axis].1
        }
    }
    var out = [Float](repeating: 0, count: dim * keep.count)
    for (j, p) in keep.enumerated() {
        for axis in 0..<dim { out[axis * keep.count + j] = host[axis * n + p] }
    }
    return MLXArray(out).reshaped([dim, keep.count])
}

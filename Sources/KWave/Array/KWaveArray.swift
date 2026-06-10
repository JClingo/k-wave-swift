import Foundation
import MLX

/// Off-grid transducer array (mirrors k-wave-python `kWaveArray`, 2D elements). Each element is
/// described in physical (Cartesian) coordinates; its surface is covered with integration points
/// (`upsamplingRate` points per grid cell of element measure), each spread onto the grid with a
/// band-limited interpolant (`offGridPoints`), giving grid weights that drive or sample the
/// element off-grid.
///
/// Elements: 2D `arc`, `line`, `rect`, `disc`; 3D `disc` and focused `bowl`. Affine array
/// transforms and annulus elements are not yet implemented.
public final class KWaveArray {
    private enum Geometry {
        case arc(position: (Double, Double), radius: Double, diameter: Double, focus: (Double, Double))
        case line(start: [Double], end: [Double])
        case rect(position: (Double, Double), lx: Double, ly: Double, theta: Double)
        case disc(position: [Double], diameter: Double, focus: [Double]?)
        case bowl(position: [Double], radius: Double, diameter: Double, focus: [Double])
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

    /// Add a disc element (2D: in-plane; 3D: normal oriented at `focusPos`).
    public func addDiscElement(position: [Double], diameter: Double, focusPos: [Double]? = nil) {
        precondition(position.count == 2 || position.count == 3, "position must be 2D or 3D")
        precondition(position.count == 2 || focusPos != nil, "3D disc elements require focusPos")
        let area = Double.pi * (diameter / 2) * (diameter / 2)
        elements.append(ArrayElement(geometry: .disc(position: position, diameter: diameter,
                                                     focus: focusPos),
                                     dim: 2, measure: area))
    }

    /// Add a 3D focused-bowl element (spherical-cap area `2πR²(1−cos φmax)`).
    public func addBowlElement(position: [Double], radius: Double, diameter: Double,
                               focusPos: [Double]) {
        precondition(position.count == 3 && focusPos.count == 3, "bowl elements are 3D")
        let varphiMax = asin(diameter / (2 * radius))
        let area = 2 * Double.pi * radius * radius * (1 - cos(varphiMax))
        elements.append(ArrayElement(geometry: .bowl(position: position, radius: radius,
                                                     diameter: diameter, focus: focusPos),
                                     dim: 2, measure: area))
    }

    /// Element measure in grid cells (assumes dx == dy).
    private func measureInGridCells(_ el: ArrayElement, _ grid: KWaveGrid) -> Double {
        el.measure / pow(grid.dx, Double(el.dim))
    }

    /// Trimmed integration points covering one element, plus the per-point BLI scale.
    private func integrationPoints(grid: KWaveGrid, element: Int) -> (points: MLXArray, scale: Double) {
        precondition(element >= 0 && element < elements.count, "element index out of range")
        let el = elements[element]
        let mGrid = measureInGridCells(el, grid)
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
        case let .disc(position, diameter, focus):
            points = makeCartDisc(discPos: position, radius: diameter / 2, focusPos: focus,
                                  numPoints: mIntRequested)
        case let .bowl(position, radius, diameter, focus):
            points = makeCartBowl(bowlPos: position, radius: radius, diameter: diameter,
                                  focusPos: focus, numPoints: mIntRequested)
        }

        // Scale uses the actual generated point count (some samplers round up to a full grid),
        // computed BEFORE trimming — matching k-Wave's order.
        let scale = mGrid / Double(points.dim(1))
        return (trimCartPoints(grid: grid, points: points), scale)
    }

    /// Grid weights for one element: integration points spread with the BLI and scaled so the
    /// weights integrate to the element measure in grid units.
    public func elementGridWeights(grid: KWaveGrid, element: Int) -> MLXArray {
        let (points, scale) = integrationPoints(grid: grid, element: element)
        return offGridPoints(grid: grid, points: points, scale: [scale],
                             bliTolerance: bliTolerance)
    }

    /// Binary mask covering every grid node touched by any element's BLI star — the mask to use
    /// as `source.pMask`/`sensor.mask` (k-Wave `get_array_binary_mask`).
    public func arrayBinaryMask(grid: KWaveGrid) -> MLXArray {
        precondition(!elements.isEmpty, "Cannot call method on an array with zero elements.")
        var mask = [Bool](repeating: false, count: grid.size.reduce(1, *))
        for i in 0..<elements.count {
            let (points, _) = integrationPoints(grid: grid, element: i)
            for (j, on) in offGridPointsMask(grid: grid, points: points,
                                             bliTolerance: bliTolerance).enumerated() where on {
                mask[j] = true
            }
        }
        return MLXArray(mask.map { Float($0 ? 1 : 0) }).reshaped(grid.size)
    }

    /// Distribute per-element source signals onto the grid source points selected by
    /// `arrayBinaryMask` (k-Wave `get_distributed_source_signal`, C order). The output rows match
    /// the ascending flat-index order the solver uses for source masks.
    ///
    /// - Parameter sourceSignal: `[numberElements, Nt]`.
    /// - Returns: `[numSourcePoints, Nt]` per-grid-point signals.
    public func distributedSourceSignal(grid: KWaveGrid, sourceSignal: MLXArray) -> MLXArray {
        precondition(sourceSignal.ndim == 2 && sourceSignal.dim(0) == elements.count,
                     "sourceSignal must be [numberElements, Nt]")
        let nt = sourceSignal.dim(1)
        let sig = sourceSignal.reshaped([elements.count * nt]).asArray(Float.self)

        let mask = arrayBinaryMask(grid: grid).reshaped([grid.size.reduce(1, *)]).asArray(Float.self)
        let maskInd = mask.enumerated().compactMap { $0.element != 0 ? $0.offset : nil }
        var rowOf = [Int: Int]()
        for (row, flat) in maskInd.enumerated() { rowOf[flat] = row }

        var out = [Float](repeating: 0, count: maskInd.count * nt)
        for el in 0..<elements.count {
            let w = elementGridWeights(grid: grid, element: el)
                .reshaped([grid.size.reduce(1, *)]).asArray(Float.self)
            for (flat, weight) in w.enumerated() where weight != 0 {
                guard let row = rowOf[flat] else { continue }
                for t in 0..<nt { out[row * nt + t] += weight * sig[el * nt + t] }
            }
        }
        return MLXArray(out).reshaped([maskInd.count, nt])
    }

    /// Combine per-grid-point sensor data (recorded with `arrayBinaryMask` as the sensor mask)
    /// back into per-element signals (k-Wave `combine_sensor_data`, C order): the weighted sum
    /// over each element's grid points, normalised by the element measure in grid cells.
    ///
    /// - Parameter sensorData: `[numSourcePoints, Nt]` in ascending flat-index order.
    /// - Returns: `[numberElements, Nt]`.
    public func combineSensorData(grid: KWaveGrid, sensorData: MLXArray) -> MLXArray {
        precondition(sensorData.ndim == 2, "sensorData must be [numSensorPoints, Nt]")
        let nt = sensorData.dim(1)
        let data = sensorData.reshaped([sensorData.dim(0) * nt]).asArray(Float.self)

        let mask = arrayBinaryMask(grid: grid).reshaped([grid.size.reduce(1, *)]).asArray(Float.self)
        let maskInd = mask.enumerated().compactMap { $0.element != 0 ? $0.offset : nil }
        precondition(maskInd.count == sensorData.dim(0),
                     "sensorData rows must match the array binary mask point count")
        var rowOf = [Int: Int]()
        for (row, flat) in maskInd.enumerated() { rowOf[flat] = row }

        var out = [Float](repeating: 0, count: elements.count * nt)
        for el in 0..<elements.count {
            let w = elementGridWeights(grid: grid, element: el)
                .reshaped([grid.size.reduce(1, *)]).asArray(Float.self)
            for (flat, weight) in w.enumerated() where weight != 0 {
                guard let row = rowOf[flat] else { continue }
                for t in 0..<nt { out[el * nt + t] += weight * data[row * nt + t] }
            }
            let mGrid = Float(measureInGridCells(elements[el], grid))
            for t in 0..<nt { out[el * nt + t] /= mGrid }
        }
        return MLXArray(out).reshaped([elements.count, nt])
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

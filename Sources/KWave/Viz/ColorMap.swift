import Foundation

/// An RGB colour with components in [0, 1].
public typealias RGB = (r: Double, g: Double, b: Double)

/// MATLAB-style `hot` colormap (black → red → yellow → white) of length `m`.
public func hotColorMap(_ m: Int) -> [RGB] {
    guard m > 0 else { return [] }
    let n = Int((3.0 / 8.0 * Double(m)).rounded(.towardZero))
    func ramp(_ count: Int) -> [Double] { (1...max(count, 1)).map { Double($0) / Double(count) } }

    var r = [Double](repeating: 1, count: m)
    var g = [Double](repeating: 1, count: m)
    var b = [Double](repeating: 1, count: m)
    let rUp = n > 0 ? ramp(n) : []
    for i in 0..<m { r[i] = i < n ? rUp[i] : 1 }
    for i in 0..<m { g[i] = i < n ? 0 : (i < 2 * n ? rUp[i - n] : 1) }
    let bCount = m - 2 * n
    let bUp = bCount > 0 ? (1...bCount).map { Double($0) / Double(bCount) } : []
    for i in 0..<m { b[i] = i < 2 * n ? 0 : bUp[i - 2 * n] }
    return (0..<m).map { (r[$0], g[$0], b[$0]) }
}

/// MATLAB-style grayscale colormap of length `m`.
public func grayColorMap(_ m: Int) -> [RGB] {
    guard m > 0 else { return [] }
    let denom = Double(max(m - 1, 1))
    return (0..<m).map { let v = Double($0) / denom; return (v, v, v) }
}

/// MATLAB-style `bone` colormap: `(7*gray + fliplr(hot)) / 8`.
public func boneColorMap(_ m: Int) -> [RGB] {
    let g = grayColorMap(m), h = hotColorMap(m)
    return (0..<m).map {
        // fliplr(hot) swaps the R and B channels.
        ((7 * g[$0].r + h[$0].b) / 8,
         (7 * g[$0].g + h[$0].g) / 8,
         (7 * g[$0].b + h[$0].r) / 8)
    }
}

/// The default k-Wave diverging colormap: blue-greys for negatives, hot for positives, white at
/// zero. Mirrors k-wave-python `get_color_map`.
public func getColorMap(numColors: Int = 256) -> [RGB] {
    let negPad = Int((48.0 * Double(numColors) / 256.0).rounded())
    let half = numColors / 2
    let neg = Array(boneColorMap(half + negPad).dropFirst(negPad))   // bone(half+pad)[pad:]
    let pos = Array(hotColorMap(half).reversed())                    // flipud(hot(half))
    return neg + pos
}

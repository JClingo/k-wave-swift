import Foundation
import CoreGraphics
import MLX

/// Map a 2D field to RGBA8 pixels using a colormap — the rendering core of the simulation
/// monitor and movie recorder. Row `i` of the matrix becomes row `i` of the image (matrix
/// orientation, like MATLAB `imagesc`): width = `ny`, height = `nx`.
///
/// - Parameters:
///   - field: `[nx, ny]` field values.
///   - colorMap: colour table (defaults to the k-Wave diverging map).
///   - plotScale: `.auto` uses a symmetric range `±max|field|` (k-Wave's display convention for
///     pressure, putting zero at the map's centre); `.fixed(lo, hi)` clamps to the given range.
/// - Returns: RGBA8 pixel buffer (row-major, 4 bytes per pixel) and image dimensions.
public func fieldToRGBA(
    _ field: MLXArray, colorMap: [RGB] = getColorMap(), plotScale: PlotScale = .auto
) -> (pixels: [UInt8], width: Int, height: Int) {
    precondition(field.ndim == 2, "field must be [nx, ny]")
    precondition(!colorMap.isEmpty, "colorMap must not be empty")
    let nx = field.dim(0), ny = field.dim(1)
    let host = field.asType(.float32).reshaped([nx * ny]).asArray(Float.self)

    let lo: Float, hi: Float
    switch plotScale {
    case .auto:
        let m = host.map { $0.isFinite ? abs($0) : 0 }.max() ?? 0
        let bound = m > 0 ? m : 1
        lo = -bound; hi = bound
    case let .fixed(l, h):
        lo = Float(l); hi = Float(h)
    }
    let span = max(hi - lo, .leastNormalMagnitude)
    let nColors = colorMap.count

    var pixels = [UInt8](repeating: 255, count: nx * ny * 4)
    for i in 0..<(nx * ny) {
        let v = host[i].isFinite ? host[i] : lo
        let frac = min(max((v - lo) / span, 0), 1)
        let idx = min(Int(frac * Float(nColors - 1) + 0.5), nColors - 1)
        let c = colorMap[idx]
        pixels[i * 4] = UInt8(min(max(c.r, 0), 1) * 255 + 0.5)
        pixels[i * 4 + 1] = UInt8(min(max(c.g, 0), 1) * 255 + 0.5)
        pixels[i * 4 + 2] = UInt8(min(max(c.b, 0), 1) * 255 + 0.5)
    }
    return (pixels, ny, nx)
}

/// Render a 2D field to a `CGImage` using a colormap (see `fieldToRGBA`).
public func fieldToImage(
    _ field: MLXArray, colorMap: [RGB] = getColorMap(), plotScale: PlotScale = .auto
) -> CGImage {
    let (pixels, width, height) = fieldToRGBA(field, colorMap: colorMap, plotScale: plotScale)
    let data = CFDataCreate(nil, pixels, pixels.count)!
    let provider = CGDataProvider(data: data)!
    return CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
}

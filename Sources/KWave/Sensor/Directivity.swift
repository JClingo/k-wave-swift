import Foundation
import MLX
import MLXFFT

/// 2D sensor directivity, a 1:1 port of k-Wave MATLAB `directionalResponse.m`.
///
/// The pressure field is decomposed into plane waves via the 2D FFT; each plane wave is weighted by
/// a per-element directional pattern, and the filtered field is sampled at the sensor points. k-Wave
/// applies this only in 2D with a binary sensor mask, for the pressure recording.
///
/// k-Wave builds its directional weights on centered k-grids and `fftshift`s them into FFT-natural
/// order to match `fft2(p)`. This grid's `kx`/`ky` are already FFT-natural (`fftfreq` order), so the
/// weights are built directly with no shift.
struct DirectivityFilter {
    /// One entry per unique directivity angle: the k-space weighting, the rows of the sensor list
    /// that share this angle, and those rows' flat grid indices.
    let entries: [(weight: MLXArray, rows: MLXArray, gather: MLXArray)]
    let nSensor: Int
    let gridSize: Int

    /// Directional sensor pressures `[nSensor]` from the k-space pressure field `pk = fft2(p)`.
    func apply(pk: MLXArray) -> MLXArray {
        var out = MLXArray.zeros([nSensor], dtype: .float32)
        for e in entries {
            let filtered = MLXFFT.ifft2(pk * e.weight).realPart().reshaped([gridSize])
            out[e.rows] = filtered[e.gather]
        }
        return out
    }
}

/// Normalized sinc `sin(πx)/(πx)` (==1 at 0), matching MATLAB `sinc`.
private func sincNormalized(_ x: MLXArray) -> MLXArray {
    let pix = Float.pi * x
    return MLX.which(x .== 0, MLXArray(Float(1)), MLX.sin(pix) / pix)
}

/// Build the 2D directivity filter from `sensor.directivityAngle`/`directivitySize`/
/// `directivityPattern`, or `nil` when directivity is not configured / not 2D / mask absent.
func makeDirectivityFilter(sensor: KWaveSensor, grid: KWaveGrid) -> DirectivityFilter? {
    guard grid.dim == 2, let mask = sensor.mask, let angleGrid = sensor.directivityAngle else {
        return nil
    }
    let gridSize = grid.nx * grid.ny
    let sensorFlat = flatNonzeroIndices(mask).asArray(Int32.self)   // ascending C-flat order.
    let nSensor = sensorFlat.count
    guard nSensor > 0 else { return nil }

    let angleHost = angleGrid.reshaped([angleGrid.size]).asArray(Float.self)
    let rowAngle = sensorFlat.map { Double(angleHost[Int($0)]) }
    let uniqueAngles = Array(Set(rowAngle)).sorted()
    let elemSize = sensor.directivitySize ?? (10 * max(grid.dx, grid.dy))

    // k-grids in FFT-natural order, matching fft2(p). directivity_wavenumbers = [ky; kx].
    let kx = grid.kx.asType(.float32)
    let ky = grid.ky.asType(.float32)

    var entries: [(weight: MLXArray, rows: MLXArray, gather: MLXArray)] = []
    for theta in uniqueAngles {
        let weight: MLXArray
        switch sensor.directivityPattern {
        case .pressure:
            // k_tangent = [cos θ, -sin θ]·[ky; kx]; weight = sinc(k_tangent·size/2).
            let kTangent = Float(cos(theta)) * ky - Float(sin(theta)) * kx
            weight = sincNormalized(kTangent * Float(elemSize / 2))
        case .gradient:
            // k_normal = [sin θ, cos θ]·[ky; kx]; weight = k_normal / |k| (0 at DC).
            let kNormal = Float(sin(theta)) * ky + Float(cos(theta)) * kx
            let kMag = grid.k.asType(.float32)
            weight = MLX.which(kMag .== 0, MLXArray(Float(0)), kNormal / kMag)
        }
        let rows = (0..<nSensor).filter { rowAngle[$0] == theta }
        entries.append((
            weight: weight,
            rows: MLXArray(rows.map { Int32($0) }),
            gather: MLXArray(rows.map { sensorFlat[$0] })))
    }
    return DirectivityFilter(entries: entries, nSensor: nSensor, gridSize: gridSize)
}

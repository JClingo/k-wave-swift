import Foundation
import MLX
import MLXFFT

/// Centered (monotonic) angular wavenumber axis, matching k-Wave `kgrid.kx` = `fftshift(2π·fftfreq)`.
private func centeredWavenumber(_ n: Int, _ d: Double) -> [Double] {
    let nat = wavenumberVector(n, d: d)            // FFT-natural order.
    let shift = n / 2                              // fftshift.
    return (0..<n).map { nat[(($0 - shift) % n + n) % n] }
}

private func fftShift2(_ x: MLXArray) -> MLXArray {
    MLX.roll(MLX.roll(x, shift: x.dim(0) / 2, axis: 0), shift: x.dim(1) / 2, axis: 1)
}
private func ifftShift2(_ x: MLXArray) -> MLXArray {
    MLX.roll(MLX.roll(x, shift: -(x.dim(0) / 2), axis: 0), shift: -(x.dim(1) / 2), axis: 1)
}

/// Photoacoustic FFT-based line reconstruction, mirroring k-wave-python `kspaceLineRecon`.
///
/// Takes a pressure time series `p` recorded over an evenly spaced line of sensors and reconstructs
/// the initial pressure distribution via the k-space dispersion-relation method: FFT in (t, y),
/// remap from `(ω, ky)` to `(kx, ky)` by nearest-neighbour interpolation, inverse FFT.
///
/// Only nearest-neighbour interpolation is implemented (k-Wave's default).
///
/// - Parameters:
///   - p: pressure series; `[Nt, Ny]` for `dataOrder == "ty"` (default) or `[Ny, Nt]` for `"yt"`.
///   - dy: sensor spacing [m]; dt: time step [s]; c: (homogeneous) sound speed [m/s].
///   - posCond: clamp negative values of the reconstruction to zero.
/// - Returns: `[Nt, Ny]` reconstructed initial pressure `p(x, y)`.
public func kspaceLineRecon(
    p: MLXArray, dy: Double, dt: Double, c: Double, dataOrder: String = "ty", posCond: Bool = false
) -> MLXArray {
    precondition(dataOrder == "ty" || dataOrder == "yt", "dataOrder must be \"ty\" or \"yt\"")
    let pTy = dataOrder == "yt" ? p.transposed() : p

    // Mirror the time data about t = 0 so the cosine transform is an FFT.
    let nt0 = pTy.dim(0)
    let revIdx = MLXArray((0..<nt0).reversed().map { Int32($0) })
    let mirrored = MLX.concatenated([MLX.take(pTy, revIdx, axis: 0), pTy[1..<nt0]], axis: 0)
    let nt = mirrored.dim(0), ny = mirrored.dim(1)

    // Centered wavenumber grids (kx along the mirrored-time axis with spacing dt·c).
    let kxc = centeredWavenumber(nt, dt * c)
    let kyc = centeredWavenumber(ny, dy)

    // Scaling factor sf and the inhomogeneous/DC handling (built on host).
    // sf = c²·√((ω/c)² − ky²)/(2ω); DC → c/2; zero where |ω| < |c·ky| (evanescent).
    var sfHost = [Float](repeating: 0, count: nt * ny)
    for i in 0..<nt {
        let w = c * kxc[i]
        for j in 0..<ny {
            let ky = kyc[j]
            let idx = i * ny + j
            if abs(w) < abs(c * ky) {
                sfHost[idx] = 0                                  // evanescent — excluded.
            } else if w == 0 && ky == 0 {
                sfHost[idx] = Float(c / 2)                       // DC limit.
            } else {
                let arg = (w / c) * (w / c) - ky * ky
                sfHost[idx] = Float(c * c * max(arg, 0).squareRoot() / (2 * w))
            }
        }
    }
    let sf = MLXArray(sfHost).reshaped([nt, ny]).asType(.complex64)

    // FFT of the (centered) mirrored data, scaled: p(ω, ky).
    let pSpec = sf * fftShift2(MLXFFT.fft2(ifftShift2(mirrored.asType(.complex64))))

    // Nearest-neighbour remap (ω, ky) → (kx, ky): along ky the query matches the grid exactly, so
    // only the ω/kx axis is interpolated. The index map is data-independent, so precompute it.
    var gather = [Int32](repeating: 0, count: nt * ny)
    var oobHost = [Float](repeating: 0, count: nt * ny)          // 1 = out of bounds → 0.
    let wLo = c * kxc[0], wHi = c * kxc[nt - 1]
    for i in 0..<nt {
        for j in 0..<ny {
            let wNew = c * (kxc[i] * kxc[i] + kyc[j] * kyc[j]).squareRoot()
            if wNew < wLo || wNew > wHi {
                oobHost[i * ny + j] = 1
                gather[i * ny + j] = Int32(j)                    // any valid index; zeroed by mask.
                continue
            }
            var best = 0, bestD = Double.greatestFiniteMagnitude
            for k in 0..<nt {
                let d = abs(c * kxc[k] - wNew)
                if d < bestD { bestD = d; best = k }
            }
            gather[i * ny + j] = Int32(best * ny + j)
        }
    }
    let gathered = pSpec.reshaped([nt * ny])[MLXArray(gather)].reshaped([nt, ny])
    let oob = MLXArray(oobHost).reshaped([nt, ny])
    let pInterp = MLX.which(oob .!= 0, MLXArray(Float(0)).asType(.complex64), gathered)

    // Inverse FFT to p(x, y); keep the non-negative-time half; correct scaling.
    var out = fftShift2(MLXFFT.ifft2(ifftShift2(pInterp))).realPart()
    out = out[(nt / 2)..<nt, 0..<ny]
    out = Float(4.0 / c) * out
    if posCond { out = MLX.maximum(out, MLXArray(Float(0))) }
    return out
}

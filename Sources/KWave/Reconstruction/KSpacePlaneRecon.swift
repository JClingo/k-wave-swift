import Foundation
import MLX
import MLXFFT

/// Photoacoustic FFT-based planar reconstruction, mirroring k-wave-python `kspacePlaneRecon` — the
/// 3D analog of `kspaceLineRecon`. Reconstructs `p(x, y, z)` from a pressure time series recorded on
/// a plane of sensors, by FFT in (t, y, z), dispersion remap `(ω, ky, kz) → (kx, ky, kz)` (nearest),
/// and inverse FFT.
///
/// Only nearest-neighbour interpolation is implemented (k-Wave's default).
///
/// - Parameters:
///   - p: pressure series; `[Nt, Ny, Nz]` for `dataOrder == "tyz"` (default) or `[Ny, Nz, Nt]` for
///     `"yzt"`.
///   - dy, dz: sensor spacings [m]; dt: time step [s]; c: (homogeneous) sound speed [m/s].
///   - posCond: clamp negative values of the reconstruction to zero.
/// - Returns: `[Nt, Ny, Nz]` reconstructed initial pressure `p(x, y, z)`.
public func kspacePlaneRecon(
    p: MLXArray, dy: Double, dz: Double, dt: Double, c: Double,
    dataOrder: String = "tyz", posCond: Bool = false
) -> MLXArray {
    precondition(dataOrder == "tyz" || dataOrder == "yzt", "dataOrder must be \"tyz\" or \"yzt\"")
    let pTyz = dataOrder == "yzt" ? p.transposed(2, 0, 1) : p

    // Mirror the time data about t = 0 (axis 0).
    let nt0 = pTyz.dim(0)
    let revIdx = MLXArray((0..<nt0).reversed().map { Int32($0) })
    let mirrored = MLX.concatenated([MLX.take(pTyz, revIdx, axis: 0), pTyz[1..<nt0]], axis: 0)
    let nt = mirrored.dim(0), ny = mirrored.dim(1), nz = mirrored.dim(2)

    // Centered wavenumber axes (kx along mirrored-time axis with spacing dt·c).
    let kxc = centeredWavenumber(nt, dt * c)
    let kyc = centeredWavenumber(ny, dy)
    let kzc = centeredWavenumber(nz, dz)

    // sf = c²·√((ω/c)² − ky² − kz²)/(2ω); DC → c/2; zero where |ω| < c·√(ky²+kz²) (evanescent).
    var sfHost = [Float](repeating: 0, count: nt * ny * nz)
    for i in 0..<nt {
        let w = c * kxc[i]
        for j in 0..<ny {
            for k in 0..<nz {
                let kPerpSq = kyc[j] * kyc[j] + kzc[k] * kzc[k]
                let idx = (i * ny + j) * nz + k
                if abs(w) < c * kPerpSq.squareRoot() {
                    sfHost[idx] = 0
                } else if w == 0 && kyc[j] == 0 && kzc[k] == 0 {
                    sfHost[idx] = Float(c / 2)
                } else {
                    let arg = (w / c) * (w / c) - kPerpSq
                    sfHost[idx] = Float(c * c * max(arg, 0).squareRoot() / (2 * w))
                }
            }
        }
    }
    let sf = MLXArray(sfHost).reshaped([nt, ny, nz]).asType(.complex64)

    let pSpec = sf * fftShiftAll(MLXFFT.fftn(ifftShiftAll(mirrored.asType(.complex64))))

    // Nearest remap along the ω/kx axis only (ky, kz queries match the grid exactly); precompute.
    let size = nt * ny * nz
    var gather = [Int32](repeating: 0, count: size)
    var oobHost = [Float](repeating: 0, count: size)
    let wLo = c * kxc[0], wHi = c * kxc[nt - 1]
    for i in 0..<nt {
        for j in 0..<ny {
            for k in 0..<nz {
                let idx = (i * ny + j) * nz + k
                let wNew = c * (kxc[i] * kxc[i] + kyc[j] * kyc[j] + kzc[k] * kzc[k]).squareRoot()
                if wNew < wLo || wNew > wHi {
                    oobHost[idx] = 1
                    gather[idx] = Int32(j * nz + k)
                    continue
                }
                var best = 0, bestD = Double.greatestFiniteMagnitude
                for m in 0..<nt {
                    let d = abs(c * kxc[m] - wNew)
                    if d < bestD { bestD = d; best = m }
                }
                gather[idx] = Int32((best * ny + j) * nz + k)
            }
        }
    }
    let gathered = pSpec.reshaped([size])[MLXArray(gather)].reshaped([nt, ny, nz])
    let oob = MLXArray(oobHost).reshaped([nt, ny, nz])
    let pInterp = MLX.which(oob .!= 0, MLXArray(Float(0)).asType(.complex64), gathered)

    var out = fftShiftAll(MLXFFT.ifftn(ifftShiftAll(pInterp))).realPart()
    out = out[(nt / 2)..<nt, 0..<ny, 0..<nz]
    out = Float(4.0 / c) * out
    if posCond { out = MLX.maximum(out, MLXArray(Float(0))) }
    return out
}

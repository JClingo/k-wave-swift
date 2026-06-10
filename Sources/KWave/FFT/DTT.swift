import Foundation
import MLX

/// Discrete cosine transform type (k-Wave `DiscreteCosine`, FFTW REDFT00–REDFT11).
public enum DiscreteCosineType: Int, Sendable { case i = 1, ii = 2, iii = 3, iv = 4 }
/// Discrete sine transform type (k-Wave `DiscreteSine`, FFTW RODFT00–RODFT11).
public enum DiscreteSineType: Int, Sendable { case i = 1, ii = 2, iii = 3, iv = 4 }

/// Unnormalized DCT matrix `M` with `X = M·x` (FFTW / `scipy.fft.dct(norm=None)` convention).
private func dctMatrix(_ type: DiscreteCosineType, _ n: Int) -> MLXArray {
    var m = [Float](repeating: 0, count: n * n)
    for k in 0..<n {
        for j in 0..<n {
            let v: Double
            switch type {
            case .i:
                // X_k = x_0 + (−1)^k·x_{N−1} + 2·Σ_{n=1}^{N−2} x_n·cos(πnk/(N−1))
                if j == 0 { v = 1 }
                else if j == n - 1 { v = k % 2 == 0 ? 1 : -1 }
                else { v = 2 * cos(.pi * Double(j * k) / Double(n - 1)) }
            case .ii:
                v = 2 * cos(.pi * Double((2 * j + 1) * k) / Double(2 * n))
            case .iii:
                v = j == 0 ? 1 : 2 * cos(.pi * Double(j * (2 * k + 1)) / Double(2 * n))
            case .iv:
                v = 2 * cos(.pi * Double((2 * j + 1) * (2 * k + 1)) / Double(4 * n))
            }
            m[k * n + j] = Float(v)
        }
    }
    return MLXArray(m).reshaped([n, n])
}

/// Unnormalized DST matrix `M` with `X = M·x` (FFTW / `scipy.fft.dst(norm=None)` convention).
private func dstMatrix(_ type: DiscreteSineType, _ n: Int) -> MLXArray {
    var m = [Float](repeating: 0, count: n * n)
    for k in 0..<n {
        for j in 0..<n {
            let v: Double
            switch type {
            case .i:
                v = 2 * sin(.pi * Double((j + 1) * (k + 1)) / Double(n + 1))
            case .ii:
                v = 2 * sin(.pi * Double((2 * j + 1) * (k + 1)) / Double(2 * n))
            case .iii:
                // X_k = (−1)^k·x_{N−1} + 2·Σ_{n=0}^{N−2} x_n·sin(π(n+1)(2k+1)/(2N))
                if j == n - 1 { v = k % 2 == 0 ? 1 : -1 }
                else { v = 2 * sin(.pi * Double((j + 1) * (2 * k + 1)) / Double(2 * n)) }
            case .iv:
                v = 2 * sin(.pi * Double((2 * j + 1) * (2 * k + 1)) / Double(4 * n))
            }
            m[k * n + j] = Float(v)
        }
    }
    return MLXArray(m).reshaped([n, n])
}

/// Apply transform matrix `M` along `axis` of `x` (`result[..., k, ...] = Σ_j M[k,j]·x[..., j, ...]`).
private func applyAlongAxis(_ x: MLXArray, _ m: MLXArray, axis: Int) -> MLXArray {
    let ax = axis < 0 ? x.ndim + axis : axis
    let moved = MLX.swappedAxes(x, ax, x.ndim - 1)
    let out = MLX.matmul(moved, m.transposed())
    return MLX.swappedAxes(out, ax, x.ndim - 1)
}

/// Discrete cosine transform of types I–IV along `axis`, in the unnormalized FFTW convention
/// (matching `scipy.fft.dct(norm=None)` and k-Wave `dtt1D`). Implemented as a precomputed-matrix
/// product — exact, and the matrix is a setup-time constant when used inside a solver loop.
public func dct(_ x: MLXArray, type: DiscreteCosineType, axis: Int = -1) -> MLXArray {
    let ax = axis < 0 ? x.ndim + axis : axis
    let n = x.dim(ax)
    precondition(type != .i || n >= 2, "DCT-I requires n >= 2")
    return applyAlongAxis(x.asType(.float32), dctMatrix(type, n), axis: ax)
}

/// Discrete sine transform of types I–IV along `axis`, in the unnormalized FFTW convention
/// (matching `scipy.fft.dst(norm=None)` and k-Wave `dtt1D`).
public func dst(_ x: MLXArray, type: DiscreteSineType, axis: Int = -1) -> MLXArray {
    let ax = axis < 0 ? x.ndim + axis : axis
    return applyAlongAxis(x.asType(.float32), dstMatrix(type, x.dim(ax)), axis: ax)
}

import Foundation
import MLX

/// The computational grid — the central k-Wave data structure.
///
/// A single type covers 1D, 2D, and 3D; `dim` selects the active dimensionality and unused
/// sizes are 1. Wavenumber grids (`kx`, `ky`, `kz`, `k`) are precomputed in FFT order so they
/// can be applied directly to the output of `MLXFFT.fft`/`fftn` without an `ifftshift`.
public struct KWaveGrid {
    public let dim: Int
    public let nx: Int, ny: Int, nz: Int
    public let dx: Double, dy: Double, dz: Double

    /// 1-D angular wavenumber vectors (FFT order).
    public let kxVec: MLXArray
    public let kyVec: MLXArray
    public let kzVec: MLXArray

    /// N-D broadcast wavenumber grids (FFT order) and magnitude.
    public let kx: MLXArray
    public let ky: MLXArray
    public let kz: MLXArray
    public let k: MLXArray

    // Time, populated by `makeTime`.
    public private(set) var dt: Double = 0
    public private(set) var nt: Int = 0
    public private(set) var tArray: MLXArray = MLXArray([Double]())

    public var totalGridPoints: Int { nx * ny * nz }

    /// Grid point spacing in each active dimension.
    public var spacing: [Double] {
        switch dim {
        case 1: return [dx]
        case 2: return [dx, dy]
        default: return [dx, dy, dz]
        }
    }

    /// Active grid size in each dimension.
    public var size: [Int] {
        switch dim {
        case 1: return [nx]
        case 2: return [nx, ny]
        default: return [nx, ny, nz]
        }
    }

    // MARK: - Constructors

    /// 1D grid.
    public init(nx: Int, dx: Double) {
        self.init(dim: 1, nx: nx, dx: dx, ny: 1, dy: 1, nz: 1, dz: 1)
    }

    /// 2D grid.
    public init(nx: Int, dx: Double, ny: Int, dy: Double) {
        self.init(dim: 2, nx: nx, dx: dx, ny: ny, dy: dy, nz: 1, dz: 1)
    }

    /// 3D grid.
    public init(nx: Int, dx: Double, ny: Int, dy: Double, nz: Int, dz: Double) {
        self.init(dim: 3, nx: nx, dx: dx, ny: ny, dy: dy, nz: nz, dz: dz)
    }

    private init(dim: Int, nx: Int, dx: Double, ny: Int, dy: Double, nz: Int, dz: Double) {
        self.dim = dim
        self.nx = nx; self.ny = ny; self.nz = nz
        self.dx = dx; self.dy = dy; self.dz = dz

        let kxV = MLXArray(converting: wavenumberVector(nx, d: dx))
        let kyV = MLXArray(converting: wavenumberVector(ny, d: dy))
        let kzV = MLXArray(converting: wavenumberVector(nz, d: dz))
        self.kxVec = kxV
        self.kyVec = kyV
        self.kzVec = kzV

        switch dim {
        case 1:
            self.kx = kxV
            self.ky = MLXArray(converting: [0.0])
            self.kz = MLXArray(converting: [0.0])
            self.k = MLX.abs(kxV)
        case 2:
            let kxg = kxV.reshaped([nx, 1])
            let kyg = kyV.reshaped([1, ny])
            self.kx = MLX.broadcast(kxg, to: [nx, ny])
            self.ky = MLX.broadcast(kyg, to: [nx, ny])
            self.kz = MLXArray(converting: [0.0])
            self.k = MLX.sqrt(self.kx * self.kx + self.ky * self.ky)
        default:
            let kxg = kxV.reshaped([nx, 1, 1])
            let kyg = kyV.reshaped([1, ny, 1])
            let kzg = kzV.reshaped([1, 1, nz])
            self.kx = MLX.broadcast(kxg, to: [nx, ny, nz])
            self.ky = MLX.broadcast(kyg, to: [nx, ny, nz])
            self.kz = MLX.broadcast(kzg, to: [nx, ny, nz])
            self.k = MLX.sqrt(self.kx * self.kx + self.ky * self.ky + self.kz * self.kz)
        }
    }

    // MARK: - Time

    /// Compute the time step and array from a CFL number, mirroring MATLAB `kWaveGrid.makeTime`.
    ///
    /// - Parameters:
    ///   - soundSpeedMax: reference (maximum) sound speed used for the CFL stability limit.
    ///   - soundSpeedMin: minimum sound speed used for the default end time. Defaults to max.
    ///   - cfl: Courant–Friedrichs–Lewy number (default 0.3).
    ///   - tEnd: optional end time; if nil, set to the grid diagonal traversal time.
    public mutating func makeTime(
        soundSpeedMax: Double,
        soundSpeedMin: Double? = nil,
        cfl: Double = 0.3,
        tEnd: Double? = nil
    ) {
        let cMin = soundSpeedMin ?? soundSpeedMax
        let dxMin = spacing.min() ?? dx
        let step = cfl * dxMin / soundSpeedMax

        let end: Double
        if let tEnd { end = tEnd } else {
            let diag = sqrt(zip(size, spacing).reduce(0.0) { acc, pair in
                let extent = Double(pair.0) * pair.1
                return acc + extent * extent
            })
            end = diag / cMin
        }

        let count = Int(floor(end / step)) + 1
        self.dt = step
        self.nt = count
        self.tArray = MLXArray(converting: (0..<count).map { Double($0) * step })
    }

    /// Set the time array explicitly (mirrors MATLAB `kWaveGrid.setTime`).
    public mutating func setTime(nt: Int, dt: Double) {
        self.dt = dt
        self.nt = nt
        self.tArray = MLXArray(converting: (0..<nt).map { Double($0) * dt })
    }
}

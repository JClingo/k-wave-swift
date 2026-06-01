import XCTest
import MLX
@testable import KWave

/// Parity for 2D sensor directivity. The k-wave-python NumPy backend does not apply directivity, so
/// the reference reproduces k-Wave MATLAB `directionalResponse.m` (verbatim formula) applied to the
/// NumPy solver's full-grid pressure. The Swift `DirectivityFilter` ports the same MATLAB formula;
/// this cross-checks the two implementations on a real simulated field.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_directivity.py`.
final class ParityDirectivityTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_2d_directivity.h5").path
    }

    private func run(pattern: DirectivityPattern, refKey: String, tol: Float) throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_directivity.py")
        }
        let f = try HDF5File(open: refPath)
        let nx = Int(try f.readFloatDataset("Nx").data[0])
        let ny = Int(try f.readFloatDataset("Ny").data[0])
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let rho0 = Double(try f.readFloatDataset("rho0").data[0])
        let dt = Double(try f.readFloatDataset("dt").data[0])
        let nt = Int(try f.readFloatDataset("Nt").data[0])
        let pml = Int(try f.readFloatDataset("pml_size").data[0])
        let sx = Int(try f.readFloatDataset("src_x").data[0])
        let sy = Int(try f.readFloatDataset("src_y").data[0])
        let size = Double(try f.readFloatDataset("size").data[0])
        let signal = try f.readFloatDataset("signal").data
        let (maskShape, maskData) = try f.readFloatDataset("line_mask")
        let (angShape, angData) = try f.readFloatDataset("angle_grid")
        let (refShape, refData) = try f.readFloatDataset(refKey)

        var grid = KWaveGrid(nx: nx, dx: dx, ny: ny, dy: dx)
        grid.setTime(nt: nt, dt: dt)

        var maskHost = [Float](repeating: 0, count: nx * ny)
        maskHost[sx * ny + sy] = 1
        var source = KWaveSource()
        source.pMask = MLXArray(maskHost).reshaped([nx, ny])
        source.p = MLXArray(signal)
        source.pMode = .additive

        var sensor = KWaveSensor(mask: MLXArray(maskData).reshaped(maskShape))
        sensor.directivityAngle = MLXArray(angData).reshaped(angShape)
        sensor.directivitySize = size
        sensor.directivityPattern = pattern

        var opts = SimulationOptions()
        opts.pmlSize = .uniform(pml)
        opts.smoothP0 = false

        let out = kspaceFirstOrder(grid: grid, medium: KWaveMedium(soundSpeed: c0, density: rho0),
                                   source: source, sensor: sensor, options: opts)
        let mine = out.p!
        let ref = MLXArray(refData).reshaped(refShape)
        XCTAssertEqual(mine.shape, refShape)

        let diff = MLX.abs(mine - ref)
        let n = Float(refData.count)
        let relL2 = sqrt(MLX.sum(diff * diff).item(Float.self) / n)
            / sqrt(MLX.sum(ref * ref).item(Float.self) / n)
        let relMax = MLX.max(diff).item(Float.self) / MLX.max(MLX.abs(ref)).item(Float.self)
        XCTAssertLessThan(relL2, tol, "\(refKey): rel-l2")
        XCTAssertLessThan(relMax, tol, "\(refKey): rel-max")
    }

    func test2DDirectivityPressureParity() throws {
        try run(pattern: .pressure, refKey: "ref_pressure", tol: 1e-4)
    }

    func test2DDirectivityGradientParity() throws {
        try run(pattern: .gradient, refKey: "ref_gradient", tol: 1e-3)
    }
}

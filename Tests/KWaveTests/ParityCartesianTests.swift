import XCTest
import MLX
@testable import KWave

/// Parity for the Cartesian-sensor / interpolation slice against k-wave-python (NumPy backend):
/// `fourierShift`, `interpCartData` (nearest), and Cartesian (off-grid) sensor recording.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_cartesian.py`.
final class ParityCartesianTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_2d_cartesian.h5").path
    }

    private func openRef() throws -> HDF5File {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_cartesian.py")
        }
        return try HDF5File(open: refPath)
    }

    private func assertClose(_ mine: MLXArray, _ refData: [Float], _ refShape: [Int],
                             _ label: String, tol: Float = 1e-4) {
        let ref = MLXArray(refData).reshaped(refShape)
        XCTAssertEqual(mine.shape, refShape, "\(label): shape")
        let diff = MLX.abs(mine - ref)
        let n = Float(refData.count)
        let relL2 = sqrt(MLX.sum(diff * diff).item(Float.self) / n)
            / sqrt(MLX.sum(ref * ref).item(Float.self) / n)
        let relMax = MLX.max(diff).item(Float.self) / MLX.max(MLX.abs(ref)).item(Float.self)
        XCTAssertLessThan(relL2, tol, "\(label): rel-l2")
        XCTAssertLessThan(relMax, tol, "\(label): rel-max")
    }

    func testFourierShiftParity() throws {
        let f = try openRef()
        let (inShape, inData) = try f.readFloatDataset("fs_in")
        let input = MLXArray(inData).reshaped(inShape)
        let (o1Shape, o1Data) = try f.readFloatDataset("fs_out_ax1")
        let (o0Shape, o0Data) = try f.readFloatDataset("fs_out_ax0")
        assertClose(fourierShift(input, shift: 0.5, axis: 1), o1Data, o1Shape, "fourierShift axis 1")
        assertClose(fourierShift(input, shift: 0.5, axis: 0), o0Data, o0Shape, "fourierShift axis 0")
    }

    func testInterpCartDataParity() throws {
        let f = try openRef()
        let nx = Int(try f.readFloatDataset("Nx").data[0])
        let ny = Int(try f.readFloatDataset("Ny").data[0])
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let dy = Double(try f.readFloatDataset("dy").data[0])

        let (cartShape, cartData) = try f.readFloatDataset("ic_cart_pts")
        let (dataShape, dataData) = try f.readFloatDataset("ic_cart_data")
        let (maskShape, maskData) = try f.readFloatDataset("ic_binary_mask")
        let (outShape, outData) = try f.readFloatDataset("ic_out")

        let grid = KWaveGrid(nx: nx, dx: dx, ny: ny, dy: dy)
        let result = interpCartData(
            grid: grid,
            cartSensorData: MLXArray(dataData).reshaped(dataShape),
            cartSensorMask: MLXArray(cartData).reshaped(cartShape),
            binarySensorMask: MLXArray(maskData).reshaped(maskShape))
        assertClose(result, outData, outShape, "interpCartData", tol: 1e-6)
    }

    func testCartesianSensorRecordingParity() throws {
        let f = try openRef()
        let nx = Int(try f.readFloatDataset("Nx").data[0])
        let ny = Int(try f.readFloatDataset("Ny").data[0])
        let dx = Double(try f.readFloatDataset("dx").data[0])
        let dy = Double(try f.readFloatDataset("dy").data[0])
        let c0 = Double(try f.readFloatDataset("c0").data[0])
        let rho0 = Double(try f.readFloatDataset("rho0").data[0])
        let dt = Double(try f.readFloatDataset("dt").data[0])
        let nt = Int(try f.readFloatDataset("Nt").data[0])
        let pml = Int(try f.readFloatDataset("pml_size").data[0])
        let sx = Int(try f.readFloatDataset("src_x").data[0])
        let sy = Int(try f.readFloatDataset("src_y").data[0])
        let signal = try f.readFloatDataset("rec_signal").data
        let (ptsShape, ptsData) = try f.readFloatDataset("rec_pts")
        let (pShape, pData) = try f.readFloatDataset("rec_p")

        var grid = KWaveGrid(nx: nx, dx: dx, ny: ny, dy: dy)
        grid.setTime(nt: nt, dt: dt)

        var maskHost = [Float](repeating: 0, count: nx * ny)
        maskHost[sx * ny + sy] = 1
        var source = KWaveSource()
        source.pMask = MLXArray(maskHost).reshaped([nx, ny])
        source.p = MLXArray(signal)
        source.pMode = .additive

        // Cartesian sensor mask: [dim, nPoints] physical coordinates.
        let sensor = KWaveSensor(mask: MLXArray(ptsData).reshaped(ptsShape))
        var opts = SimulationOptions()
        opts.pmlSize = .uniform(pml)
        opts.smoothP0 = false

        let out = kspaceFirstOrder(grid: grid, medium: KWaveMedium(soundSpeed: c0, density: rho0),
                                   source: source, sensor: sensor, options: opts)
        assertClose(out.p!, pData, pShape, "cartesian recorded p", tol: 1e-4)
    }
}

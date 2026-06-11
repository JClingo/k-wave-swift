import XCTest
import MLX
@testable import KWave

/// Parity for `KWaveArray` 2D element grid weights (arc, line, rect) against k-wave-python
/// `kWaveArray.get_element_grid_weights` / `get_array_grid_weights`.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_kwavearray.py`.
final class ParityKWaveArrayTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_kwavearray.h5").path
    }

    func testKWaveArrayGridWeightsParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_kwavearray.py")
        }
        let f = try HDF5File(open: refPath)
        let grid = KWaveGrid(nx: 48, dx: 1e-4, ny: 40, dy: 1e-4)

        let array = KWaveArray(bliTolerance: 0.1, upsamplingRate: 10)
        array.addArcElement(position: (-1.2e-3, 0.0), radius: 2e-3, diameter: 1.5e-3,
                            focusPos: (1e-3, 0.0))
        array.addLineElement(start: [0.4e-3, -1.1e-3], end: [1.3e-3, 0.7e-3])
        array.addRectElement(position: (-0.5e-3, 1.2e-3), lx: 0.8e-3, ly: 0.4e-3, theta: 25.0)

        func check(_ mine: MLXArray, _ key: String) throws {
            let (shape, data) = try f.readFloatDataset(key)
            let ref = MLXArray(data).reshaped(shape)
            XCTAssertEqual(mine.shape, shape, "\(key): shape")
            let maxErr = MLX.max(MLX.abs(mine - ref)).item(Float.self)
            let scale = MLX.max(MLX.abs(ref)).item(Float.self)
            XCTAssertLessThan(maxErr / scale, 1e-4, "\(key): max rel error")
        }

        try check(array.elementGridWeights(grid: grid, element: 0), "w_arc")
        try check(array.elementGridWeights(grid: grid, element: 1), "w_line")
        try check(array.elementGridWeights(grid: grid, element: 2), "w_rect")
        try check(array.arrayGridWeights(grid: grid), "w_all")

        // Binary mask: exact match.
        let (mShape, mData) = try f.readFloatDataset("mask")
        let mask = array.arrayBinaryMask(grid: grid)
        XCTAssertEqual(mask.shape, mShape)
        XCTAssertEqual(MLX.max(MLX.abs(mask - MLXArray(mData).reshaped(mShape))).item(Float.self),
                       0, "binary mask")

        // Source-signal distribution and sensor-data combination (C order).
        let (sigShape, sigData) = try f.readFloatDataset("sig")
        let dist = array.distributedSourceSignal(grid: grid,
                                                 sourceSignal: MLXArray(sigData).reshaped(sigShape))
        try check(dist, "dist")

        let (sdShape, sdData) = try f.readFloatDataset("sensor_data")
        let combined = array.combineSensorData(grid: grid,
                                               sensorData: MLXArray(sdData).reshaped(sdShape))
        try check(combined, "combined")
    }

    /// Disc/bowl Cartesian samplers and the corresponding array elements (2D disc, 3D bowl).
    func testDiscAndBowlElementsParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_kwavearray.py")
        }
        let f = try HDF5File(open: refPath)

        func check(_ mine: MLXArray, _ key: String) throws {
            let (shape, data) = try f.readFloatDataset(key)
            let ref = MLXArray(data).reshaped(shape)
            XCTAssertEqual(mine.shape, shape, "\(key): shape")
            let maxErr = MLX.max(MLX.abs(mine - ref)).item(Float.self)
            let scale = MLX.max(MLX.abs(ref)).item(Float.self)
            XCTAssertLessThan(maxErr / scale, 1e-4, "\(key): max rel error")
        }

        // Samplers.
        try check(makeCartDisc(discPos: [0.3e-3, -0.2e-3], radius: 0.6e-3, numPoints: 60), "disc2")
        try check(makeCartDisc(discPos: [0.2e-3, -0.1e-3, 0.4e-3], radius: 0.5e-3,
                               focusPos: [1.0e-3, 0.8e-3, -0.5e-3], numPoints: 60), "disc3")
        try check(makeCartBowl(bowlPos: [-1.0e-3, 0.0, 0.0], radius: 2.0e-3, diameter: 1.4e-3,
                               focusPos: [1.0e-3, 0.2e-3, 0.1e-3], numPoints: 80), "bowl")

        // 2D disc element weights.
        let g2 = KWaveGrid(nx: 48, dx: 1e-4, ny: 40, dy: 1e-4)
        let arr2 = KWaveArray(bliTolerance: 0.1, upsamplingRate: 10)
        arr2.addDiscElement(position: [0.3e-3, -0.2e-3], diameter: 1.2e-3)
        try check(arr2.elementGridWeights(grid: g2, element: 0), "w_disc2")

        // 3D bowl element weights.
        let g3 = KWaveGrid(nx: 32, dx: 1e-4, ny: 28, dy: 1e-4, nz: 24, dz: 1e-4)
        let arr3 = KWaveArray(bliTolerance: 0.1, upsamplingRate: 10)
        arr3.addBowlElement(position: [-1.0e-3, 0.0, 0.0], radius: 2.0e-3, diameter: 1.4e-3,
                            focusPos: [1.0e-3, 0.2e-3, 0.1e-3])
        try check(arr3.elementGridWeights(grid: g3, element: 0), "w_bowl")

        // Spherical-segment sampler and a 4-element annular array (LIFU-style).
        try check(makeCartSphericalSegment(bowlPos: [-1.0e-3, 0.0, 0.0], radius: 2.0e-3,
                                           innerDiameter: 0.6e-3, outerDiameter: 1.2e-3,
                                           focusPos: [1.0e-3, 0.2e-3, 0.1e-3], numPoints: 70),
                  "seg")
        let arr4 = KWaveArray(bliTolerance: 0.1, upsamplingRate: 10)
        arr4.addAnnularArray(position: [-1.0e-3, 0.0, 0.0], radius: 2.0e-3,
                             diameters: [(0.0, 0.5e-3), (0.6e-3, 0.9e-3),
                                         (1.0e-3, 1.3e-3), (1.4e-3, 1.6e-3)],
                             focusPos: [1.0e-3, 0.0, 0.0])
        for i in 0..<4 {
            try check(arr4.elementGridWeights(grid: g3, element: i), "w_ann\(i)")
        }

        // Affine array transforms: 2D translated+rotated arc; 3D translated + Euler-rotated bowl.
        let arr5 = KWaveArray(bliTolerance: 0.1, upsamplingRate: 10)
        arr5.addArcElement(position: (-1.2e-3, 0.0), radius: 2e-3, diameter: 1.5e-3,
                           focusPos: (1e-3, 0.0))
        arr5.setArrayPosition(translation: [0.4e-3, -0.3e-3], rotation: [30.0])
        try check(arr5.elementGridWeights(grid: g2, element: 0), "w_aff2")

        let arr6 = KWaveArray(bliTolerance: 0.1, upsamplingRate: 10)
        arr6.addBowlElement(position: [-1.0e-3, 0.0, 0.0], radius: 2.0e-3, diameter: 1.4e-3,
                            focusPos: [1.0e-3, 0.2e-3, 0.1e-3])
        arr6.setArrayPosition(translation: [0.2e-3, -0.1e-3, 0.3e-3], rotation: [20.0, -15.0, 40.0])
        try check(arr6.elementGridWeights(grid: g3, element: 0), "w_aff3")
    }
}

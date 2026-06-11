import XCTest
import MLX
@testable import KWave

/// Parity for `KWaveTransducer` (linear-array model) against k-wave-python `NotATransducer`:
/// masks, beamforming/elevation delays, delay mask, apodization mask, input-signal padding,
/// receive combination, and scan-line beamforming.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_transducer.py`.
final class ParityTransducerTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_transducer.h5").path
    }

    private func makeTransducer(signal: [Float]) -> KWaveTransducer {
        // Mirrors the reference: 12×64×16 grid, dx = 1e-4, dt = 2e-8; python position [1,5,3]
        // is 1-based → Swift (0, 4, 2); elements 0–1 inactive.
        let geometry = KWaveTransducerGeometry(
            numberElements: 8, elementWidth: 4, elementLength: 12, elementSpacing: 2,
            position: (0, 4, 2), dy: 1e-4, dz: 1e-4)
        var active = [Bool](repeating: true, count: 8)
        active[0] = false
        active[1] = false
        return KWaveTransducer(
            geometry: geometry, gridSize: [12, 64, 16], dt: 2e-8, soundSpeed: 1540,
            activeElements: active, focusDistance: 30e-3, elevationFocusDistance: 20e-3,
            steeringAngle: 10.0, transmitApodization: .hanning, receiveApodization: .hanning,
            inputSignal: signal)
    }

    func testTransducerParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_transducer.py")
        }
        let f = try HDF5File(open: refPath)
        let signal = try f.readFloatDataset("signal").data
        let t = makeTransducer(signal: signal)

        func checkExact(_ mine: MLXArray, _ key: String) throws {
            let (shape, data) = try f.readFloatDataset(key)
            let ref = MLXArray(data).reshaped(shape)
            XCTAssertEqual(mine.shape, shape, "\(key): shape")
            XCTAssertEqual(MLX.max(MLX.abs(mine - ref)).item(Float.self), 0, key)
        }
        func checkClose(_ mine: MLXArray, _ key: String, tol: Float = 1e-5) throws {
            let (shape, data) = try f.readFloatDataset(key)
            let ref = MLXArray(data).reshaped(shape)
            XCTAssertEqual(mine.shape, shape, "\(key): shape")
            let maxErr = MLX.max(MLX.abs(mine - ref)).item(Float.self)
            let scale = max(MLX.max(MLX.abs(ref)).item(Float.self), 1)
            XCTAssertLessThan(maxErr / scale, tol, key)
        }

        // Masks.
        try checkExact(t.allElementsMask, "all_mask")
        try checkExact(t.activeElementsMask, "act_mask")
        try checkExact(MLXArray(t.indexedActiveElementsMask.map { Float($0) })
            .reshaped(t.gridSize), "idx_act")

        // Delays (exact integers).
        let refBF = try f.readFloatDataset("bf").data.map { Int($0) }
        XCTAssertEqual(t.beamformingDelays, refBF, "beamforming_delays")
        let refElev = try f.readFloatDataset("elev").data.map { Int($0) }
        XCTAssertEqual(t.elevationBeamformingDelays, refElev, "elevation_delays")
        try checkExact(MLXArray(t.delayMask().map { Float($0) }).reshaped(t.gridSize), "dmask")

        // Apodization mask and padded input signal.
        try checkClose(t.transmitApodizationMask, "apod_mask")
        let refPadded = try f.readFloatDataset("padded").data
        XCTAssertEqual(t.paddedInputSignal(), refPadded, "padded input signal")

        // Receive combination (Swift C-order rows) and scan-line beamforming.
        let (sdShape, sdData) = try f.readFloatDataset("sd_swift")
        let combined = t.combineSensorData(MLXArray(sdData).reshaped(sdShape))
        try checkClose(combined, "combined", tol: 1e-5)

        let (seShape, seData) = try f.readFloatDataset("sd_elements")
        let line = t.scanLine(MLXArray(seData).reshaped(seShape))
        try checkClose(line, "line", tol: 1e-5)
    }
}

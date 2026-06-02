import XCTest
import MLX
@testable import KWave

/// Parity for water material-property functions against k-wave-python.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_water.py`.
final class ParityWaterTests: XCTestCase {
    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_water.h5").path
    }

    func testWaterPropertiesParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_water.py")
        }
        let f = try HDF5File(open: refPath)

        func check(_ label: String, _ temps: [Float], _ expected: [Float], _ fn: (Double) -> Double) {
            for (t, e) in zip(temps, expected) {
                let got = Float(fn(Double(t)))
                XCTAssertEqual(got, e, accuracy: max(abs(e) * 1e-5, 1e-6), "\(label) @ \(t)")
            }
        }

        check("soundSpeed", try f.readFloatDataset("ss_temp").data, try f.readFloatDataset("ss").data,
              { waterSoundSpeed($0) })
        check("density", try f.readFloatDataset("de_temp").data, try f.readFloatDataset("de").data,
              { waterDensity($0) })
        check("nonlinearity", try f.readFloatDataset("nl_temp").data, try f.readFloatDataset("nl").data,
              { waterNonlinearity($0) })

        let abF = Double(try f.readFloatDataset("ab_f").data[0])
        check("absorption", try f.readFloatDataset("ab_temp").data, try f.readFloatDataset("ab").data,
              { waterAbsorption(f: abF, temp: $0) })
    }
}

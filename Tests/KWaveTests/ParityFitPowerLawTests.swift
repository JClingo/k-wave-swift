import XCTest
import MLX
@testable import KWave

/// Parity for `fitPowerLawParams` (Nelder-Mead fit of absorption parameters) against k-wave-python.
/// The Swift Nelder-Mead mirrors scipy `fmin` (same simplex construction, coefficients, and stopping
/// rules), so the fitted parameters agree to well within the optimizer's own tolerance.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_fitpl.py`.
final class ParityFitPowerLawTests: XCTestCase {
    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_fitpl.h5").path
    }

    func testFitPowerLawParamsParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_fitpl.py")
        }
        let f = try HDF5File(open: refPath)

        for key in ["c1", "c2"] {
            let inp = try f.readFloatDataset(key + "_in").data.map { Double($0) }
            let out = try f.readFloatDataset(key + "_out").data.map { Double($0) }
            let (a0Fit, yFit) = fitPowerLawParams(a0: inp[0], y: inp[1], c0: inp[2],
                                                  fMin: inp[3], fMax: inp[4])
            XCTAssertEqual(a0Fit, out[0], accuracy: abs(out[0]) * 1e-3, "\(key): a0_fit")
            XCTAssertEqual(yFit, out[1], accuracy: 1e-3, "\(key): y_fit")
        }
    }
}

import XCTest
import MLX
@testable import KWave

/// Parity for spectral utilities `spect` and `extractAmpPhase` against k-wave-python.
///
/// Regenerate with: `.venv-kwave/bin/python Scripts/parity/generate_reference_spect.py`.
final class ParitySpectTests: XCTestCase {
    private var refPath: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Scripts/parity/reference_spect.h5").path
    }

    func testSpectAndExtractAmpPhaseParity() throws {
        guard FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("reference not generated: run Scripts/parity/generate_reference_spect.py")
        }
        let f = try HDF5File(open: refPath)
        let fs = Double(try f.readFloatDataset("fs").data[0])
        let sig = MLXArray(try f.readFloatDataset("sig").data)
        let refF = try f.readFloatDataset("f").data
        let refAmp = try f.readFloatDataset("amp").data
        let refPhase = try f.readFloatDataset("phase").data

        let (fOut, amp, phase) = spect(sig, fs: fs)
        XCTAssertEqual(fOut.count, refF.count)

        let ampMax = refAmp.map { abs($0) }.max() ?? 1
        for k in 0..<refF.count {
            XCTAssertEqual(Float(fOut[k]), refF[k], accuracy: max(abs(refF[k]) * 1e-5, 1e-3), "f[\(k)]")
            XCTAssertEqual(Float(amp[k]), refAmp[k], accuracy: max(abs(refAmp[k]) * 1e-4, 1e-5), "amp[\(k)]")
            // Phase is ill-defined where amplitude ≈ 0; only check significant bins.
            if refAmp[k] > 0.1 * ampMax {
                XCTAssertEqual(Float(phase[k]), refPhase[k], accuracy: 1e-3, "phase[\(k)]")
            }
        }

        // extractAmpPhase at 1 MHz.
        let eapAmp = Double(try f.readFloatDataset("eap_amp").data[0])
        let eapPhase = Double(try f.readFloatDataset("eap_phase").data[0])
        let eapF = Double(try f.readFloatDataset("eap_f").data[0])
        let (a, p, fr) = extractAmpPhase(sig, fs: fs, sourceFreq: Double(try f.readFloatDataset("eap_freq").data[0]))
        XCTAssertEqual(a, eapAmp, accuracy: abs(eapAmp) * 1e-4, "extract amp")
        XCTAssertEqual(p, eapPhase, accuracy: 1e-3, "extract phase")
        XCTAssertEqual(fr, eapF, accuracy: max(abs(eapF) * 1e-5, 1e-3), "extract freq")
    }
}

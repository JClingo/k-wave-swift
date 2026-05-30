import XCTest
import MLX
import MLXFFT
@testable import KWave

final class FFTTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    func testFFTRoundTrip() {
        let x = MLXArray(converting: (0..<32).map { sin(Double($0) * 0.3) })
        let restored = MLXFFT.ifft(MLXFFT.fft(x.asType(.complex64))).realPart()
        let err = MLX.max(MLX.abs(restored - x)).item(Float.self)
        XCTAssertLessThan(err, 1e-4)
    }

    func testSpectralDerivativeOfSine() {
        // f(x) = sin(2*pi*x/L) on a periodic domain → f'(x) = (2*pi/L) cos(2*pi*x/L)
        let n = 64
        let L = 1.0
        let dx = L / Double(n)
        let xs = (0..<n).map { Double($0) * dx }
        let f = MLXArray(converting: xs.map { sin(2 * Double.pi * $0 / L) })
        let expected = MLXArray(converting: xs.map { (2 * Double.pi / L) * cos(2 * Double.pi * $0 / L) })

        let kVec = MLXArray(converting: wavenumberVector(n, d: dx))
        let deriv = spectralDerivative(f, kVec: kVec, axis: 0)

        let err = MLX.max(MLX.abs(deriv - expected)).item(Float.self)
        XCTAssertLessThan(err, 1e-3)
    }
}

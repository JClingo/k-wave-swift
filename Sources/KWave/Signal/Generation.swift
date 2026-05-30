import Foundation

/// Sampled Gaussian distribution, mirroring MATLAB k-Wave `gaussian`.
///
/// - Parameters:
///   - x: sample points.
///   - magnitude: peak value. Defaults to the unit-area normalisation `1/sqrt(2*pi*variance)`.
///   - mean: distribution mean.
///   - variance: distribution variance.
public func gaussian(_ x: [Double], magnitude: Double? = nil, mean: Double = 0, variance: Double = 1) -> [Double] {
    let mag = magnitude ?? 1 / sqrt(2 * .pi * variance)
    return x.map { mag * exp(-($0 - mean) * ($0 - mean) / (2 * variance)) }
}

/// Envelope applied to a tone burst.
public enum ToneBurstEnvelope: Sendable {
    case gaussian
    case rectangular
}

/// Create an enveloped sinusoidal tone burst, mirroring MATLAB k-Wave `toneBurst`.
///
/// - Parameters:
///   - sampleFreq: sampling frequency [Hz].
///   - signalFreq: tone frequency [Hz].
///   - numCycles: number of sinusoid cycles in the burst.
///   - envelope: amplitude envelope (default Gaussian, matching k-Wave).
/// - Returns: the windowed burst samples.
public func toneBurst(
    sampleFreq: Double, signalFreq: Double, numCycles: Double,
    envelope: ToneBurstEnvelope = .gaussian
) -> [Double] {
    let dt = 1 / sampleFreq
    let toneLength = numCycles / signalFreq
    let count = Int(floor(toneLength / dt)) + 1
    let burst = (0..<count).map { sin(2 * .pi * signalFreq * Double($0) * dt) }

    let window: [Double]
    switch envelope {
    case .rectangular:
        window = [Double](repeating: 1, count: count)
    case .gaussian:
        let xLim = 3.0
        let step = count > 1 ? 2 * xLim / Double(count - 1) : 0
        let xs = (0..<count).map { -xLim + Double($0) * step }
        window = gaussian(xs, magnitude: 1, mean: 0, variance: 1)
    }
    return zip(burst, window).map(*)
}

import MLX
import MLXFFT

/// What the solver must record each step, derived from `sensor.record`. Pressure/velocity time
/// series are needed not just when requested directly but also when an aggregate (max/min/rms) or
/// intensity is requested (those are post-processed from the series). `pFinal` is always returned.
struct RecordPlan {
    let recordP: Bool        // accumulate the pressure time series.
    let recordU: Bool        // accumulate the (collocated) velocity time series.
    let recordUFinal: Bool   // keep the final velocity field(s).

    init(_ record: RecordField) {
        let pSeries: RecordField = [.p, .pMax, .pMin, .pRms, .iAvg]
        let uSeries: RecordField = [.ux, .uy, .uz, .uMax, .uRms, .iAvg]
        recordP = !record.isDisjoint(with: pSeries)
        recordU = !record.isDisjoint(with: uSeries)
        recordUFinal = record.contains(.uFinal)
    }

    /// Whether any per-step sensor sampling is needed (requires a sensor mask).
    var needsSampling: Bool { recordP || recordU }
}

/// `exp(-i·k·d/2)` multiplier that interpolates a staggered (mid-cell) velocity field back onto the
/// collocated pressure grid in k-space, matching k-wave-python's `unstagger_ops`.
func collocationOp(_ kVec: MLXArray, spacing: Double) -> MLXArray {
    let theta = kVec * (spacing / 2.0)
    return MLX.cos(theta) - imagUnit() * MLX.sin(theta)
}

/// Time-averaged acoustic intensity `mean_t(p · u_shifted)`, where the velocity time series is
/// shifted forward by half a sample (dt/2) along time via a Fourier interpolant to align with the
/// pressure samples. Mirrors k-wave-python `acoustic_intensity` (the `*_avg` outputs).
///
/// `p` and `u` are `[numSensorPoints, nt]`. The half-sample shift operator is built directly in
/// FFT-natural frequency order (`exp(i·ω_k·0.5)`, ω wrapped to [-π, π)); the Nyquist bin differs
/// from a literal `ifftshift` but is suppressed by the real part, so the result is identical.
func intensityAvg(p: MLXArray, u: MLXArray) -> MLXArray {
    let nt = p.dim(1)
    let half = nt / 2
    let omega = (0..<nt).map { k -> Float in
        let kk = k <= half ? k : k - nt
        return Float(2.0 * Double.pi) * Float(kk) / Float(nt)
    }
    let theta = MLXArray(omega) * Float(0.5)
    let shiftOp = (MLX.cos(theta) + imagUnit() * MLX.sin(theta)).reshaped([1, nt])
    let uShifted = MLXFFT.ifft(shiftOp * MLXFFT.fft(u.asType(.complex64), axis: 1), axis: 1).realPart()
    return MLX.mean(p * uShifted, axis: 1)
}

/// Assemble the `SimulationOutput` from the recorded time series and final fields, exposing only the
/// fields named in `record` (aggregates and intensity are computed here from the series). `pFinal`
/// is always set. Velocity arguments are `nil` for dimensions/fields that don't apply.
func finalizeRecording(
    record: RecordField,
    p: MLXArray?, ux: MLXArray?, uy: MLXArray?, uz: MLXArray?,
    pFinal: MLXArray, uxFinal: MLXArray?, uyFinal: MLXArray?, uzFinal: MLXArray?
) -> SimulationOutput {
    var out = SimulationOutput()
    out.pFinal = pFinal

    if record.contains(.p) { out.p = p }
    if let p {
        if record.contains(.pMax) { out.pMax = MLX.max(p, axis: 1) }
        if record.contains(.pMin) { out.pMin = MLX.min(p, axis: 1) }
        if record.contains(.pRms) { out.pRms = MLX.sqrt(MLX.mean(p * p, axis: 1)) }
    }

    if record.contains(.ux) { out.ux = ux }
    if record.contains(.uy) { out.uy = uy }
    if record.contains(.uz) { out.uz = uz }
    if record.contains(.uMax) {
        out.uxMax = ux.map { MLX.max($0, axis: 1) }
        out.uyMax = uy.map { MLX.max($0, axis: 1) }
        out.uzMax = uz.map { MLX.max($0, axis: 1) }
    }
    if record.contains(.uRms) {
        out.uxRms = ux.map { MLX.sqrt(MLX.mean($0 * $0, axis: 1)) }
        out.uyRms = uy.map { MLX.sqrt(MLX.mean($0 * $0, axis: 1)) }
        out.uzRms = uz.map { MLX.sqrt(MLX.mean($0 * $0, axis: 1)) }
    }
    if record.contains(.uFinal) {
        out.uxFinal = uxFinal
        out.uyFinal = uyFinal
        out.uzFinal = uzFinal
    }
    if record.contains(.iAvg), let p {
        out.ixAvg = ux.map { intensityAvg(p: p, u: $0) }
        out.iyAvg = uy.map { intensityAvg(p: p, u: $0) }
        out.izAvg = uz.map { intensityAvg(p: p, u: $0) }
    }
    return out
}

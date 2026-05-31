import MLX

/// Time-reversal photoacoustic reconstruction (mirrors k-wave-python `TimeReversal`).
///
/// Given pressure recorded at a sensor mask during a forward simulation, this re-injects that
/// data — time-reversed — as a Dirichlet pressure boundary condition on the same mask, runs the
/// solver forward, and returns the reconstructed initial pressure `p0`. The raw reconstruction is
/// scaled by `compensationFactor` (default 2.0, compensating for half-plane recording) and the
/// positivity condition (`p0 < 0 → 0`) is applied.
///
/// Time reversal is pure composition of existing primitives: the heavy lifting is the standard
/// Dirichlet pressure source (already validated in `ParitySourceTests`), so no new solver math is
/// involved.
///
/// - Parameters:
///   - recordedPressure: sensor pressure time series, shape `[numSensorPoints, nt]` — exactly the
///     `output.p` of a forward run whose sensor mask matches `sensor.mask` here (row order is the
///     ascending flat-index order of the mask, so the two line up point-for-point).
///   - sensor: must carry the recording `mask`; its other fields are ignored.
public func timeReversal(
    grid: KWaveGrid,
    medium: KWaveMedium,
    sensor: KWaveSensor,
    recordedPressure: MLXArray,
    compensationFactor: Double = 2.0,
    options: SimulationOptions = .init()
) -> MLXArray {
    precondition(sensor.mask != nil, "time reversal requires sensor.mask")
    precondition(grid.nt > 0, "set grid time (makeTime/setTime) before time reversal")
    precondition(recordedPressure.ndim == 2,
                 "recordedPressure must be [numSensorPoints, nt]")

    // Time-reverse the recorded data along the time axis (axis 1).
    let nt = recordedPressure.dim(1)
    let reversedTime = MLXArray((0..<nt).reversed().map { Int32($0) })
    let flipped = MLX.take(recordedPressure, reversedTime, axis: 1)

    // Re-inject as a Dirichlet pressure source on the sensor mask.
    var source = KWaveSource()
    source.pMask = sensor.mask
    source.p = flipped
    source.pMode = .dirichlet

    // Reconstruct over the whole grid (pFinal is the reconstruction estimate of p0).
    let out = kspaceFirstOrder(grid: grid, medium: medium, source: source,
                               sensor: KWaveSensor(), options: options)

    var p0 = out.pFinal! * Float(compensationFactor)
    p0 = MLX.maximum(p0, MLXArray(Float(0)))   // positivity condition
    return p0
}

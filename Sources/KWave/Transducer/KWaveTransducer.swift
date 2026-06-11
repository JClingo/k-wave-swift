import Foundation
import MLX

/// Physical geometry of a linear-array transducer (k-wave-python `kWaveTransducerSimple`).
/// The array lies in the plane `x = position.x`, with elements laid out along y (width +
/// kerf spacing per element) and extending along z (the elevation direction). Positions and
/// dimensions are in grid points, 0-based.
public struct KWaveTransducerGeometry {
    public let numberElements: Int
    public let elementWidth: Int       // grid points along y.
    public let elementLength: Int      // grid points along z (elevation).
    public let elementSpacing: Int     // kerf, grid points along y.
    public let position: (x: Int, y: Int, z: Int)
    public let dy: Double
    public let dz: Double

    public init(numberElements: Int, elementWidth: Int = 1, elementLength: Int = 20,
                elementSpacing: Int = 0, position: (x: Int, y: Int, z: Int) = (0, 0, 0),
                dy: Double, dz: Double) {
        precondition(numberElements > 0 && elementWidth > 0 && elementLength > 0
                     && elementSpacing >= 0, "invalid transducer geometry")
        self.numberElements = numberElements
        self.elementWidth = elementWidth
        self.elementLength = elementLength
        self.elementSpacing = elementSpacing
        self.position = position
        self.dy = dy
        self.dz = dz
    }

    /// Centre-to-centre element spacing [m].
    public var elementPitch: Double { Double(elementSpacing + elementWidth) * dy }
    /// Total transducer footprint along y in grid points.
    public var transducerWidth: Int {
        numberElements * elementWidth + (numberElements - 1) * elementSpacing
    }
}

/// Apodization specification for transmit/receive beamforming.
public enum TransducerApodization {
    case rectangular
    case hanning
    case custom([Double])
}

/// Active linear-array transducer model (k-wave-python `NotATransducer`): element masks,
/// transmit/receive beamforming delays (azimuth focus + steering, elevation focus), apodization,
/// drive-signal padding, and receive-side recombination.
///
/// All masks use this codebase's C-order flat-index convention (matching the solvers' source and
/// sensor row ordering), not MATLAB's F-order.
public final class KWaveTransducer {
    public let geometry: KWaveTransducerGeometry
    public let gridSize: [Int]                 // [Nx, Ny, Nz].
    public let dt: Double
    public let soundSpeed: Double
    public let activeElements: [Bool]
    public let focusDistance: Double           // [m]; .infinity = unfocused.
    public let elevationFocusDistance: Double  // [m]; .infinity = unfocused.
    public let steeringAngle: Double           // [deg].
    public let transmitApodization: TransducerApodization
    public let receiveApodization: TransducerApodization
    public let inputSignal: [Float]?

    /// Element number (1-based; 0 = empty) per grid voxel, C-order flat.
    private let indexedMask: [Int]
    /// Position along the element length (1...elementLength; 0 = empty) per voxel, C-order flat.
    private let voxelMask: [Int]

    public init(geometry: KWaveTransducerGeometry, gridSize: [Int], dt: Double,
                soundSpeed: Double = 1540, activeElements: [Bool]? = nil,
                focusDistance: Double = .infinity, elevationFocusDistance: Double = .infinity,
                steeringAngle: Double = 0,
                transmitApodization: TransducerApodization = .rectangular,
                receiveApodization: TransducerApodization = .rectangular,
                inputSignal: [Float]? = nil) {
        precondition(gridSize.count == 3, "transducers require a 3D grid")
        let (nx, ny, nz) = (gridSize[0], gridSize[1], gridSize[2])
        precondition(geometry.position.x < nx, "transducer outside the grid in x")
        precondition(geometry.position.y + geometry.transducerWidth <= ny,
                     "transducer too large or outside the grid in y")
        precondition(geometry.position.z + geometry.elementLength <= nz,
                     "transducer too large or outside the grid in z")

        self.geometry = geometry
        self.gridSize = gridSize
        self.dt = dt
        self.soundSpeed = soundSpeed
        self.activeElements = activeElements ?? [Bool](repeating: true,
                                                       count: geometry.numberElements)
        precondition(self.activeElements.count == geometry.numberElements,
                     "activeElements must have one entry per element")
        self.focusDistance = focusDistance
        self.elevationFocusDistance = elevationFocusDistance
        self.steeringAngle = steeringAngle
        self.transmitApodization = transmitApodization
        self.receiveApodization = receiveApodization
        self.inputSignal = inputSignal

        var indexed = [Int](repeating: 0, count: nx * ny * nz)
        var voxels = [Int](repeating: 0, count: nx * ny * nz)
        let px = geometry.position.x, pz = geometry.position.z
        for e in 0..<geometry.numberElements {
            let y0 = geometry.position.y + (geometry.elementWidth + geometry.elementSpacing) * e
            for w in 0..<geometry.elementWidth {
                for l in 0..<geometry.elementLength {
                    let flat = (px * ny + y0 + w) * nz + pz + l
                    indexed[flat] = e + 1
                    voxels[flat] = l + 1
                }
            }
        }
        self.indexedMask = indexed
        self.voxelMask = voxels
    }

    public var numberActiveElements: Int { activeElements.filter { $0 }.count }

    // MARK: - Masks

    /// Binary mask over all elements, active or not.
    public var allElementsMask: MLXArray {
        MLXArray(indexedMask.map { Float($0 != 0 ? 1 : 0) }).reshaped(gridSize)
    }

    /// Binary mask over the active elements — the mask to use as the solver's source/sensor mask.
    public var activeElementsMask: MLXArray {
        MLXArray(indexedMask.map { Float($0 != 0 && activeElements[$0 - 1] ? 1 : 0) })
            .reshaped(gridSize)
    }

    /// Active-element number (renumbered from 1 at the lowest active element; 0 = empty) per voxel.
    var indexedActiveElementsMask: [Int] {
        guard let lowest = activeElements.firstIndex(of: true) else {
            return [Int](repeating: 0, count: indexedMask.count)
        }
        return indexedMask.map { e in
            e != 0 && activeElements[e - 1] ? e - lowest : 0
        }
    }

    // MARK: - Beamforming

    /// Per-active-element transmit delays in time samples (focus + steering; k-Wave
    /// `beamforming_delays`).
    public var beamformingDelays: [Int] {
        let n = numberActiveElements
        let pitch = geometry.elementPitch
        let theta = steeringAngle * .pi / 180
        return (0..<n).map { i in
            let index = Double(i) - Double(n - 1) / 2
            let seconds: Double
            if focusDistance.isInfinite {
                seconds = pitch * index * sin(theta) / soundSpeed
            } else {
                let q = index * pitch / focusDistance
                seconds = focusDistance / soundSpeed * (1 - (1 + q * q - 2 * q * sin(theta)).squareRoot())
            }
            return Int((seconds / dt).rounded(.toNearestOrEven))
        }
    }

    /// Per-voxel elevation-focusing delays in time samples along the element length (k-Wave
    /// `elevation_beamforming_delays`).
    public var elevationBeamformingDelays: [Int] {
        guard !elevationFocusDistance.isInfinite else {
            return [Int](repeating: 0, count: geometry.elementLength)
        }
        let n = geometry.elementLength
        return (0..<n).map { i in
            let index = Double(i) - Double(n - 1) / 2
            let dist = (pow(index * geometry.dz, 2) + elevationFocusDistance * elevationFocusDistance)
                .squareRoot()
            let seconds = (elevationFocusDistance - dist) / soundSpeed
            return -Int((seconds / dt).rounded(.toNearestOrEven))
        }
    }

    /// Which delay components `delayMask` includes.
    public enum DelayMaskMode { case both, elevationOnly, azimuthOnly }

    /// Per-voxel transmit delay in time samples over the active elements (k-Wave `delay_mask`):
    /// azimuth beamforming per element plus elevation focusing per voxel, shifted so the minimum
    /// active-voxel delay is zero.
    public func delayMask(mode: DelayMaskMode = .both) -> [Int] {
        var mask = [Int](repeating: 0, count: indexedMask.count)
        let activeIndex = indexedActiveElementsMask

        if (!focusDistance.isInfinite || steeringAngle != 0) && mode != .elevationOnly {
            let delays = beamformingDelays.map { -$0 }
            for (flat, e) in activeIndex.enumerated() where e != 0 {
                mask[flat] = delays[e - 1]
            }
        }
        if !elevationFocusDistance.isInfinite && geometry.elementLength > 1 && mode != .azimuthOnly {
            let elev = elevationBeamformingDelays
            for (flat, e) in activeIndex.enumerated() where e != 0 {
                mask[flat] += elev[voxelMask[flat] - 1]
            }
        }
        // Shift so the smallest active delay is zero.
        var minDelay = Int.max
        for (flat, e) in activeIndex.enumerated() where e != 0 {
            minDelay = min(minDelay, mask[flat])
        }
        if minDelay != Int.max && minDelay != 0 {
            for (flat, e) in activeIndex.enumerated() where e != 0 { mask[flat] -= minDelay }
        }
        return mask
    }

    // MARK: - Apodization

    private func apodizationValues(_ spec: TransducerApodization) -> [Double] {
        let n = numberActiveElements
        switch spec {
        case .rectangular:
            return [Double](repeating: 1, count: n)
        case .hanning:
            guard n > 1 else { return [1] }
            return (0..<n).map { 0.5 - 0.5 * cos(2 * .pi * Double($0) / Double(n - 1)) }
        case let .custom(values):
            precondition(values.count == n, "apodization length must match active elements")
            return values
        }
    }

    public func getTransmitApodization() -> [Double] { apodizationValues(transmitApodization) }
    public func getReceiveApodization() -> [Double] { apodizationValues(receiveApodization) }

    /// Transmit apodization scattered onto the active-element grid points (k-Wave
    /// `transmit_apodization_mask`).
    public var transmitApodizationMask: MLXArray {
        let apod = getTransmitApodization()
        let activeIndex = indexedActiveElementsMask
        var mask = [Float](repeating: 0, count: indexedMask.count)
        for (flat, e) in activeIndex.enumerated() where e != 0 {
            mask[flat] = Float(apod[e - 1])
        }
        return MLXArray(mask).reshaped(gridSize)
    }

    // MARK: - Signals

    /// The drive signal with enough leading/trailing zeros prepended and appended to accommodate
    /// the maximum transmit delay (k-Wave `input_signal`).
    public func paddedInputSignal() -> [Float] {
        guard var signal = inputSignal else {
            preconditionFailure("Transducer input signal is not defined")
        }
        let delayMax = delayMask().max() ?? 0
        let leading = signal.firstIndex(where: { $0 != 0 }) ?? signal.count
        let trailing = signal.reversed().firstIndex(where: { $0 != 0 }) ?? signal.count
        if leading < delayMax + 1 {
            signal = [Float](repeating: 0, count: delayMax - leading + 1) + signal
        }
        if trailing < delayMax + 1 {
            signal += [Float](repeating: 0, count: delayMax - trailing + 1)
        }
        return signal
    }

    /// Combine per-grid-point sensor data (rows = active-mask voxels in ascending C-flat order)
    /// into per-element series, compensating the elevation-focusing delays (k-Wave
    /// `combine_sensor_data`; requires elevation focusing).
    public func combineSensorData(_ sensorData: MLXArray) -> MLXArray {
        precondition(!elevationFocusDistance.isInfinite,
                     "combineSensorData requires elevation focusing")
        let nt = sensorData.dim(1)
        let expected = numberActiveElements * geometry.elementWidth * geometry.elementLength
        precondition(sensorData.dim(0) == expected,
                     "sensorData rows must match the active transducer grid points")
        let data = sensorData.reshaped([expected * nt]).asArray(Float.self)

        let activeIndex = indexedActiveElementsMask
        let dm = delayMask(mode: .elevationOnly)
        var out = [Float](repeating: 0, count: numberActiveElements * nt)
        var row = 0
        for (flat, e) in activeIndex.enumerated() where e != 0 {
            let d = dm[flat]
            for t in 0..<(nt - d) {
                out[(e - 1) * nt + t] += data[row * nt + t + d]
            }
            row += 1
        }
        let scale = Float(1) / Float(geometry.elementWidth * geometry.elementLength)
        for i in 0..<out.count { out[i] *= scale }
        return MLXArray(out).reshaped([numberActiveElements, nt])
    }

    /// Receive beamforming: undo the per-element transmit delays, apply the receive apodization,
    /// and sum the per-element series into a single scan line (k-Wave `scan_line`).
    public func scanLine(_ sensorData: MLXArray) -> MLXArray {
        let n = numberActiveElements
        precondition(sensorData.ndim == 2 && sensorData.dim(0) == n,
                     "sensorData must be [numberActiveElements, Nt]")
        let nt = sensorData.dim(1)
        let data = sensorData.reshaped([n * nt]).asArray(Float.self)
        let apod = getReceiveApodization()
        let delays = beamformingDelays.map { -$0 }

        var line = [Float](repeating: 0, count: nt)
        for e in 0..<n {
            let d = delays[e]
            let a = Float(apod[e])
            if d > 0 {
                for t in 0..<(nt - d) { line[t] += data[e * nt + t + d] * a }
            } else if d < 0 {
                for t in 0..<(nt + d) { line[t - d] += data[e * nt + t] * a }
            } else {
                for t in 0..<nt { line[t] += data[e * nt + t] * a }
            }
        }
        return MLXArray(line)
    }
}

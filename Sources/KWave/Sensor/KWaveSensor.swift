import MLX

/// Fields that a sensor can record.
public struct RecordField: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    public static let p       = RecordField(rawValue: 1 << 0)
    public static let pMax    = RecordField(rawValue: 1 << 1)
    public static let pMin    = RecordField(rawValue: 1 << 2)
    public static let pRms    = RecordField(rawValue: 1 << 3)
    public static let pFinal  = RecordField(rawValue: 1 << 4)
    public static let ux      = RecordField(rawValue: 1 << 5)
    public static let uy      = RecordField(rawValue: 1 << 6)
    public static let uz      = RecordField(rawValue: 1 << 7)
    public static let uMax    = RecordField(rawValue: 1 << 8)
    public static let uRms    = RecordField(rawValue: 1 << 9)
    public static let uFinal  = RecordField(rawValue: 1 << 10)
    public static let iAvg    = RecordField(rawValue: 1 << 11)
    public static let iMax    = RecordField(rawValue: 1 << 12)
}

/// 2D sensor directivity pattern (mirrors k-Wave `sensor.directivity_pattern`).
public enum DirectivityPattern: Sendable {
    /// Spatial averaging over the element face — a `sinc` response.
    case pressure
    /// Pressure-gradient detector response.
    case gradient
}

/// Sensor / detector definition.
public struct KWaveSensor {
    /// Binary mask (grid-shaped) selecting recording points, or Cartesian point coordinates.
    public var mask: MLXArray?
    public var record: RecordField
    public var timeReversalBoundaryData: MLXArray?
    /// Grid-shaped matrix giving the direction of maximum response (radians) at each sensor point;
    /// enables 2D sensor directivity when set.
    public var directivityAngle: MLXArray?
    /// Equivalent element size (m); larger = more directional. Defaults to `10·max(dx,dy)`.
    public var directivitySize: Double?
    public var directivityPattern: DirectivityPattern
    public var frequencyResponse: (centerFreq: Double, bandwidth: Double)?

    public init(
        mask: MLXArray? = nil,
        record: RecordField = [.p],
        timeReversalBoundaryData: MLXArray? = nil,
        directivityAngle: MLXArray? = nil,
        directivitySize: Double? = nil,
        directivityPattern: DirectivityPattern = .pressure,
        frequencyResponse: (centerFreq: Double, bandwidth: Double)? = nil
    ) {
        self.mask = mask
        self.record = record
        self.timeReversalBoundaryData = timeReversalBoundaryData
        self.directivityAngle = directivityAngle
        self.directivitySize = directivitySize
        self.directivityPattern = directivityPattern
        self.frequencyResponse = frequencyResponse
    }
}

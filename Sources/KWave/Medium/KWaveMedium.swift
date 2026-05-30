import MLX

/// Acoustic absorption model.
public enum AbsorptionMode: Sendable {
    case noAbsorption
    case noDispersion
    case stokes
}

/// Acoustic medium properties. Scalars or spatially varying maps are both `MLXArray`
/// (a scalar is a 0-d / single-element array).
public struct KWaveMedium {
    public var soundSpeed: MLXArray          // [m/s]
    public var density: MLXArray             // [kg/m³]
    public var alphaCoeff: MLXArray?         // power-law absorption prefactor
    public var alphaPower: Double?           // power-law exponent
    public var alphaMode: AbsorptionMode
    public var bOnA: MLXArray?               // nonlinearity parameter B/A

    // Elastic media (Phase 4)
    public var soundSpeedCompression: MLXArray?
    public var soundSpeedShear: MLXArray?

    public init(
        soundSpeed: MLXArray,
        density: MLXArray,
        alphaCoeff: MLXArray? = nil,
        alphaPower: Double? = nil,
        alphaMode: AbsorptionMode = .noAbsorption,
        bOnA: MLXArray? = nil,
        soundSpeedCompression: MLXArray? = nil,
        soundSpeedShear: MLXArray? = nil
    ) {
        self.soundSpeed = soundSpeed
        self.density = density
        self.alphaCoeff = alphaCoeff
        self.alphaPower = alphaPower
        self.alphaMode = alphaMode
        self.bOnA = bOnA
        self.soundSpeedCompression = soundSpeedCompression
        self.soundSpeedShear = soundSpeedShear
    }

    /// Convenience for homogeneous media.
    public init(
        soundSpeed: Double,
        density: Double,
        alphaCoeff: Double? = nil,
        alphaPower: Double? = nil,
        alphaMode: AbsorptionMode = .noAbsorption,
        bOnA: Double? = nil
    ) {
        self.init(
            soundSpeed: MLXArray(soundSpeed),
            density: MLXArray(density),
            alphaCoeff: alphaCoeff.map { MLXArray($0) },
            alphaPower: alphaPower,
            alphaMode: alphaMode,
            bOnA: bOnA.map { MLXArray($0) }
        )
    }

    /// True when the medium is acoustically homogeneous (scalar sound speed and density).
    public var isHomogeneous: Bool {
        soundSpeed.size == 1 && density.size == 1
    }
}

import MLX

/// How a time-varying source is injected into the field.
public enum SourceMode: Sendable {
    case dirichlet
    case additive
    case additiveNoCorrection
}

/// Source definitions: initial pressure, time-varying pressure/velocity, and (elastic) stress.
public struct KWaveSource {
    // Initial value (photoacoustic)
    public var p0: MLXArray?

    // Time-varying pressure source
    public var pMask: MLXArray?
    public var p: MLXArray?
    public var pMode: SourceMode

    // Time-varying velocity source
    public var uMask: MLXArray?
    public var ux: MLXArray?
    public var uy: MLXArray?
    public var uz: MLXArray?
    public var uMode: SourceMode

    // Stress source (elastic)
    public var sMask: MLXArray?
    public var sxx: MLXArray?
    public var syy: MLXArray?
    public var szz: MLXArray?
    public var sxy: MLXArray?
    public var sxz: MLXArray?
    public var syz: MLXArray?

    public init(
        p0: MLXArray? = nil,
        pMask: MLXArray? = nil,
        p: MLXArray? = nil,
        pMode: SourceMode = .additive,
        uMask: MLXArray? = nil,
        ux: MLXArray? = nil,
        uy: MLXArray? = nil,
        uz: MLXArray? = nil,
        uMode: SourceMode = .additive,
        sMask: MLXArray? = nil,
        sxx: MLXArray? = nil,
        syy: MLXArray? = nil,
        szz: MLXArray? = nil,
        sxy: MLXArray? = nil,
        sxz: MLXArray? = nil,
        syz: MLXArray? = nil
    ) {
        self.p0 = p0
        self.pMask = pMask
        self.p = p
        self.pMode = pMode
        self.uMask = uMask
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.uMode = uMode
        self.sMask = sMask
        self.sxx = sxx
        self.syy = syy
        self.szz = szz
        self.sxy = sxy
        self.sxz = sxz
        self.syz = syz
    }
}

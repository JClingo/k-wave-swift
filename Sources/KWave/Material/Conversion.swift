import Foundation

/// Convert an absorption coefficient from dB/(MHz^y cm) to Np/((rad/s)^y m).
///
/// Mirrors MATLAB k-Wave `db2neper`. `y` is the power-law exponent (default 1).
public func db2neper(_ alpha: Double, y: Double = 1) -> Double {
    100 * alpha * pow(1e-6 / (2 * .pi), y) / (20 * log10(M_E))
}

/// Convert an absorption coefficient from Np/((rad/s)^y m) to dB/(MHz^y cm).
///
/// Mirrors MATLAB k-Wave `neper2db`. `y` is the power-law exponent (default 1).
public func neper2db(_ alpha: Double, y: Double = 1) -> Double {
    20 * log10(M_E) * alpha * pow(2 * .pi * 1e6, y) / 100
}

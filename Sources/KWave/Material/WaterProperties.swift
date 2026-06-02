import Foundation

/// Horner evaluation of a polynomial with coefficients highest-degree first (like `numpy.polyval`).
private func polyval(_ p: [Double], _ x: Double) -> Double {
    p.reduce(0) { $0 * x + $1 }
}

/// Sound speed in distilled water [m/s] as a function of temperature [°C], per Marczak (1997).
/// Valid for `temp` in 0…95 °C.
public func waterSoundSpeed(_ temp: Double) -> Double {
    precondition(temp >= 0 && temp <= 95, "temp must be between 0 and 95 °C")
    return polyval([2.787860e-9, -1.398845e-6, 3.287156e-4, -5.779136e-2, 5.038813, 1.402385e3], temp)
}

/// Density of air-saturated water [kg/m³] as a function of temperature [°C], per Jones & Harris
/// (1992). Valid for `temp` in 5…40 °C.
public func waterDensity(_ temp: Double) -> Double {
    precondition(temp >= 5 && temp <= 40, "temp must be between 5 and 40 °C")
    return 999.84847 + 6.337563e-2 * temp - 8.523829e-3 * temp * temp
        + 6.943248e-5 * pow(temp, 3) - 3.821216e-7 * pow(temp, 4)
}

/// Ultrasonic absorption in distilled water [dB/cm] at frequency `f` [MHz] and temperature `temp`
/// [°C], from a 7th-order polynomial fit to Pinkerton (1949). Valid for `temp` in 0…60 °C.
public func waterAbsorption(f: Double, temp: Double) -> Double {
    precondition(temp >= 0 && temp <= 60, "temp must be between 0 and 60 °C")
    let neper2db = 8.686
    // Coefficients in increasing powers of temperature.
    let a = [56.723531840522710, -2.899633796917384, 0.099253401567561, -0.002067402501557,
             2.189417428917596e-5, -6.210860973978427e-8, -6.402634551821596e-10, 3.869387679459408e-12]
    let aOnFsqr = polyval(a.reversed(), temp) * 1e-17
    return neper2db * 1e12 * f * f * aOnFsqr
}

/// Parameter of nonlinearity B/A of water as a function of temperature [°C], from a 4th-order
/// polynomial fit to Beyer (1960). Valid for `temp` in 0…100 °C.
public func waterNonlinearity(_ temp: Double) -> Double {
    precondition(temp >= 0 && temp <= 100, "temp must be between 0 and 100 °C")
    return polyval([-4.587913769504693e-8, 1.047843302423604e-5, -9.355518377254833e-4,
                    5.380874771364909e-2, 4.186533937275504], temp)
}

import MLX

/// Solver numeric precision. float64 runs only on the CPU stream (Metal has no double).
public enum DTypePrecision: Sendable {
    case float32
    case float64

    var mlx: DType { self == .float32 ? .float32 : .float64 }
}

/// Compute device selection for the solver.
public enum DeviceKind: Sendable {
    case gpu
    case cpu

    var stream: StreamOrDevice {
        switch self {
        case .gpu: return .gpu
        case .cpu: return .cpu
        }
    }
}

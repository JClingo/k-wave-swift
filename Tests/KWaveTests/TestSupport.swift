import MLX

/// MLX's precompiled Metal shader library is not produced by `swift build`/`swift test` on the
/// command line (only Xcode builds it), so GPU evaluation fails to load the default metallib.
/// Force the CPU backend for all tests run via SwiftPM. GPU paths must be validated in Xcode.
func useCPUBackend() {
    Device.setDefault(device: Device(.cpu))
}

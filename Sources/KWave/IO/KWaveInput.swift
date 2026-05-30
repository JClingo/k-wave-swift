import Foundation
import MLX

/// Assembles a k-Wave C++/CUDA-compatible HDF5 input file for a 2D simulation and (optionally)
/// drives the `kspaceFirstOrder-OMP` binary. The dataset names, `[Nz, Ny, Nx]` transposed layout,
/// uint64 integer / float32 real types, and root version attributes (1.2) mirror what
/// k-wave-python's `save_to_disk` writes, so the same OMP binary accepts both.
public enum KWaveCpp {

    /// Write a full 2D input file for a homogeneous, linear, lossless simulation with an
    /// initial-pressure (`p0`) source and a binary sensor mask.
    public static func writeInput2D(
        path: String,
        grid: KWaveGrid,
        medium: KWaveMedium,
        source: KWaveSource,
        sensor: KWaveSensor,
        pmlSize: Int,
        pmlAlpha: Double
    ) throws {
        precondition(grid.dim == 2, "writeInput2D requires a 2D grid")
        precondition(source.p0 != nil, "writeInput2D currently supports a p0 source only")
        precondition(sensor.mask != nil, "writeInput2D requires a binary sensor mask")

        let nx = grid.nx, ny = grid.ny
        let c0 = Float(medium.soundSpeed.item(Float.self))
        let rho0 = Float(medium.density.item(Float.self))

        let f = try HDF5File(create: path)
        try f.writeRootAttributes(fileType: .input, createdBy: "k-wave-swift",
                                  fileDescription: "k-Wave input file (k-wave-swift)")

        // --- Integer (uint64) scalars ---
        try f.writeScalarUInt64("Nx", UInt64(nx))
        try f.writeScalarUInt64("Ny", UInt64(ny))
        try f.writeScalarUInt64("Nz", 1)
        try f.writeScalarUInt64("Nt", UInt64(grid.nt))
        // Source flags (only p0 active).
        try f.writeScalarUInt64("p0_source_flag", 1)
        try f.writeScalarUInt64("p_source_flag", 0)
        try f.writeScalarUInt64("ux_source_flag", 0)
        try f.writeScalarUInt64("uy_source_flag", 0)
        try f.writeScalarUInt64("uz_source_flag", 0)
        try f.writeScalarUInt64("sxx_source_flag", 0)
        try f.writeScalarUInt64("syy_source_flag", 0)
        try f.writeScalarUInt64("szz_source_flag", 0)
        try f.writeScalarUInt64("sxy_source_flag", 0)
        try f.writeScalarUInt64("sxz_source_flag", 0)
        try f.writeScalarUInt64("syz_source_flag", 0)
        try f.writeScalarUInt64("transducer_source_flag", 0)
        try f.writeScalarUInt64("nonuniform_grid_flag", 0)
        try f.writeScalarUInt64("nonlinear_flag", 0)
        try f.writeScalarUInt64("absorbing_flag", 0)
        try f.writeScalarUInt64("elastic_flag", 0)
        try f.writeScalarUInt64("axisymmetric_flag", 0)
        try f.writeScalarUInt64("sensor_mask_type", 0)
        try f.writeScalarUInt64("pml_x_size", UInt64(pmlSize))
        try f.writeScalarUInt64("pml_y_size", UInt64(pmlSize))
        try f.writeScalarUInt64("pml_z_size", 0)

        // --- Float (float32) scalars (z-variables omitted in 2D) ---
        try f.writeScalarFloat("dx", Float(grid.dx))
        try f.writeScalarFloat("dy", Float(grid.dy))
        try f.writeScalarFloat("pml_x_alpha", Float(pmlAlpha))
        try f.writeScalarFloat("pml_y_alpha", Float(pmlAlpha))
        try f.writeScalarFloat("dt", Float(grid.dt))
        try f.writeScalarFloat("c0", c0)
        try f.writeScalarFloat("c_ref", c0)          // homogeneous: c_ref = max(c0) = c0
        try f.writeScalarFloat("rho0", rho0)
        try f.writeScalarFloat("rho0_sgx", rho0)     // constant density ⇒ staggered values equal
        try f.writeScalarFloat("rho0_sgy", rho0)

        // --- Fields ---
        let p0Host = source.p0!.asType(.float32).reshaped([nx * ny]).asArray(Float.self)
        try f.writeFieldDataset2D("p0_source_input", nx: nx, ny: ny, data: p0Host)

        // Sensor mask indices: 1-based MATLAB column-major linear indices of nonzero mask points.
        let maskHost = sensor.mask!.reshaped([nx * ny]).asArray(Float.self)
        var indices: [UInt64] = []
        indices.reserveCapacity(nx * ny)
        for j in 0..<ny {           // column-major: column j outer, row i inner
            for i in 0..<nx {
                if maskHost[i * ny + j] != 0 { indices.append(UInt64(j * nx + i + 1)) }
            }
        }
        try f.writeUInt64Vector("sensor_mask_index", indices)
    }

    /// Read `p_final` (shape `[Nx, Ny]`, transposed back from the on-disk `[1, Ny, Nx]`) from a
    /// k-Wave++ output file.
    public static func readPFinal2D(path: String, nx: Int, ny: Int) throws -> MLXArray {
        let f = try HDF5File(open: path)
        let (shape, disk) = try f.readFloatDataset("p_final")
        precondition(shape.suffix(2) == [ny, nx], "unexpected p_final shape \(shape)")
        // On disk: disk[i*nx + j] = field[j, i]  (i over Ny, j over Nx). Undo the transpose.
        var host = [Float](repeating: 0, count: nx * ny)
        for j in 0..<nx { for i in 0..<ny { host[j * ny + i] = disk[i * nx + j] } }
        return MLXArray(host).reshaped([nx, ny])
    }

    /// Run the `kspaceFirstOrder-OMP` binary on `inputPath`, writing results to `outputPath`.
    /// Records only the final pressure field (`--p_final`). Throws if the binary is missing or exits
    /// non-zero.
    @discardableResult
    public static func runOMP(
        binary: String, inputPath: String, outputPath: String,
        recordStartIndex: Int = 1, extraArgs: [String] = []
    ) throws -> String {
        guard FileManager.default.fileExists(atPath: binary) else {
            throw HDF5Error.openFailed("OMP binary not found at \(binary)")
        }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: binary)
        proc.arguments = ["-i", inputPath, "-o", outputPath,
                          "--p_final", "-s", String(recordStartIndex)] + extraArgs
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        try proc.run()
        proc.waitUntilExit()
        let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        guard proc.terminationStatus == 0 else {
            throw HDF5Error.writeFailed("OMP exited \(proc.terminationStatus): \(out)")
        }
        return out
    }
}

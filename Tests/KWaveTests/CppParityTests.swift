import XCTest
import MLX
@testable import KWave

/// End-to-end parity for the k-Wave C++ (OMP) path: assemble a 2D input file with Swift, run the
/// `kspaceFirstOrder-OMP` binary on it, and compare its `p_final` to the golden output produced by
/// k-wave-python's own input writer driving the same binary. This validates that the Swift HDF5
/// input assembler is byte-faithful enough for the unmodified C++ engine.
///
/// Regenerate golden artifacts with:
///   `.venv-kwave/bin/python Scripts/parity/generate_reference_cpp.py`
final class CppParityTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    private func repoRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    func testSwiftInputDrivesOMPBinary() throws {
        let root = repoRoot()
        let metaPath = root.appendingPathComponent("Scripts/parity/reference_cpp_meta.h5").path
        let binary = root.appendingPathComponent(
            ".venv-kwave/lib/python3.13/site-packages/kwave/bin/darwin/kspaceFirstOrder-OMP").path

        guard FileManager.default.fileExists(atPath: metaPath) else {
            throw XCTSkip("golden not generated: run Scripts/parity/generate_reference_cpp.py")
        }
        guard FileManager.default.fileExists(atPath: binary) else {
            throw XCTSkip("OMP binary not available at \(binary)")
        }

        let meta = try HDF5File(open: metaPath)
        let nx = Int(try meta.readFloatDataset("Nx").data[0])
        let ny = Int(try meta.readFloatDataset("Ny").data[0])
        let dx = Double(try meta.readFloatDataset("dx").data[0])
        let c0 = Double(try meta.readFloatDataset("c0").data[0])
        let rho0 = Double(try meta.readFloatDataset("rho0").data[0])
        let dt = Double(try meta.readFloatDataset("dt").data[0])
        let nt = Int(try meta.readFloatDataset("Nt").data[0])
        let pml = Int(try meta.readFloatDataset("pml_size").data[0])
        let (_, p0Data) = try meta.readFloatDataset("p0")
        let (goldShape, goldData) = try meta.readFloatDataset("p_final")

        var grid = KWaveGrid(nx: nx, dx: dx, ny: ny, dy: dx)
        grid.makeTime(soundSpeedMax: c0, cfl: 0.3)
        grid.setTime(nt: nt, dt: dt)

        var source = KWaveSource()
        source.p0 = MLXArray(p0Data).reshaped([nx, ny])
        var sensor = KWaveSensor()
        sensor.mask = MLX.ones([nx, ny])

        let tmp = FileManager.default.temporaryDirectory
        let inPath = tmp.appendingPathComponent("kwave_swift_in_\(UUID().uuidString).h5").path
        let outPath = tmp.appendingPathComponent("kwave_swift_out_\(UUID().uuidString).h5").path
        defer {
            try? FileManager.default.removeItem(atPath: inPath)
            try? FileManager.default.removeItem(atPath: outPath)
        }

        try KWaveCpp.writeInput2D(path: inPath, grid: grid,
                                  medium: KWaveMedium(soundSpeed: c0, density: rho0),
                                  source: source, sensor: sensor,
                                  pmlSize: pml, pmlAlpha: 2.0)
        try KWaveCpp.runOMP(binary: binary, inputPath: inPath, outputPath: outPath)

        let mine = try KWaveCpp.readPFinal2D(path: outPath, nx: nx, ny: ny)
        // Golden p_final stored row-major [Nx, Ny] in meta.
        let gold = MLXArray(goldData).reshaped(goldShape.count == 2 ? goldShape : [nx, ny])
        let diff = MLX.abs(mine - gold)
        let maxDiff = MLX.max(diff).item(Float.self)
        let maxRef = MLX.max(MLX.abs(gold)).item(Float.self)
        // Same binary, same physics: results should match to float round-off.
        XCTAssertLessThan(maxDiff / maxRef, 1e-5, "Swift-driven OMP p_final differs from golden")
    }
}

import XCTest
import AVFoundation
import MLX
@testable import KWave

/// Visualization layer: field→RGBA mapping, the solver field-monitor hook, and movie recording.
final class VizTests: XCTestCase {
    override func setUp() { useCPUBackend() }

    /// Extremes and centre of a fixed-scale field map to the first, last, and middle colormap
    /// entries; image dimensions follow the matrix (width = ny, height = nx).
    func testFieldToRGBAMapping() {
        let map = getColorMap(numColors: 256)
        let field = MLXArray([Float(-1), 0, 1, -1]).reshaped([2, 2])
        let (pixels, width, height) = fieldToRGBA(field, colorMap: map,
                                                  plotScale: .fixed(-1, 1))
        XCTAssertEqual(width, 2)
        XCTAssertEqual(height, 2)
        XCTAssertEqual(pixels.count, 16)

        func rgb(_ i: Int) -> (UInt8, UInt8, UInt8) {
            (pixels[i * 4], pixels[i * 4 + 1], pixels[i * 4 + 2])
        }
        func expected(_ c: RGB) -> (UInt8, UInt8, UInt8) {
            (UInt8(c.r * 255 + 0.5), UInt8(c.g * 255 + 0.5), UInt8(c.b * 255 + 0.5))
        }
        XCTAssertEqual(rgb(0).0, expected(map[0]).0, "min → first colormap entry")
        XCTAssertEqual(rgb(2).0, expected(map[255]).0, "max → last colormap entry")
        // Centre value (0 in a symmetric scale) lands mid-map — white-ish in the k-Wave map.
        let mid = rgb(1)
        XCTAssertGreaterThan(mid.0, 200)
        XCTAssertGreaterThan(mid.1, 200)
        XCTAssertGreaterThan(mid.2, 200)

        // Auto scale of the same field is identical (max|field| = 1).
        let (autoPixels, _, _) = fieldToRGBA(field, colorMap: map, plotScale: .auto)
        XCTAssertEqual(autoPixels, pixels)
    }

    /// The solver invokes `fieldMonitor` every `fieldMonitorInterval` steps with the live field.
    func testFieldMonitorHook() {
        var grid = KWaveGrid(nx: 32, dx: 1e-4, ny: 32, dy: 1e-4)
        grid.setTime(nt: 20, dt: 2e-8)
        var source = KWaveSource()
        source.p0 = makeDisc(nx: 32, ny: 32, radius: 4)

        var calls: [Int] = []
        var shapes: Set<[Int]> = []
        var options = SimulationOptions()
        options.fieldMonitorInterval = 5
        options.fieldMonitor = { t, p in
            calls.append(t)
            shapes.insert(p.shape)
        }
        _ = kspaceFirstOrder(grid: grid, medium: KWaveMedium(soundSpeed: 1500, density: 1000),
                             source: source, sensor: KWaveSensor(), options: options)
        XCTAssertEqual(calls, [0, 5, 10, 15])
        XCTAssertEqual(shapes, [[32, 32]])
    }

    /// MovieRecorder writes a playable H.264 file with the requested dimensions and duration.
    func testMovieRecorderWritesFile() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kwave_viz_test_\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: url) }

        let recorder = try MovieRecorder(url: url, width: 64, height: 48, fps: 24)
        for t in 0..<12 {
            var host = [Float](repeating: 0, count: 48 * 64)
            for i in 0..<host.count {
                let phase = Double(i % 64) * 0.2 + Double(t) * 0.5
                host[i] = Float(sin(phase))
            }
            let field = MLXArray(host).reshaped([48, 64])
            try recorder.append(fieldToImage(field))
        }
        try recorder.finish()

        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration)
        XCTAssertEqual(duration.seconds, 0.5, accuracy: 0.05, "12 frames at 24 fps")
        let tracks = try await asset.loadTracks(withMediaType: .video)
        XCTAssertEqual(tracks.count, 1)
        let size = try await tracks[0].load(.naturalSize)
        XCTAssertEqual(Int(size.width), 64)
        XCTAssertEqual(Int(size.height), 48)
    }

    /// The `recordMovie` option produces a movie alongside normal solver output.
    func testRecordMovieOption() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kwave_sim_movie_\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: url) }

        var grid = KWaveGrid(nx: 32, dx: 1e-4, ny: 32, dy: 1e-4)
        grid.setTime(nt: 20, dt: 2e-8)
        var source = KWaveSource()
        source.p0 = makeDisc(nx: 32, ny: 32, radius: 4)
        var options = SimulationOptions()
        options.recordMovie = url.path
        options.fieldMonitorInterval = 2

        let out = kspaceFirstOrder(grid: grid,
                                   medium: KWaveMedium(soundSpeed: 1500, density: 1000),
                                   source: source, sensor: KWaveSensor(), options: options)
        XCTAssertNotNil(out.pFinal)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))

        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        XCTAssertEqual(tracks.count, 1)
        let duration = try await asset.load(.duration)
        XCTAssertEqual(duration.seconds, 10.0 / 30.0, accuracy: 0.05, "10 frames at 30 fps")
    }
}

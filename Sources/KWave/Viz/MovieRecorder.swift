import Foundation
import AVFoundation
import CoreGraphics
import CoreVideo
import MLX

/// H.264 movie writer for simulation frames (the back end of `SimulationOptions.recordMovie`).
/// Append `CGImage` frames at a fixed rate, then `finish()` to write the file.
public final class MovieRecorder {
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let fps: Int32
    private var frameCount: Int64 = 0
    public let width: Int
    public let height: Int

    /// - Parameters:
    ///   - url: output `.mov`/`.mp4` path (an existing file is replaced).
    ///   - width/height: frame dimensions in pixels.
    ///   - fps: playback frame rate (default 30).
    public init(url: URL, width: Int, height: Int, fps: Int = 30) throws {
        try? FileManager.default.removeItem(at: url)
        self.width = width
        self.height = height
        self.fps = Int32(fps)
        let fileType: AVFileType = url.pathExtension.lowercased() == "mp4" ? .mp4 : .mov
        writer = try AVAssetWriter(outputURL: url, fileType: fileType)
        input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ])
        input.expectsMediaDataInRealTime = false
        adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
    }

    /// Append one frame (must match the recorder's dimensions).
    public func append(_ image: CGImage) throws {
        precondition(image.width == width && image.height == height,
                     "frame size must match the recorder dimensions")
        guard let pool = adaptor.pixelBufferPool else {
            throw NSError(domain: "MovieRecorder", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "pixel buffer pool unavailable"])
        }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        guard let buffer else {
            throw NSError(domain: "MovieRecorder", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "could not create pixel buffer"])
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)!
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, [])

        // Block until the writer is ready for the next frame (offline encoding).
        while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.005) }
        adaptor.append(buffer, withPresentationTime: CMTime(value: frameCount, timescale: fps))
        frameCount += 1
    }

    /// Finalize the file, blocking until writing completes.
    public func finish() throws {
        input.markAsFinished()
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()
        if writer.status == .failed {
            throw writer.error ?? NSError(domain: "MovieRecorder", code: 3)
        }
    }
}

/// Run `body` with a field monitor installed that records every monitored frame to `path`
/// (composing with any user-provided monitor), then finalize the movie. Used by the solver
/// dispatchers to implement `SimulationOptions.recordMovie`.
func withMovieRecording(
    options: SimulationOptions, path: String,
    body: (SimulationOptions) -> SimulationOutput
) -> SimulationOutput {
    var recorder: MovieRecorder?
    let colorMap = getColorMap()
    var patched = options
    let userMonitor = options.fieldMonitor
    patched.fieldMonitor = { t, p in
        userMonitor?(t, p)
        let image = fieldToImage(p, colorMap: colorMap, plotScale: options.plotScale)
        if recorder == nil {
            recorder = try? MovieRecorder(url: URL(fileURLWithPath: path),
                                          width: image.width, height: image.height)
        }
        try? recorder?.append(image)
    }
    let output = body(patched)
    try? recorder?.finish()
    return output
}

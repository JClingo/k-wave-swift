#if canImport(SwiftUI)
import SwiftUI
import CoreGraphics
import MLX

/// Observable bridge between a solver's `fieldMonitor` callback and SwiftUI. Install
/// `monitor.callback` as `options.fieldMonitor` (typically with the simulation running off the
/// main thread) and show a `SimulationMonitorView` bound to it.
@MainActor
public final class FieldMonitorModel: ObservableObject {
    @Published public private(set) var image: CGImage?
    @Published public private(set) var timeIndex: Int = 0

    public let colorMap: [RGB]
    public let plotScale: PlotScale

    public init(colorMap: [RGB] = getColorMap(), plotScale: PlotScale = .auto) {
        self.colorMap = colorMap
        self.plotScale = plotScale
    }

    /// Thread-safe callback for `SimulationOptions.fieldMonitor`: renders the field and posts the
    /// frame to the main actor.
    public nonisolated var callback: (Int, MLXArray) -> Void {
        { [weak self] t, field in
            guard let self else { return }
            let frame = fieldToImage(field, colorMap: self.colorMap, plotScale: self.plotScale)
            Task { @MainActor in
                self.image = frame
                self.timeIndex = t
            }
        }
    }
}

/// Live pressure-field display for a running simulation (nearest-neighbour scaling — grid cells
/// stay crisp).
public struct SimulationMonitorView: View {
    @ObservedObject private var model: FieldMonitorModel

    public init(model: FieldMonitorModel) {
        self.model = model
    }

    public var body: some View {
        Group {
            if let image = model.image {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(.none)
                    .aspectRatio(contentMode: .fit)
            } else {
                Text("Waiting for simulation…")
                    .foregroundStyle(.secondary)
            }
        }
    }
}
#endif

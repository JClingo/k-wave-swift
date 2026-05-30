// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "KWave",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "KWave", targets: ["KWave"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3")
    ],
    targets: [
        .systemLibrary(
            name: "CHDF5",
            pkgConfig: "hdf5",
            providers: [.brew(["hdf5"])]
        ),
        .target(
            name: "KWave",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                "CHDF5"
            ]
        ),
        .testTarget(
            name: "KWaveTests",
            dependencies: ["KWave"]
        )
    ]
)

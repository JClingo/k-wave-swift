import XCTest
@testable import KWave

final class IOTests: XCTestCase {
    func testFloatDatasetRoundTrip() throws {
        let path = NSTemporaryDirectory() + "kwave_io_test_\(UUID().uuidString).h5"
        defer { try? FileManager.default.removeItem(atPath: path) }

        let shape = [3, 4]
        let data: [Float] = (0..<12).map { Float($0) * 0.5 - 1 }

        do {
            let f = try HDF5File(create: path)
            try f.writeFloatDataset("p", shape: shape, data: data)
        }

        let f = try HDF5File(open: path)
        let (readShape, readData) = try f.readFloatDataset("p")
        XCTAssertEqual(readShape, shape)
        XCTAssertEqual(readData, data)
    }

    func testKWaveSchemaRoundTrip() throws {
        let path = NSTemporaryDirectory() + "kwave_schema_\(UUID().uuidString).h5"
        defer { try? FileManager.default.removeItem(atPath: path) }

        let nx = 3, ny = 4
        let field: [Float] = (0..<nx * ny).map { Float($0) }
        do {
            let f = try HDF5File(create: path)
            try f.writeRootAttributes(fileType: .input)
            try f.writeFieldDataset2D("p0_source_input", nx: nx, ny: ny, data: field)
            try f.writeScalarUInt64("Nx", UInt64(nx))
            try f.writeScalarFloat("dt", 1.5e-8)
            try f.writeUInt64Vector("sensor_mask_index", [1, 5, 9])
        }

        let f = try HDF5File(open: path)
        XCTAssertEqual(try f.readRootStringAttribute("file_type"), "input")
        XCTAssertEqual(try f.readRootStringAttribute("major_version"), "1")
        XCTAssertEqual(try f.readRootStringAttribute("minor_version"), "2")

        // Field is stored transposed as [1, ny, nx]; reading raw confirms the layout.
        let (shape, raw) = try f.readFloatDataset("p0_source_input")
        XCTAssertEqual(shape, [1, ny, nx])
        // raw[i*nx + j] == field[j*ny + i]
        for j in 0..<nx { for i in 0..<ny {
            XCTAssertEqual(raw[i * nx + j], field[j * ny + i])
        } }
    }

    func testMissingDatasetThrows() throws {
        let path = NSTemporaryDirectory() + "kwave_io_missing_\(UUID().uuidString).h5"
        defer { try? FileManager.default.removeItem(atPath: path) }
        _ = try HDF5File(create: path)
        let f = try HDF5File(open: path)
        XCTAssertThrowsError(try f.readFloatDataset("nope"))
    }
}

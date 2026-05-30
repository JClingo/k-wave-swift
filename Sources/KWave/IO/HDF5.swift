import CHDF5
import Foundation

/// Errors raised by the HDF5 wrapper.
public enum HDF5Error: Error, CustomStringConvertible {
    case openFailed(String)
    case datasetNotFound(String)
    case readFailed(String)
    case writeFailed(String)

    public var description: String {
        switch self {
        case .openFailed(let p): return "HDF5: could not open file \(p)"
        case .datasetNotFound(let n): return "HDF5: dataset \(n) not found"
        case .readFailed(let n): return "HDF5: failed to read dataset \(n)"
        case .writeFailed(let n): return "HDF5: failed to write dataset \(n)"
        }
    }
}

/// A thin RAII wrapper over the HDF5 C API for reading/writing float datasets.
///
/// Datasets are stored row-major (C order), matching the `shape` arrays used here. This is the
/// foundation for k-Wave-compatible I/O; k-Wave's specific dataset names, attributes, and
/// `[Nz, Ny, Nx]` layout are layered on top by the input/output builders.
public final class HDF5File {
    private let fileID: hid_t

    /// Access mode for opening a file.
    public enum Mode { case readOnly, readWrite }

    /// Open an existing file.
    public init(open path: String, mode: Mode = .readOnly) throws {
        let flags = mode == .readOnly ? kw_acc_rdonly() : kw_acc_rdwr()
        let id = H5Fopen(path, flags, kw_p_default())
        guard id >= 0 else { throw HDF5Error.openFailed(path) }
        self.fileID = id
    }

    /// Create a new file, truncating any existing file at `path`.
    public init(create path: String) throws {
        let id = H5Fcreate(path, kw_acc_trunc(), kw_p_default(), kw_p_default())
        guard id >= 0 else { throw HDF5Error.openFailed(path) }
        self.fileID = id
    }

    deinit { H5Fclose(fileID) }

    /// Read a float dataset, returning its row-major data and dimensions.
    public func readFloatDataset(_ name: String) throws -> (shape: [Int], data: [Float]) {
        let dset = H5Dopen2(fileID, name, kw_p_default())
        guard dset >= 0 else { throw HDF5Error.datasetNotFound(name) }
        defer { H5Dclose(dset) }

        let space = H5Dget_space(dset)
        defer { H5Sclose(space) }
        let ndim = Int(H5Sget_simple_extent_ndims(space))
        var dims = [hsize_t](repeating: 0, count: max(ndim, 1))
        _ = H5Sget_simple_extent_dims(space, &dims, nil)
        let shape = (0..<ndim).map { Int(dims[$0]) }
        let count = shape.reduce(1, *)

        var data = [Float](repeating: 0, count: count)
        let status = data.withUnsafeMutableBytes { buf in
            H5Dread(dset, kw_type_native_float(), kw_s_all(), kw_s_all(), kw_p_default(),
                    buf.baseAddress)
        }
        guard status >= 0 else { throw HDF5Error.readFailed(name) }
        return (shape, data)
    }

    /// Write a float dataset of the given row-major `shape`.
    public func writeFloatDataset(_ name: String, shape: [Int], data: [Float]) throws {
        precondition(shape.reduce(1, *) == data.count, "shape/data size mismatch for \(name)")
        var dims = shape.map { hsize_t($0) }
        let space = H5Screate_simple(Int32(shape.count), &dims, nil)
        defer { H5Sclose(space) }

        let dset = H5Dcreate2(fileID, name, kw_type_native_float(), space,
                              kw_p_default(), kw_p_default(), kw_p_default())
        guard dset >= 0 else { throw HDF5Error.writeFailed(name) }
        defer { H5Dclose(dset) }

        let status = data.withUnsafeBytes { buf in
            H5Dwrite(dset, kw_type_native_float(), kw_s_all(), kw_s_all(), kw_p_default(),
                     buf.baseAddress)
        }
        guard status >= 0 else { throw HDF5Error.writeFailed(name) }
    }

    // MARK: - k-Wave schema

    /// k-Wave HDF5 file role (root `file_type` attribute).
    public enum KWaveFileType: String { case input, output, checkpoint }

    /// Write the root attributes required by the k-Wave C++/CUDA binaries.
    public func writeRootAttributes(
        fileType: KWaveFileType, createdBy: String = "k-wave-swift",
        fileDescription: String = "k-Wave input file"
    ) throws {
        try writeStringAttribute(fileID, "created_by", createdBy)
        try writeStringAttribute(fileID, "creation_date", Self.dateString())
        try writeStringAttribute(fileID, "file_description", fileDescription)
        try writeStringAttribute(fileID, "file_type", fileType.rawValue)
        try writeStringAttribute(fileID, "major_version", "1")
        try writeStringAttribute(fileID, "minor_version", "2")
    }

    /// Write a 2D field as a k-Wave `[Nz=1, Ny, Nx]` float dataset (data transposed to match the
    /// C++ Fortran-order convention), tagging it with `domain_type`/`data_type` attributes.
    public func writeFieldDataset2D(_ name: String, nx: Int, ny: Int, data: [Float]) throws {
        precondition(data.count == nx * ny, "data size mismatch for \(name)")
        // On disk k-Wave stores the transpose: buf[i*nx + j] = data[j*ny + i].
        var buf = [Float](repeating: 0, count: nx * ny)
        for j in 0..<nx { for i in 0..<ny { buf[i * nx + j] = data[j * ny + i] } }
        try writeTagged(name, dims: [1, hsize_t(ny), hsize_t(nx)],
                        type: kw_type_native_float(), bytes: buf, dataTypeC: "float")
    }

    /// Write a scalar as a k-Wave `[1, 1, 1]` float dataset.
    public func writeScalarFloat(_ name: String, _ value: Float) throws {
        try writeTagged(name, dims: [1, 1, 1], type: kw_type_native_float(),
                        bytes: [value], dataTypeC: "float")
    }

    /// Write a scalar as a k-Wave `[1, 1, 1]` uint64 (`long`) dataset.
    public func writeScalarUInt64(_ name: String, _ value: UInt64) throws {
        try writeTagged(name, dims: [1, 1, 1], type: kw_type_native_uint64(),
                        bytes: [value], dataTypeC: "long")
    }

    /// Write a 1D integer index vector as a k-Wave `[1, 1, N]` uint64 (`long`) dataset.
    public func writeUInt64Vector(_ name: String, _ values: [UInt64]) throws {
        try writeTagged(name, dims: [1, 1, hsize_t(values.count)], type: kw_type_native_uint64(),
                        bytes: values, dataTypeC: "long")
    }

    /// Read a string attribute from the root group (used for round-trip verification).
    public func readRootStringAttribute(_ name: String) throws -> String {
        let attr = H5Aopen(fileID, name, kw_p_default())
        guard attr >= 0 else { throw HDF5Error.datasetNotFound("attr \(name)") }
        defer { H5Aclose(attr) }
        let atype = H5Aget_type(attr)
        defer { H5Tclose(atype) }
        let size = H5Tget_size(atype)
        var buf = [CChar](repeating: 0, count: size + 1)
        let status = H5Aread(attr, atype, &buf)
        guard status >= 0 else { throw HDF5Error.readFailed("attr \(name)") }
        return String(cString: buf)
    }

    // MARK: - private helpers

    private func writeTagged<T>(
        _ name: String, dims: [hsize_t], type: hid_t, bytes: [T], dataTypeC: String
    ) throws {
        var dimsVar = dims
        let space = H5Screate_simple(Int32(dims.count), &dimsVar, nil)
        defer { H5Sclose(space) }
        let dset = H5Dcreate2(fileID, name, type, space,
                              kw_p_default(), kw_p_default(), kw_p_default())
        guard dset >= 0 else { throw HDF5Error.writeFailed(name) }
        defer { H5Dclose(dset) }
        let status = bytes.withUnsafeBytes { buf in
            H5Dwrite(dset, type, kw_s_all(), kw_s_all(), kw_p_default(), buf.baseAddress)
        }
        guard status >= 0 else { throw HDF5Error.writeFailed(name) }
        try writeStringAttribute(dset, "domain_type", "real")
        try writeStringAttribute(dset, "data_type", dataTypeC)
    }

    private func writeStringAttribute(_ objID: hid_t, _ name: String, _ value: String) throws {
        var bytes = Array(value.utf8); bytes.append(0)
        let strType = kw_make_string_type(bytes.count)
        defer { H5Tclose(strType) }
        let space = kw_screate_scalar()
        defer { H5Sclose(space) }
        let attr = H5Acreate2(objID, name, strType, space, kw_p_default(), kw_p_default())
        guard attr >= 0 else { throw HDF5Error.writeFailed("attr \(name)") }
        defer { H5Aclose(attr) }
        let status = bytes.withUnsafeBytes { H5Awrite(attr, strType, $0.baseAddress) }
        guard status >= 0 else { throw HDF5Error.writeFailed("attr \(name)") }
    }

    private static func dateString() -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "dd-MMM-yyyy-HH-mm-ss"
        fmt.locale = Locale(identifier: "en_US_POSIX")
        return fmt.string(from: Date())
    }
}

#include <hdf5.h>

// HDF5 exposes its native datatypes and access flags as macros (some expanding to mutable
// globals via H5OPEN). Those do not import cleanly into Swift 6 strict-concurrency. Wrap them
// in static inline accessors so Swift sees plain function calls returning hid_t/unsigned.

static inline hid_t  kw_type_native_float(void)  { return H5T_NATIVE_FLOAT; }
static inline hid_t  kw_type_native_double(void) { return H5T_NATIVE_DOUBLE; }
static inline hid_t  kw_type_native_uint64(void) { return H5T_NATIVE_UINT64; }
static inline hid_t  kw_type_native_int(void)    { return H5T_NATIVE_INT; }

static inline unsigned kw_acc_rdonly(void) { return H5F_ACC_RDONLY; }
static inline unsigned kw_acc_rdwr(void)   { return H5F_ACC_RDWR; }
static inline unsigned kw_acc_trunc(void)  { return H5F_ACC_TRUNC; }

static inline hid_t kw_p_default(void) { return H5P_DEFAULT; }
static inline hid_t kw_s_all(void)     { return H5S_ALL; }

// String-attribute support.
static inline hid_t kw_type_c_s1(void) { return H5T_C_S1; }

// Create a fixed-length, null-terminated string datatype of `len` bytes.
static inline hid_t kw_make_string_type(size_t len) {
    hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, len);
    H5Tset_strpad(t, H5T_STR_NULLTERM);
    return t;
}

// Create a scalar dataspace (for scalar attributes).
static inline hid_t kw_screate_scalar(void) { return H5Screate(H5S_SCALAR); }

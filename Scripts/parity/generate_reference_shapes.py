#!/usr/bin/env python3
"""Reference for binary/Cartesian geometry generators: make_arc (finite radius) and
make_cart_sphere (Golden Section Spiral). The Swift ports must match exactly.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_shapes.py`
"""
import os
import numpy as np
import h5py
from kwave.data import Vector
from kwave.utils.mapgen import make_arc, make_cart_sphere, make_cart_arc

out_dir = os.path.dirname(os.path.abspath(__file__))

# --- make_arc (binary 2D, finite radius); positions are 1-based ------------
arc = make_arc(Vector([64, 64]), np.array([32, 16]), 20, 21, Vector([32, 32])).astype(np.float32)
with h5py.File(os.path.join(out_dir, "reference_arc.h5"), "w") as f:
    f.create_dataset("arc", data=arc)
    for k, v in [("Nx", 64), ("Ny", 64), ("ax", 32), ("ay", 16),
                 ("radius", 20), ("diameter", 21), ("fx", 32), ("fy", 32)]:
        f.create_dataset(k, data=np.float32(v))
print("wrote reference_arc.h5  arc sum", float(arc.sum()))

# --- make_cart_sphere ([3, N] point set) -----------------------------------
sphere = make_cart_sphere(5e-3, 40, Vector([1e-3, -2e-3, 3e-3])).astype(np.float32)
with h5py.File(os.path.join(out_dir, "reference_cartsphere.h5"), "w") as f:
    f.create_dataset("sphere", data=sphere)
print("wrote reference_cartsphere.h5  shape", sphere.shape)

# --- make_cart_arc ([2, N] point set) --------------------------------------
arc2 = make_cart_arc(np.array([0.0, 0.0]), 8e-3, 6e-3, np.array([0.0, 10e-3]), 25).astype(np.float32)
with h5py.File(os.path.join(out_dir, "reference_cartarc.h5"), "w") as f:
    f.create_dataset("arc", data=arc2)
print("wrote reference_cartarc.h5  shape", arc2.shape)

#!/usr/bin/env python3
"""Reference for Cartesian point-set geometry (k-wave-python NumPy): make_cart_circle (full + arc)
and make_cart_rect (2D, unrotated + rotated). The Swift ports must match these coordinates exactly.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_cartshapes.py`
"""
import os
import numpy as np
import h5py
from kwave.data import Vector
from kwave.utils.mapgen import make_cart_circle, make_cart_rect

out = {}
out["circle_full"] = make_cart_circle(5e-3, 30, Vector([1e-3, -2e-3])).astype(np.float32)
out["circle_arc"] = make_cart_circle(4e-3, 20, Vector([0.0, 0.0]), arc_angle=np.pi).astype(np.float32)
out["rect_plain"] = make_cart_rect(Vector([0.0, 0.0]), 4e-3, 2e-3, num_points=50).astype(np.float32)
out["rect_rot"] = make_cart_rect(Vector([1e-3, 2e-3]), 4e-3, 2e-3, theta=30, num_points=50).astype(np.float32)

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, "reference_cartshapes.h5")
with h5py.File(ref_path, "w") as f:
    for k, v in out.items():
        f.create_dataset(k, data=np.asarray(v, dtype=np.float32))

print("wrote", ref_path)
for k, v in out.items():
    print(f"  {k:12} shape {np.asarray(v).shape}")

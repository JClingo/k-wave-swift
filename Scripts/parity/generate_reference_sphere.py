#!/usr/bin/env python3
"""Generate makeSphere references from k-wave-python for the Swift `makeSphere` parity test.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_sphere.py`
"""
import os
import numpy as np
import h5py

from kwave.data import Vector
from kwave.utils.mapgen import make_sphere

CASES = [(32, 7), (40, 12), (24, 5)]

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, "reference_sphere.h5")
with h5py.File(ref_path, "w") as f:
    f.create_dataset("ncases", data=np.float32(len(CASES)))
    for i, (n, r) in enumerate(CASES):
        s = make_sphere(Vector([n, n, n]), r, binary=True).astype(np.float32)
        f.create_dataset(f"sphere_{i}", data=s)
        f.create_dataset(f"r_{i}", data=np.float32(r))
        print(n, r, "sum", int(s.sum()), "shape", s.shape)

print("wrote", ref_path)

#!/usr/bin/env python3
"""Reference for discrete trig transforms (DCT/DST types I-IV, unnormalized FFTW convention) —
the foundation of the axisymmetric solver's WSWA/WSWS symmetry machinery.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_dtt.py`
"""
import os
import numpy as np
import h5py
from scipy.fft import dct, dst

rng = np.random.default_rng(7)
x1 = rng.standard_normal(8)          # even length
x2 = rng.standard_normal(7)          # odd length
x3 = rng.standard_normal((5, 6))     # 2D, transform along each axis

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_dtt.h5")
with h5py.File(ref, "w") as f:
    f.create_dataset("x1", data=x1.astype(np.float32))
    f.create_dataset("x2", data=x2.astype(np.float32))
    f.create_dataset("x3", data=x3.astype(np.float32))
    for t in (1, 2, 3, 4):
        for name, x in (("x1", x1), ("x2", x2)):
            f.create_dataset(f"dct{t}_{name}", data=dct(x, type=t, norm=None).astype(np.float32))
            f.create_dataset(f"dst{t}_{name}", data=dst(x, type=t, norm=None).astype(np.float32))
    f.create_dataset("dct2_x3_ax0", data=dct(x3, type=2, axis=0, norm=None).astype(np.float32))
    f.create_dataset("dst3_x3_ax1", data=dst(x3, type=3, axis=1, norm=None).astype(np.float32))

print("wrote", ref)

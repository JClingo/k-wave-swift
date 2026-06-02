#!/usr/bin/env python3
"""Reference for `power_law_kramers_kronig` (sound-speed dispersion from power-law absorption).

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_kk.py`
"""
import os
import numpy as np
import h5py
from kwave.utils.mapgen import power_law_kramers_kronig

w = 2 * np.pi * np.linspace(0.5e6, 5e6, 60)
w0 = 2 * np.pi * 1e6
c0 = 1500.0
# Per-exponent a0 [Np] chosen for mild, non-singular dispersion about c0.
cases = {"y0p5": (0.5, 1.0), "y1": (1.0, 1e-6), "y1p5": (1.5, 3e-8)}

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_kk.h5")
with h5py.File(ref, "w") as h:
    h.create_dataset("w", data=w.astype(np.float32))
    h.create_dataset("w0", data=np.float32(w0))
    h.create_dataset("c0", data=np.float32(c0))
    for key, (y, a0) in cases.items():
        h.create_dataset(key, data=power_law_kramers_kronig(w, w0, c0, a0, y).astype(np.float32))
        h.create_dataset(key + "_y", data=np.float32(y))
        h.create_dataset(key + "_a0", data=np.float32(a0))
    cases = {k: power_law_kramers_kronig(w, w0, c0, a0, y) for k, (y, a0) in cases.items()}

print("wrote", ref, {k: (round(float(v.min()), 3), round(float(v.max()), 3)) for k, v in cases.items()})

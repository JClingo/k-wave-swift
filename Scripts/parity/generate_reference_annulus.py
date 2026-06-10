#!/usr/bin/env python3
"""Reference for `focused_annulus_oneil` (O'Neil analytic focused-annulus axial pressure).

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_annulus.py`
"""
import os
import warnings
import numpy as np
import h5py

warnings.filterwarnings("ignore")
from kwave.utils.mapgen import focused_annulus_oneil

radius, freq, c0, rho0 = 140e-3, 1e6, 1500.0, 1000.0
diameters = np.array([[0.0, 40e-3, 80e-3], [40e-3, 80e-3, 120e-3]])   # [2, 3] inner/outer
amplitude = np.array([0.1, 0.08, 0.12])
phase = np.array([0.0, 0.5, -0.7])
ax = np.arange(0, 250e-3 + 1e-3, 2e-3)
pa = focused_annulus_oneil(radius, diameters, amplitude, phase, freq, c0, rho0, ax)

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_annulus.h5")
with h5py.File(ref, "w") as f:
    for k, v in [("radius", radius), ("freq", freq), ("c0", c0), ("rho0", rho0)]:
        f.create_dataset(k, data=np.float32(v))
    f.create_dataset("diameters", data=diameters.astype(np.float32))
    f.create_dataset("amplitude", data=amplitude.astype(np.float32))
    f.create_dataset("phase", data=phase.astype(np.float32))
    f.create_dataset("ax", data=ax.astype(np.float32))
    f.create_dataset("pa", data=pa.astype(np.float32))

print("wrote", ref, "pa shape", pa.shape, "max", round(float(pa.max()), 2))

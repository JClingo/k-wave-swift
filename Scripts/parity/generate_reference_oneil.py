#!/usr/bin/env python3
"""Reference for `focused_bowl_oneil` (O'Neil analytic focused-bowl pressure).

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_oneil.py`
"""
import os
import warnings
import numpy as np
import h5py

warnings.filterwarnings("ignore")
from kwave.utils.mapgen import focused_bowl_oneil

radius, diameter, velocity, freq, c0, rho0 = 140e-3, 120e-3, 100e-3, 1e6, 1500.0, 1000.0
ax = np.arange(0, 250e-3 + 1e-3, 2e-3)
lat = np.arange(-15e-3, 15e-3 + 1e-3, 1e-3)
pa, pl, pac = focused_bowl_oneil(radius, diameter, velocity, freq, c0, rho0, ax, lat)

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_oneil.h5")
with h5py.File(ref, "w") as f:
    for k, v in [("radius", radius), ("diameter", diameter), ("velocity", velocity),
                 ("freq", freq), ("c0", c0), ("rho0", rho0)]:
        f.create_dataset(k, data=np.float32(v))
    f.create_dataset("ax", data=ax.astype(np.float32))
    f.create_dataset("lat", data=lat.astype(np.float32))
    f.create_dataset("pa", data=pa.astype(np.float32))
    f.create_dataset("pl", data=pl.astype(np.float32))
    f.create_dataset("pac_re", data=np.real(pac).astype(np.float32))
    f.create_dataset("pac_im", data=np.imag(pac).astype(np.float32))

print("wrote", ref, "pa max", round(float(pa.max()), 2), "pl max", round(float(pl.max()), 2))

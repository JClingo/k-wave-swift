#!/usr/bin/env python3
"""Reference for broadband `angular_spectrum` (time-domain plane projection).

Uses odd Nt only: the even-Nt double-sided spectrum rebuild in k-wave-python drops one bin
(`[1:-2]`, off-by-one vs MATLAB `2:end-1`) and crashes on record_time_series. z_pos must be a
scalar per call (beartype), so multiple z values are separate calls.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_angspec.py`
"""
import os
import warnings
import numpy as np
import h5py

warnings.filterwarnings("ignore")
from kwave.utils.angular_spectrum import angular_spectrum

Nx = Ny = 16
Nt = 21                      # odd — see module docstring
dx = 1e-4
dt = 2e-8                    # CFL = 1500·2e-8/1e-4 = 0.3
c0 = 1500

# Smooth input: Gaussian spatial envelope × tone burst in time.
x = (np.arange(Nx) - Nx // 2)[:, None, None]
y = (np.arange(Ny) - Ny // 2)[None, :, None]
t = np.arange(Nt)[None, None, :]
f0 = 5e6
envelope = np.exp(-(x**2 + y**2) / 18.0)
burst = np.sin(2 * np.pi * f0 * t * dt) * np.exp(-((t - Nt / 2) ** 2) / 40.0)
input_plane = (envelope * burst).astype(np.float64)

z_vals = [0.0, 1e-3, 2.5e-3]
pmax = np.zeros((Nx, Ny, len(z_vals)))
ptime_z1 = None
for zi, z in enumerate(z_vals):
    out = angular_spectrum(input_plane, dx, dt, z, c0, record_time_series=True)
    pmax[:, :, zi] = np.asarray(out[0])[:, :, 0]
    if zi == 1:
        ptime_z1 = np.asarray(out[1])[:, :, :, 0]

# Grid-expansion case (single z).
out_ge = angular_spectrum(input_plane, dx, dt, 1e-3, c0, grid_expansion=4)
pmax_ge = np.asarray(out_ge if not isinstance(out_ge, tuple) else out_ge[0])[:, :, 0]

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_angspec.h5")
with h5py.File(ref, "w") as f:
    for k, v in [("dx", dx), ("dt", dt), ("c0", c0), ("f0", f0)]:
        f.create_dataset(k, data=np.float32(v))
    f.create_dataset("z_vals", data=np.array(z_vals, np.float32))
    f.create_dataset("input", data=input_plane.astype(np.float32))
    f.create_dataset("pmax", data=pmax.astype(np.float32))
    f.create_dataset("ptime_z1", data=ptime_z1.astype(np.float32))
    f.create_dataset("pmax_ge", data=pmax_ge.astype(np.float32))

print("wrote", ref, "pmax", pmax.shape, "max", round(float(pmax.max()), 5),
      "ge max", round(float(pmax_ge.max()), 5))

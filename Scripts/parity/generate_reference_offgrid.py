#!/usr/bin/env python3
"""Reference for `offGridPoints` (exact band-limited off-grid source spreading).

3D uses `off_grid_points(bli_type='exact')` directly. The 2D exact path in off_grid_points is buggy
upstream (divides by kgrid.dz==0), so the 2D reference is built from the same `get_delta_bli`
primitive via separable outer products — exactly what the Swift port computes.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_offgrid.py`
"""
import os
import numpy as np
import h5py
from kwave.kgrid import kWaveGrid
from kwave.data import Vector

from kwave.utils.interp import get_delta_bli

# --- 2D via get_delta_bli outer products -----------------------------------
Nx, Ny, dx = 40, 32, 1e-4
g2 = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dx]))
xv, yv = g2.x_vec.squeeze(), g2.y_vec.squeeze()
pts2 = np.array([[3.7e-4, -5.2e-4, 1.1e-4], [-2.3e-4, 4.8e-4, 0.0]])
scale2 = np.array([1.0, 0.5, 2.0])
m2 = np.zeros((Nx, Ny))
for p in range(pts2.shape[1]):
    bx = get_delta_bli(Nx, dx, xv, pts2[0, p])
    by = get_delta_bli(Ny, dx, yv, pts2[1, p])
    m2 += scale2[p] * np.outer(bx, by)

# --- 3D via get_delta_bli outer products (off_grid_points is broken upstream) ---
N3 = (24, 20, 16)
g3 = kWaveGrid(Vector(list(N3)), Vector([1e-4, 1e-4, 1e-4]))
xv3, yv3, zv3 = g3.x_vec.squeeze(), g3.y_vec.squeeze(), g3.z_vec.squeeze()
pts3 = np.array([[2.1e-4, -1.5e-4], [-3.3e-4, 2.2e-4], [1.7e-4, -0.9e-4]])
scale3 = np.array([1.0, 0.75])
m3 = np.zeros(N3)
for p in range(pts3.shape[1]):
    bx = get_delta_bli(N3[0], 1e-4, xv3, pts3[0, p])
    by = get_delta_bli(N3[1], 1e-4, yv3, pts3[1, p])
    bz = get_delta_bli(N3[2], 1e-4, zv3, pts3[2, p])
    m3 += scale3[p] * (bx[:, None, None] * by[None, :, None] * bz[None, None, :])

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_offgrid.h5")
with h5py.File(ref, "w") as f:
    f.create_dataset("pts2", data=pts2.astype(np.float32))
    f.create_dataset("scale2", data=scale2.astype(np.float32))
    f.create_dataset("m2", data=np.asarray(m2, np.float32))
    f.create_dataset("pts3", data=pts3.astype(np.float32))
    f.create_dataset("scale3", data=scale3.astype(np.float32))
    f.create_dataset("m3", data=np.asarray(m3, np.float32))

print("wrote", ref, "m2", m2.shape, round(float(np.max(np.abs(m2))), 4),
      "m3", m3.shape, round(float(np.max(np.abs(m3))), 4))

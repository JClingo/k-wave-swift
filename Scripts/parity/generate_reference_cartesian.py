#!/usr/bin/env python3
"""Generate references for the Cartesian-sensor / interpolation slice (k-wave-python NumPy backend):

  1. fourierShift  — Fourier-interpolant resampling (phase_shift_interpolate).
  2. interpCartData — nearest-neighbour resampling of Cartesian sensor data onto a binary mask.
  3. Cartesian sensor recording — a 2D sim whose sensor.mask is a [dim, N] point set; the recorded
     pressure is multilinearly interpolated off-grid (RegularGridInterpolator linear).

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_cartesian.py`
"""
import os
import numpy as np
import h5py

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.signals import tone_burst
from kwave.utils.math import phase_shift_interpolate
from kwave.utils.interp import interp_cart_data

rng = np.random.default_rng(1)
out = {}

# --- 1. fourierShift -------------------------------------------------------
fs_in = rng.standard_normal((8, 16)).astype(np.float32)
out["fs_in"] = fs_in
out["fs_out_ax1"] = phase_shift_interpolate(fs_in, 0.5).astype(np.float32)          # default axis = last
out["fs_out_ax0"] = phase_shift_interpolate(fs_in, 0.5, shift_dim=1).astype(np.float32)  # 1-based -> axis 0

# --- Grid shared by interpCartData + Cartesian recording -------------------
Nx, Ny = 64, 64
dx = dy = 1e-4
c0, rho0 = 1500.0, 1000.0
PML_SIZE = 10
kgrid = kWaveGrid([Nx, Ny], [dx, dy])

# Sanity: confirm the centered-axis coordinate convention the Swift port assumes.
assert np.allclose(kgrid.x_vec.squeeze(), (np.arange(Nx) - Nx // 2) * dx), "x coord convention mismatch"

# --- 2. interpCartData -----------------------------------------------------
# A handful of Cartesian sensor points (physical coords) and random data over them.
cart_pts = np.array([[-12, -4, 0, 6, 11], [-9, 3, 0, -7, 10]], dtype=np.float32) * dx  # (2, 5)
nt_ic = 7
cart_data = rng.standard_normal((cart_pts.shape[1], nt_ic)).astype(np.float32)
binary_mask = np.zeros((Nx, Ny), dtype=bool)
binary_mask[30, 28] = True
binary_mask[34, 36] = True
binary_mask[40, 20] = True
ic_out = interp_cart_data(kgrid, cart_data, cart_pts, binary_mask, interp="nearest").astype(np.float32)
out["ic_cart_pts"] = cart_pts
out["ic_cart_data"] = cart_data
out["ic_binary_mask"] = binary_mask.astype(np.float32)
out["ic_out"] = ic_out

# --- 3. Cartesian sensor recording -----------------------------------------
NSTEPS = 120
kgrid.makeTime(c0, cfl=0.3)
kgrid.setTime(NSTEPS, float(kgrid.dt))

p_mask = np.zeros((Nx, Ny), dtype=bool)
p_mask[Nx // 2, Ny // 2] = True
fs = 1.0 / float(kgrid.dt)
signal = tone_burst(fs, 1e6, 3).squeeze().astype(np.float32).reshape(1, -1)
source = kSource(); source.p_mask = p_mask; source.p = signal; source.p_mode = "additive"

# Off-grid Cartesian sensor points (deliberately not on grid nodes), inside the interior.
rec_pts = np.array([[-8.3, -2.7, 4.4, 9.1], [3.6, -6.2, 7.8, -10.5]], dtype=np.float32) * dx  # (2, 4)
sensor = kSensor(); sensor.mask = rec_pts; sensor.record = ["p"]
res = kspaceFirstOrder(kgrid, medium=kWaveMedium(sound_speed=c0, density=rho0),
                       source=source, sensor=sensor, pml_size=PML_SIZE, pml_inside=True,
                       smooth_p0=False, backend="python", device="cpu", dtype=np.float32, quiet=True)
out["rec_pts"] = rec_pts
out["rec_signal"] = signal.squeeze().astype(np.float32)
out["rec_p"] = np.asarray(res["p"], dtype=np.float32)

meta = dict(Nx=Nx, Ny=Ny, dx=dx, dy=dy, c0=c0, rho0=rho0, dt=float(kgrid.dt),
            Nt=int(kgrid.Nt), pml_size=PML_SIZE, src_x=Nx // 2, src_y=Ny // 2)

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, "reference_2d_cartesian.h5")
with h5py.File(ref_path, "w") as f:
    for k, v in meta.items():
        f.create_dataset(k, data=np.float32(v))
    for k, v in out.items():
        f.create_dataset(k, data=np.asarray(v, dtype=np.float32))

print("wrote", ref_path, "Nt", int(kgrid.Nt))
for k in ["fs_out_ax1", "ic_out", "rec_p"]:
    print(f"  {k:12} shape {np.asarray(out[k]).shape} max {float(np.max(np.abs(out[k]))):.4e}")

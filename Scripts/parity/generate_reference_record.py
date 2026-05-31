#!/usr/bin/env python3
"""Generate a 2D reference exercising the extended sensor recording fields, using k-wave-python's
pure-NumPy backend: collocated velocity time series (ux/uy), aggregates (p_max/p_min/p_rms,
u_max/u_rms), and time-averaged intensity (Ix_avg/Iy_avg).

A tone-burst pressure source at the grid centre radiates to a vertical line sensor; all fields are
recorded. The Swift solver ports the NumPy recording 1:1, so the tolerance is tight.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_record.py`
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

Nx, Ny = 64, 64
dx = dy = 1e-4
c0, rho0 = 1500.0, 1000.0
PML_SIZE = 10
NSTEPS = 120

kgrid = kWaveGrid([Nx, Ny], [dx, dy])
medium = kWaveMedium(sound_speed=c0, density=rho0)
kgrid.makeTime(medium.sound_speed, cfl=0.3)
kgrid.setTime(NSTEPS, float(kgrid.dt))

p_mask = np.zeros((Nx, Ny), dtype=bool)
p_mask[Nx // 2, Ny // 2] = True
fs = 1.0 / float(kgrid.dt)
signal = tone_burst(fs, 1e6, 3).squeeze().astype(np.float32).reshape(1, -1)

source = kSource()
source.p_mask = p_mask
source.p = signal
source.p_mode = "additive"

# Vertical line sensor offset from the source (inside the PML-free interior).
mask = np.zeros((Nx, Ny), dtype=bool)
mask[20, 16:48] = True

sensor = kSensor()
sensor.mask = mask
RECORD = ["p", "ux", "uy", "p_max", "p_min", "p_rms", "u_max", "u_rms", "I_avg"]
sensor.record = RECORD

out = kspaceFirstOrder(
    kgrid, medium, source, sensor,
    pml_size=PML_SIZE, pml_inside=True, smooth_p0=False,
    backend="python", device="cpu", dtype=np.float32, quiet=True,
)

# Expanded output keys we expect from the record list above.
keys = ["p", "ux", "uy", "p_max", "p_min", "p_rms",
        "ux_max", "uy_max", "ux_rms", "uy_rms", "Ix_avg", "Iy_avg"]

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, "reference_2d_record.h5")
with h5py.File(ref_path, "w") as f:
    for name, val in [("Nx", Nx), ("Ny", Ny), ("dx", dx), ("dy", dy),
                      ("c0", c0), ("rho0", rho0), ("dt", kgrid.dt),
                      ("Nt", kgrid.Nt), ("pml_size", PML_SIZE),
                      ("src_x", Nx // 2), ("src_y", Ny // 2)]:
        f.create_dataset(name, data=np.float32(val))
    f.create_dataset("signal", data=signal.squeeze().astype(np.float32))
    f.create_dataset("mask", data=mask.astype(np.float32))
    for k in keys:
        f.create_dataset(k, data=np.asarray(out[k], dtype=np.float32))

print("wrote", ref_path, "Nt", int(kgrid.Nt), "nSensor", int(mask.sum()))
for k in keys:
    print(f"  {k:8} shape {np.asarray(out[k]).shape} max {float(np.max(np.abs(out[k]))):.4e}")

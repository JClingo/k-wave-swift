#!/usr/bin/env python3
"""Generate a 3D IVP reference using k-wave-python's pure-Python backend (NumPy).

Mirrors generate_reference_py.py but in 3D, so the Swift 3D solver can be validated against the
same algorithm. The Swift parity test loads the SAME inputs (p0, dt, Nt, grid, medium) and
compares p_final against this reference.
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
from kwave.utils.mapgen import make_ball

Nx, Ny, Nz = 32, 32, 32
dx = dy = dz = 1e-4
c0, rho0 = 1500.0, 1000.0
PML_SIZE = 10
SMOOTH = os.environ.get("SMOOTH", "1") == "1"
NSTEPS = int(os.environ.get("NSTEPS", "0"))
SUFFIX = os.environ.get("SUFFIX", "")

kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])
medium = kWaveMedium(sound_speed=c0, density=rho0)
kgrid.makeTime(medium.sound_speed, cfl=0.3)
if NSTEPS > 0:
    kgrid.setTime(NSTEPS, float(kgrid.dt))

ball = make_ball(Vector([Nx, Ny, Nz]), Vector([Nx // 2, Ny // 2, Nz // 2]), 4).astype(np.float32)
source = kSource()
source.p0 = ball
sensor = kSensor()
sensor.mask = np.ones((Nx, Ny, Nz), dtype=bool)
sensor.record = ["p_final"]

out = kspaceFirstOrder(
    kgrid, medium, source, sensor,
    pml_size=PML_SIZE, pml_inside=True, smooth_p0=SMOOTH,
    backend="python", device="cpu", dtype=np.float32, quiet=True,
)
p_final = np.asarray(out["p_final"], dtype=np.float32)

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, f"reference_3d_ivp{SUFFIX}.h5")
with h5py.File(ref_path, "w") as f:
    for name, val in [("Nx", Nx), ("Ny", Ny), ("Nz", Nz), ("dx", dx), ("dy", dy), ("dz", dz),
                      ("c0", c0), ("rho0", rho0), ("dt", kgrid.dt),
                      ("Nt", kgrid.Nt), ("pml_size", PML_SIZE)]:
        f.create_dataset(name, data=np.float32(val))
    f.create_dataset("p0", data=ball)
    f.create_dataset("p_final", data=p_final)

print("wrote", ref_path, "Nt", int(kgrid.Nt), "max|p_final|", float(np.max(np.abs(p_final))),
      "shape", p_final.shape)

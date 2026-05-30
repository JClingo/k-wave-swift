#!/usr/bin/env python3
"""Generate a 2D IVP reference using k-wave-python's pure-Python backend (NumPy).

Used to validate the Swift solver against the same algorithm it mirrors, isolating
algorithm correctness from C++-vs-Python engine differences. dtype selectable via DTYPE env.
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
from kwave.utils.mapgen import make_disc

Nx, Ny = 64, 64
dx = dy = 1e-4
c0, rho0 = 1500.0, 1000.0
PML_SIZE = 20
SMOOTH = os.environ.get("SMOOTH", "1") == "1"
NSTEPS = int(os.environ.get("NSTEPS", "0"))
SUFFIX = os.environ.get("SUFFIX", "")
DTYPE = np.float32 if os.environ.get("DTYPE", "32") == "32" else np.float64

kgrid = kWaveGrid([Nx, Ny], [dx, dy])
medium = kWaveMedium(sound_speed=c0, density=rho0)
kgrid.makeTime(medium.sound_speed, cfl=0.3)
if NSTEPS > 0:
    kgrid.setTime(NSTEPS, float(kgrid.dt))

disc = make_disc(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), 5).astype(np.float32)
source = kSource()
source.p0 = disc
sensor = kSensor()
sensor.mask = np.ones((Nx, Ny), dtype=bool)
sensor.record = ["p_final"]

out = kspaceFirstOrder(
    kgrid, medium, source, sensor,
    pml_size=PML_SIZE, pml_inside=True, smooth_p0=SMOOTH,
    backend="python", device="cpu", dtype=DTYPE, quiet=True,
)
p_final = np.asarray(out["p_final"], dtype=np.float32)

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, f"reference_2d_ivp{SUFFIX}.h5")
with h5py.File(ref_path, "w") as f:
    for name, val in [("Nx", Nx), ("Ny", Ny), ("dx", dx), ("dy", dy),
                      ("c0", c0), ("rho0", rho0), ("dt", kgrid.dt),
                      ("Nt", kgrid.Nt), ("pml_size", PML_SIZE)]:
        f.create_dataset(name, data=np.float32(val))
    f.create_dataset("p0", data=disc)
    f.create_dataset("p_final", data=p_final)

print("wrote", ref_path, "Nt", int(kgrid.Nt), "max|p_final|", float(np.max(np.abs(p_final))))

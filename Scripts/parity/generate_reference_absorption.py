#!/usr/bin/env python3
"""Generate a 2D power-law absorption reference using k-wave-python's pure-Python (NumPy) backend.

The Swift solver is a 1:1 port of the NumPy `kspace_solver.py` equation of state
(p = c0^2 * (rho + absorption - dispersion)), so we validate against the NumPy
backend at tight tolerance rather than the C++ engine.

ALPHA_POWER (default 1.5) and ALPHA_MODE (default "" = full power-law; also
"no_dispersion" or "stokes") are selectable via env. Output: reference_2d_absorption{SUFFIX}.h5
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
ALPHA_COEFF = float(os.environ.get("ALPHA_COEFF", "0.75"))
ALPHA_POWER = float(os.environ.get("ALPHA_POWER", "1.5"))
ALPHA_MODE = os.environ.get("ALPHA_MODE", "")  # "", "no_dispersion", "stokes"
SUFFIX = os.environ.get("SUFFIX", "")

kgrid = kWaveGrid([Nx, Ny], [dx, dy])
medium = kWaveMedium(sound_speed=c0, density=rho0,
                     alpha_coeff=ALPHA_COEFF, alpha_power=ALPHA_POWER)
if ALPHA_MODE:
    medium.alpha_mode = ALPHA_MODE
kgrid.makeTime(medium.sound_speed, cfl=0.3)

disc = make_disc(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), 5).astype(np.float32)
source = kSource()
source.p0 = disc
sensor = kSensor()
sensor.mask = np.ones((Nx, Ny), dtype=bool)
# Record the full-grid time series; the last column is the final field WITHOUT the
# PML interior-crop that `p_final` applies, matching the Swift solver's full-grid pFinal.
sensor.record = ["p"]

out = kspaceFirstOrder(
    kgrid, medium, source, sensor,
    pml_size=PML_SIZE, pml_inside=True, smooth_p0=True,
    backend="python", device="cpu", dtype=np.float32, quiet=True,
)
p_final = np.asarray(out["p"][:, -1], dtype=np.float32).reshape(Nx, Ny)

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, f"reference_2d_absorption{SUFFIX}.h5")
with h5py.File(ref_path, "w") as f:
    for name, val in [("Nx", Nx), ("Ny", Ny), ("dx", dx), ("dy", dy),
                      ("c0", c0), ("rho0", rho0), ("dt", kgrid.dt),
                      ("Nt", kgrid.Nt), ("pml_size", PML_SIZE),
                      ("alpha_coeff", ALPHA_COEFF), ("alpha_power", ALPHA_POWER)]:
        f.create_dataset(name, data=np.float32(val))
    f.create_dataset("alpha_mode", data=np.bytes_(ALPHA_MODE or "powerlaw"))
    f.create_dataset("p0", data=disc)
    f.create_dataset("p_final", data=p_final)

print("wrote", ref_path, "Nt", int(kgrid.Nt), "mode", ALPHA_MODE or "powerlaw",
      "max|p_final|", float(np.max(np.abs(p_final))))

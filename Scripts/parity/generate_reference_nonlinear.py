#!/usr/bin/env python3
"""Generate a 2D nonlinear (B/A) IVP reference using k-wave-python's pure-Python (NumPy) backend.

The Swift solver ports the NumPy `kspace_solver.py` equation of state 1:1
(p = c0^2 * (rho + ... + BonA*rho^2/(2*rho0)), with the mass-conservation source
term scaled by nl_factor = (2*sum(rho) + rho0)/rho0), so we validate against the
NumPy backend at tight tolerance rather than the C++ engine.

Nonlinearity scales with rho^2, so a high source amplitude (P0_AMP, default 3e7 Pa)
is needed to make the term significant; BONA defaults to 10. Output: reference_2d_nonlinear{SUFFIX}.h5
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
BONA = float(os.environ.get("BONA", "10"))
P0_AMP = float(os.environ.get("P0_AMP", "3e7"))
SUFFIX = os.environ.get("SUFFIX", "")

kgrid = kWaveGrid([Nx, Ny], [dx, dy])
medium = kWaveMedium(sound_speed=c0, density=rho0, BonA=BONA)
kgrid.makeTime(medium.sound_speed, cfl=0.3)

disc = (P0_AMP * make_disc(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), 5)).astype(np.float32)
source = kSource()
source.p0 = disc
sensor = kSensor()
sensor.mask = np.ones((Nx, Ny), dtype=bool)
# Full-grid time series; last column is the final field without the PML interior-crop
# that `p_final` applies, matching the Swift solver's full-grid pFinal.
sensor.record = ["p"]

out = kspaceFirstOrder(
    kgrid, medium, source, sensor,
    pml_size=PML_SIZE, pml_inside=True, smooth_p0=True,
    backend="python", device="cpu", dtype=np.float32, quiet=True,
)
p_final = np.asarray(out["p"][:, -1], dtype=np.float32).reshape(Nx, Ny)

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, f"reference_2d_nonlinear{SUFFIX}.h5")
with h5py.File(ref_path, "w") as f:
    for name, val in [("Nx", Nx), ("Ny", Ny), ("dx", dx), ("dy", dy),
                      ("c0", c0), ("rho0", rho0), ("dt", kgrid.dt),
                      ("Nt", kgrid.Nt), ("pml_size", PML_SIZE), ("BonA", BONA)]:
        f.create_dataset(name, data=np.float32(val))
    f.create_dataset("p0", data=disc)
    f.create_dataset("p_final", data=p_final)

print("wrote", ref_path, "Nt", int(kgrid.Nt), "BonA", BONA, "P0", P0_AMP,
      "max|p_final|", float(np.max(np.abs(p_final))))

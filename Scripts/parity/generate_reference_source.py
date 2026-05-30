#!/usr/bin/env python3
"""Generate a 2D time-varying pressure-source reference using k-wave-python's
pure-Python backend, to validate the Swift solver's source injection.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_source.py`
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
MODE = os.environ.get("MODE", "additive")          # additive | additive-no-correction | dirichlet
SRC = os.environ.get("SRC", "p")                    # p | u (ux velocity source)
NSTEPS = int(os.environ.get("NSTEPS", "60"))
SUFFIX = os.environ.get("SUFFIX", "")

kgrid = kWaveGrid([Nx, Ny], [dx, dy])
medium = kWaveMedium(sound_speed=c0, density=rho0)
kgrid.makeTime(medium.sound_speed, cfl=0.3)
kgrid.setTime(NSTEPS, float(kgrid.dt))

# Single-point pressure source at the grid centre.
p_mask = np.zeros((Nx, Ny), dtype=bool)
p_mask[Nx // 2, Ny // 2] = True

fs = 1.0 / float(kgrid.dt)
signal = tone_burst(fs, 1e6, 3).squeeze().astype(np.float32)  # 1-D
signal = signal.reshape(1, -1)

source = kSource()
if SRC == "u":
    source.u_mask = p_mask
    source.ux = signal
    source.u_mode = MODE
else:
    source.p_mask = p_mask
    source.p = signal
    source.p_mode = MODE

sensor = kSensor()
sensor.mask = np.ones((Nx, Ny), dtype=bool)
sensor.record = ["p_final"]

out = kspaceFirstOrder(
    kgrid, medium, source, sensor,
    pml_size=PML_SIZE, pml_inside=True, smooth_p0=False,
    backend="python", device="cpu", dtype=np.float32, quiet=True,
)
p_final = np.asarray(out["p_final"], dtype=np.float32)

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, f"reference_source{SUFFIX}.h5")
with h5py.File(ref_path, "w") as f:
    for name, val in [("Nx", Nx), ("Ny", Ny), ("dx", dx), ("dy", dy),
                      ("c0", c0), ("rho0", rho0), ("dt", kgrid.dt),
                      ("Nt", kgrid.Nt), ("pml_size", PML_SIZE),
                      ("src_x", Nx // 2), ("src_y", Ny // 2)]:
        f.create_dataset(name, data=np.float32(val))
    f.create_dataset("signal", data=signal.squeeze().astype(np.float32))
    f.create_dataset("p_final", data=p_final)

print("wrote", ref_path, "Nt", int(kgrid.Nt), "mode", MODE,
      "max|p_final|", float(np.max(np.abs(p_final))))

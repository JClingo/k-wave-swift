#!/usr/bin/env python3
"""Generate a 2D time-reversal reconstruction reference using k-wave-python's NumPy backend.

Forward pass: a disc initial pressure radiates to a circular sensor; the sensor pressure
time series is recorded. Reconstruction: that data is time-reversed and re-injected as a
Dirichlet pressure boundary condition on the sensor mask, the solver is run forward, and the
final field is scaled by the compensation factor (2.0) with the positivity condition applied.

The reconstruction is reproduced here with the *same low-level steps* as the Swift `timeReversal`
helper (flip -> dirichlet source -> run -> *2 -> clamp), so the test isolates that composition.
The Swift solver ports the NumPy solver 1:1, so the tolerance is tight.

Stored: grid/medium scalars, the sensor `mask`, the forward `recorded_p` ([nSensor, Nt]) used as
the TR input, and the reference reconstruction `p0_recon` ([Nx, Ny], full grid).
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
from kwave.utils.mapgen import make_disc, make_circle

Nx, Ny = 64, 64
dx = dy = 1e-4
c0, rho0 = 1500.0, 1000.0
PML_SIZE = 20
COMP = 2.0

kgrid = kWaveGrid([Nx, Ny], [dx, dy])
medium = kWaveMedium(sound_speed=c0, density=rho0)
kgrid.makeTime(medium.sound_speed, cfl=0.3)

disc = make_disc(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), 5).astype(np.float32)
mask = make_circle(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), 26).astype(bool)


def run_forward(source_p0):
    s = kSource(); s.p0 = source_p0
    se = kSensor(); se.mask = mask; se.record = ["p"]
    o = kspaceFirstOrder(kgrid, medium, s, se, pml_size=PML_SIZE, pml_inside=True,
                         smooth_p0=True, backend="python", device="cpu", dtype=np.float32, quiet=True)
    return np.asarray(o["p"], dtype=np.float32)  # [nSensor, Nt]


def run_reconstruction(recorded):
    s = kSource()
    s.p_mask = mask
    s.p = np.flip(recorded, axis=1).astype(np.float32)
    s.p_mode = "dirichlet"
    se = kSensor(); se.mask = np.ones((Nx, Ny), dtype=bool); se.record = ["p"]
    o = kspaceFirstOrder(kgrid, medium, s, se, pml_size=PML_SIZE, pml_inside=True,
                         smooth_p0=False, backend="python", device="cpu", dtype=np.float32, quiet=True)
    p_final = np.asarray(o["p"][:, -1], dtype=np.float32).reshape(Nx, Ny)  # full grid
    p0 = COMP * p_final
    p0[p0 < 0] = 0.0
    return p0.astype(np.float32)


recorded = run_forward(disc)
p0_recon = run_reconstruction(recorded)

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, "reference_2d_tr.h5")
with h5py.File(ref_path, "w") as f:
    for name, val in [("Nx", Nx), ("Ny", Ny), ("dx", dx), ("dy", dy),
                      ("c0", c0), ("rho0", rho0), ("dt", kgrid.dt),
                      ("Nt", kgrid.Nt), ("pml_size", PML_SIZE), ("comp", COMP)]:
        f.create_dataset(name, data=np.float32(val))
    f.create_dataset("mask", data=mask.astype(np.float32))
    f.create_dataset("recorded_p", data=recorded)
    f.create_dataset("p0_recon", data=p0_recon)

print("wrote", ref_path, "Nt", int(kgrid.Nt), "nSensor", int(mask.sum()),
      "max|p0_recon|", float(np.max(np.abs(p0_recon))))

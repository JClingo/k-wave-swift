#!/usr/bin/env python3
"""Generate a 2D heterogeneous-medium IVP reference output from k-wave-python.

Runs k-Wave on a centred-disc initial pressure in a *spatially varying* medium (a circular
inclusion of higher sound speed and density embedded in a background) and writes inputs + outputs
to HDF5. The Swift parity test loads the SAME c0/rho0 maps, p0, dt and Nt and compares its
p_final — exercising staggered-density interpolation, the dt*rho0 density update, the c0^2*Sum(rho)
equation of state, and c_ref = max(c0).

Engine choice — IMPORTANT: this reference is produced by k-wave-python's pure-NumPy solver
(`kwave.solvers.kspace_solver`), NOT the C++/OMP binary engine. In a homogeneous medium the two
agree to float32 precision (~1e-5), so `reference_2d_ivp.h5` uses the C++ engine. In a spatially
varying medium they diverge ~2% at the medium discontinuity: the C++ engine applies an internal
heterogeneous detail that k-wave-python's own NumPy solver does not reproduce. The Swift solver is
a 1:1 port of the NumPy formulation (identical velocity/density/EOS/staggered-density/t=0 updates),
so it reproduces the NumPy field to float32 precision but inherits the same ~2% offset from C++.
We therefore validate the Swift heterogeneous path against the NumPy solver (tight ~1e-4 tolerance,
real regression detection) rather than the C++ engine (which would force a loose ~1e-1 tolerance
that masks bugs). See Scripts/parity/diag_numpy_vs_cpp.py for the measurement that established this.

We save the FULL grid (PML region included) because the Swift solver returns the full field;
`Simulation.run()` strips the PML from its returned "p_final", so we snapshot `sim.p` instead.
"""
import os
import numpy as np
import h5py

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.solvers.kspace_solver import Simulation
from kwave.utils.mapgen import make_disc

Nx, Ny = 64, 64
dx = dy = 1e-4
PML_SIZE = 20
PML_ALPHA = 2.0

# Heterogeneous medium: background + circular inclusion (kept clear of the PML).
c0_bg, c0_in = 1500.0, 1800.0
rho0_bg, rho0_in = 1000.0, 1200.0
inclusion = make_disc(Vector([Nx, Ny]), Vector([Nx // 2 + 6, Ny // 2 + 4]), 8).astype(bool)
c0 = np.full((Nx, Ny), c0_bg, dtype=np.float32)
rho0 = np.full((Nx, Ny), rho0_bg, dtype=np.float32)
c0[inclusion] = c0_in
rho0[inclusion] = rho0_in

kgrid = kWaveGrid([Nx, Ny], [dx, dy])
medium = kWaveMedium(sound_speed=c0, density=rho0)
kgrid.makeTime(medium.sound_speed, cfl=0.3)

disc = make_disc(Vector([Nx, Ny]), Vector([Nx // 2 - 8, Ny // 2 - 8]), 4).astype(np.float32)

source = kSource()
source.p0 = disc
sensor = kSensor()
sensor.mask = np.ones((Nx, Ny), dtype=bool)
sensor.record = ["p_final"]

sim = Simulation(
    kgrid, medium, source, sensor,
    device="cpu",
    smooth_p0=True,
    pml_size=(PML_SIZE, PML_SIZE),
    pml_alpha=(PML_ALPHA, PML_ALPHA),
    dtype=np.float32,
    quiet=True,
)
sim.run()
p_final = np.asarray(sim.p, dtype=np.float32)  # full grid (PML included), matches Swift output

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, "reference_2d_hetero.h5")
with h5py.File(ref_path, "w") as f:
    f.create_dataset("Nx", data=np.float32(Nx))
    f.create_dataset("Ny", data=np.float32(Ny))
    f.create_dataset("dx", data=np.float32(dx))
    f.create_dataset("dy", data=np.float32(dy))
    f.create_dataset("dt", data=np.float32(kgrid.dt))
    f.create_dataset("Nt", data=np.float32(kgrid.Nt))
    f.create_dataset("pml_size", data=np.float32(PML_SIZE))
    f.create_dataset("c0", data=c0)                  # [Nx, Ny]
    f.create_dataset("rho0", data=rho0)              # [Nx, Ny]
    f.create_dataset("p0", data=disc)                # [Nx, Ny]
    f.create_dataset("p_final", data=p_final)        # [Nx, Ny]

print("wrote", ref_path)
print("dt", float(kgrid.dt), "Nt", int(kgrid.Nt), "p_final shape", p_final.shape,
      "max|p_final|", float(np.max(np.abs(p_final))))

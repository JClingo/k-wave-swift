#!/usr/bin/env python3
"""Generate a 2D initial-value-problem reference output from k-wave-python.

Runs the canonical k-Wave C++ solver (via k-wave-python) on a centred-disc initial pressure in
a homogeneous, lossless medium, and writes inputs + outputs to an HDF5 file. The Swift parity
test loads the SAME inputs (p0, dt, Nt, grid, medium) from this file and compares its p_final
against the reference — so disc/center index conventions never have to match across ports.
"""
import os
import numpy as np
import h5py

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.mapgen import make_disc

Nx, Ny = 64, 64
dx = dy = 1e-4
c0, rho0 = 1500.0, 1000.0
PML_SIZE = 20
SMOOTH = os.environ.get("SMOOTH", "1") == "1"
NSTEPS = int(os.environ.get("NSTEPS", "0"))  # 0 = default auto t_end
SUFFIX = os.environ.get("SUFFIX", "")

kgrid = kWaveGrid([Nx, Ny], [dx, dy])
medium = kWaveMedium(sound_speed=c0, density=rho0)
kgrid.makeTime(medium.sound_speed, cfl=0.3)
if NSTEPS > 0:
    kgrid.setTime(NSTEPS, float(kgrid.dt))

disc = make_disc(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), 5).astype(np.float32)

source = kSource()
source.p0 = disc
sensor = kSensor()
sensor.mask = np.ones((Nx, Ny), dtype=bool)  # required even when only p_final is recorded
sensor.record = ["p_final"]

sim_opts = SimulationOptions(
    pml_inside=True,
    pml_size=PML_SIZE,
    smooth_p0=SMOOTH,
    save_to_disk=True,
)
exec_opts = SimulationExecutionOptions(is_gpu_simulation=False)

out = kspaceFirstOrder2D(
    kgrid=kgrid, medium=medium, source=source, sensor=sensor,
    simulation_options=sim_opts, execution_options=exec_opts,
)
p_final = np.asarray(out["p_final"], dtype=np.float32)

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, f"reference_2d_ivp{SUFFIX}.h5")
with h5py.File(ref_path, "w") as f:
    f.create_dataset("Nx", data=np.float32(Nx))
    f.create_dataset("Ny", data=np.float32(Ny))
    f.create_dataset("dx", data=np.float32(dx))
    f.create_dataset("dy", data=np.float32(dy))
    f.create_dataset("c0", data=np.float32(c0))
    f.create_dataset("rho0", data=np.float32(rho0))
    f.create_dataset("dt", data=np.float32(kgrid.dt))
    f.create_dataset("Nt", data=np.float32(kgrid.Nt))
    f.create_dataset("pml_size", data=np.float32(PML_SIZE))
    f.create_dataset("p0", data=disc)                # [Nx, Ny]
    f.create_dataset("p_final", data=p_final)        # [Nx, Ny]

print("wrote", ref_path)
print("dt", float(kgrid.dt), "Nt", int(kgrid.Nt), "p_final shape", p_final.shape,
      "max|p_final|", float(np.max(np.abs(p_final))))

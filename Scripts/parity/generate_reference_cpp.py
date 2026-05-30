#!/usr/bin/env python3
"""Generate a golden k-Wave C++ (OMP) input file + output for a tiny 2D IVP, used to
validate the Swift HDF5 input-file assembler and to drive the OMP binary from Swift.

Writes:
  reference_cpp_input.h5   - the k-Wave++ input file (golden schema)
  reference_cpp_output.h5  - the OMP binary's output (golden p_final)
  reference_cpp_meta.h5    - grid/medium params + p0 + golden p_final (float32, easy Swift read)

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_cpp.py`
"""
import os
import numpy as np
import h5py

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.mapgen import make_disc

Nx, Ny = 64, 64
dx = dy = 1e-4
c0, rho0 = 1500.0, 1000.0
PML_SIZE = 10
NSTEPS = 80

out_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(out_dir, "reference_cpp_input.h5")
output_path = os.path.join(out_dir, "reference_cpp_output.h5")
meta_path = os.path.join(out_dir, "reference_cpp_meta.h5")

kgrid = kWaveGrid([Nx, Ny], [dx, dy])
medium = kWaveMedium(sound_speed=c0, density=rho0)
kgrid.makeTime(medium.sound_speed, cfl=0.3)
kgrid.setTime(NSTEPS, float(kgrid.dt))

disc = make_disc(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), 5).astype(np.float32)
source = kSource()
source.p0 = disc

sensor = kSensor()
sensor.mask = np.ones((Nx, Ny), dtype=bool)
sensor.record = ["p_final"]

sim_opts = SimulationOptions(
    pml_inside=True,
    pml_size=PML_SIZE,
    smooth_p0=False,
    save_to_disk=True,
    input_filename=input_path,
    output_filename=output_path,
    data_cast="single",
)
exec_opts = SimulationExecutionOptions(is_gpu_simulation=False, delete_data=False, show_sim_log=False)

out = kspaceFirstOrder2DC(
    kgrid=kgrid, medium=medium, source=source, sensor=sensor,
    simulation_options=sim_opts, execution_options=exec_opts,
)
p_final = np.asarray(out["p_final"], dtype=np.float32).squeeze()

with h5py.File(meta_path, "w") as f:
    for name, val in [("Nx", Nx), ("Ny", Ny), ("dx", dx), ("dy", dy),
                      ("c0", c0), ("rho0", rho0), ("dt", kgrid.dt),
                      ("Nt", kgrid.Nt), ("pml_size", PML_SIZE)]:
        f.create_dataset(name, data=np.float32(val))
    f.create_dataset("p0", data=disc)
    f.create_dataset("p_final", data=p_final)

print("wrote", input_path)
print("wrote", output_path)
print("wrote", meta_path, "p_final shape", p_final.shape, "max", float(np.max(np.abs(p_final))))

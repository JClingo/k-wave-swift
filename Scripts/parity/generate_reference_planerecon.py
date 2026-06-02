#!/usr/bin/env python3
"""Reference for photoacoustic FFT planar reconstruction (`kspacePlaneRecon`). A ball p0 radiates to
a plane of sensors; the recorded series is reconstructed. The Swift port must match.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_planerecon.py`
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
from kwave.kspacePlaneRecon import kspacePlaneRecon
from kwave.utils.mapgen import make_ball

Nx = Ny = Nz = 32
dx = dy = dz = 1e-4
c0, rho0 = 1500.0, 1000.0
PML_SIZE = 8
NSTEPS = 100

kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])
medium = kWaveMedium(sound_speed=c0, density=rho0)
kgrid.makeTime(c0, cfl=0.3)
kgrid.setTime(NSTEPS, float(kgrid.dt))

ball = make_ball(Vector([Nx, Ny, Nz]), Vector([Nx // 2, Ny // 2, Nz // 2]), 3).astype(np.float32)
source = kSource(); source.p0 = ball

# Sensor plane at fixed x (inside the interior); records over all (y, z).
mask = np.zeros((Nx, Ny, Nz), dtype=bool)
mask[10, :, :] = True
sensor = kSensor(); sensor.mask = mask; sensor.record = ["p"]

res = kspaceFirstOrder(kgrid, medium, source, sensor, pml_size=PML_SIZE, pml_inside=True,
                       smooth_p0=True, backend="python", device="cpu", dtype=np.float32, quiet=True)
# Recorded rows are in C-flat order of the plane → (j*Nz + k); reshape to [Ny, Nz, Nt] = yzt.
p_yzt = np.asarray(res["p"], dtype=np.float64).reshape(Ny, Nz, NSTEPS)

recon = kspacePlaneRecon(p_yzt, dy, dz, float(kgrid.dt), c0, data_order="yzt", interp="nearest")

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_planerecon.h5")
with h5py.File(ref, "w") as f:
    f.create_dataset("p_yzt", data=p_yzt.astype(np.float32))
    f.create_dataset("recon", data=recon.astype(np.float32))
    for k, v in [("dy", dy), ("dz", dz), ("dt", float(kgrid.dt)), ("c0", c0)]:
        f.create_dataset(k, data=np.float32(v))

print("wrote", ref, "p_yzt", p_yzt.shape, "recon", recon.shape, "max|recon|", float(np.max(np.abs(recon))))

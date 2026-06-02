#!/usr/bin/env python3
"""Reference for photoacoustic FFT line reconstruction (`kspaceLineRecon`). A disc p0 radiates to a
line of sensors; the recorded series is reconstructed. The Swift port must match.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_linerecon.py`
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
from kwave.kspaceLineRecon import kspaceLineRecon
from kwave.utils.mapgen import make_disc

Nx = Ny = 48
dx = dy = 1e-4
c0, rho0 = 1500.0, 1000.0
PML_SIZE = 10
NSTEPS = 200

kgrid = kWaveGrid([Nx, Ny], [dx, dy])
medium = kWaveMedium(sound_speed=c0, density=rho0)
kgrid.makeTime(c0, cfl=0.3)
kgrid.setTime(NSTEPS, float(kgrid.dt))

disc = make_disc(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), 4).astype(np.float32)
source = kSource(); source.p0 = disc

# Sensor line along y at a fixed x row (near one edge of the interior).
mask = np.zeros((Nx, Ny), dtype=bool)
mask[12, :] = True
sensor = kSensor(); sensor.mask = mask; sensor.record = ["p"]

res = kspaceFirstOrder(kgrid, medium, source, sensor, pml_size=PML_SIZE, pml_inside=True,
                       smooth_p0=True, backend="python", device="cpu", dtype=np.float32, quiet=True)
p_yt = np.asarray(res["p"], dtype=np.float64)          # [Ny, Nt]  (sensor order = ascending y)

recon = kspaceLineRecon(p_yt, dy, float(kgrid.dt), c0, data_order="yt", interp="nearest")

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_linerecon.h5")
with h5py.File(ref, "w") as f:
    f.create_dataset("p_yt", data=p_yt.astype(np.float32))
    f.create_dataset("recon", data=recon.astype(np.float32))
    for k, v in [("dy", dy), ("dt", float(kgrid.dt)), ("c0", c0), ("Ny", Ny), ("Nt", NSTEPS)]:
        f.create_dataset(k, data=np.float32(v))

print("wrote", ref, "p_yt", p_yt.shape, "recon", recon.shape, "max|recon|", float(np.max(np.abs(recon))))

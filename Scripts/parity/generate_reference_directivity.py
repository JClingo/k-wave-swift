#!/usr/bin/env python3
"""Generate a 2D sensor-directivity reference. The NumPy backend does NOT apply directivity, so we
reproduce k-Wave MATLAB `directionalResponse.m` here (verbatim formula) and apply it to the
full-grid pressure recorded by the NumPy solver. The Swift `DirectivityFilter` ports the same
MATLAB formula, so the two implementations are cross-checked at tight tolerance.

MATLAB builds the directional weights on centered k-grids and `fftshift`s them to FFT-natural order;
we build directly in FFT-natural order (np.fft.fftfreq) with no shift, which is equivalent and
matches the Swift port. MATLAB `sinc` == `np.sinc` (both sin(πx)/(πx)).

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_directivity.py`
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
NSTEPS = 120
SIZE = 10 * dx  # directivity element size

kgrid = kWaveGrid([Nx, Ny], [dx, dy])
medium = kWaveMedium(sound_speed=c0, density=rho0)
kgrid.makeTime(c0, cfl=0.3)
kgrid.setTime(NSTEPS, float(kgrid.dt))

p_mask = np.zeros((Nx, Ny), dtype=bool)
p_mask[Nx // 2, Ny // 2] = True
fs = 1.0 / float(kgrid.dt)
signal = tone_burst(fs, 1e6, 3).squeeze().astype(np.float32).reshape(1, -1)
source = kSource(); source.p_mask = p_mask; source.p = signal; source.p_mode = "additive"

# Full-grid pressure time series from the NumPy solver.
sensor = kSensor(); sensor.mask = np.ones((Nx, Ny), dtype=bool); sensor.record = ["p"]
res = kspaceFirstOrder(kgrid, medium, source, sensor, pml_size=PML_SIZE, pml_inside=True,
                       smooth_p0=False, backend="python", device="cpu", dtype=np.float32, quiet=True)
p_full = np.asarray(res["p"], dtype=np.float32)  # [Nx*Ny, Nt], C-order rows

# Line sensor with a spread of directivity angles (mirrors the k-Wave directivity example).
line_mask = np.zeros((Nx, Ny), dtype=bool)
line_mask[23, 1:62:2] = True
n_line = int(line_mask.sum())
angle_grid = np.zeros((Nx, Ny), dtype=np.float32)
angle_grid[line_mask] = (np.linspace(-1, 1, n_line) * np.pi / 2).astype(np.float32)

mask_idx = np.flatnonzero(line_mask)             # C-order ascending
angle_at_idx = angle_grid.ravel()[mask_idx]

# FFT-natural-order k-grids (no fftshift), matching the Swift port.
kx = (2 * np.pi * np.fft.fftfreq(Nx, d=dx)).astype(np.float32)[:, None] * np.ones((1, Ny), np.float32)
ky = (2 * np.pi * np.fft.fftfreq(Ny, d=dy)).astype(np.float32)[None, :] * np.ones((Nx, 1), np.float32)
kmag = np.sqrt(kx**2 + ky**2)


def directional_response_series(pattern):
    out = np.zeros((n_line, NSTEPS), dtype=np.float32)
    for t in range(NSTEPS):
        P = np.fft.fft2(p_full[:, t].reshape(Nx, Ny))
        for theta in np.unique(angle_at_idx):
            rows = np.where(angle_at_idx == theta)[0]
            if pattern == "pressure":
                k_tan = np.cos(theta) * ky - np.sin(theta) * kx
                weight = np.sinc(k_tan * SIZE / 2)  # np.sinc(x) = sin(πx)/(πx) == MATLAB sinc
            else:
                k_norm = np.sin(theta) * ky + np.cos(theta) * kx
                with np.errstate(invalid="ignore", divide="ignore"):
                    weight = np.where(kmag == 0, 0.0, k_norm / kmag)
            filt = np.real(np.fft.ifft2(P * weight))
            out[rows, t] = filt.ravel()[mask_idx[rows]]
    return out


ref_pressure = directional_response_series("pressure")
ref_gradient = directional_response_series("gradient")

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, "reference_2d_directivity.h5")
with h5py.File(ref_path, "w") as f:
    for name, val in [("Nx", Nx), ("Ny", Ny), ("dx", dx), ("dy", dy), ("c0", c0), ("rho0", rho0),
                      ("dt", kgrid.dt), ("Nt", kgrid.Nt), ("pml_size", PML_SIZE),
                      ("src_x", Nx // 2), ("src_y", Ny // 2), ("size", SIZE)]:
        f.create_dataset(name, data=np.float32(val))
    f.create_dataset("signal", data=signal.squeeze().astype(np.float32))
    f.create_dataset("line_mask", data=line_mask.astype(np.float32))
    f.create_dataset("angle_grid", data=angle_grid)
    f.create_dataset("ref_pressure", data=ref_pressure)
    f.create_dataset("ref_gradient", data=ref_gradient)

print("wrote", ref_path, "Nt", int(kgrid.Nt), "n_line", n_line)
print("  ref_pressure max", float(np.max(np.abs(ref_pressure))))
print("  ref_gradient max", float(np.max(np.abs(ref_gradient))))

#!/usr/bin/env python3
"""Reference for the axisymmetric solver (`kspaceFirstOrderAS`, WSWA-FFT radial symmetry).

No runnable oracle exists here: there is no pure-NumPy axisymmetric time loop in k-wave-python,
and the macOS kspaceFirstOrder-OMP binary aborts on axisymmetric inputs ("Cannot create plan 1D
real-to-real FFT of kind 5") — its FFTW build lacks real-to-real transforms. This script therefore
reproduces the WSWA-FFT branch of MATLAB kspaceFirstOrderAS.m (ucl-bug/k-wave) verbatim in NumPy:
the radial dimension is mirrored 4x with per-variable symmetry (p, ux: WSWA; uy: HAHS; uy/r: HSHA)
and propagated with ordinary FFTs. The Swift port implements the same published algorithm
independently; agreement to float32 precision over the full run cross-checks both.

Linear, lossless, homogeneous, p0 source — matching the Swift slice.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_as.py`
"""
import os
import warnings
import numpy as np
import h5py

warnings.filterwarnings("ignore")
from kwave.utils.filters import smooth
from kwave.utils.pml import get_pml

Nx, Ny = 64, 32          # axial, radial
dx = dy = 1e-4
c0, rho0 = 1500.0, 1000.0
PML = 10
PML_ALPHA = 2.0
NSTEPS = 120
CFL = 0.3
dt = CFL * dx / c0

# Gaussian p0 centred on the axis (y index 0 = axis of symmetry), smoothed as in k-Wave.
x = (np.arange(Nx) - Nx // 2)[:, None]
y = np.arange(Ny)[None, :]
p0 = smooth(np.exp(-(x**2 + y**2) / 16.0), restore_max=True).astype(np.float64)

NyE = 4 * Ny
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)[:, None]
kyE = 2 * np.pi * np.fft.fftfreq(NyE, d=dy)[None, :]

ddx_pos = 1j * kx * np.exp(1j * kx * dx / 2)
ddx_neg = 1j * kx * np.exp(-1j * kx * dx / 2)
ddy_k = 1j * kyE
y_shift_pos = np.exp(1j * kyE * dy / 2)
y_shift_neg = np.exp(-1j * kyE * dy / 2)

k_exp = np.sqrt(kx**2 + kyE**2)
arg = c0 * k_exp * dt / 2
kappa = np.where(arg == 0, 1.0, np.sin(arg) / np.maximum(arg, 1e-30))   # k-Wave sinc = sin(x)/x

pml_x = get_pml(Nx, dx, dt, c0, PML, PML_ALPHA, staggered=False, dimension=1).reshape(Nx, 1)
pml_x_sg = get_pml(Nx, dx, dt, c0, PML, PML_ALPHA, staggered=True, dimension=1).reshape(Nx, 1)
pml_y = get_pml(Ny, dy, dt, c0, PML, PML_ALPHA, staggered=False, dimension=2, axisymmetric=True).reshape(1, Ny)
pml_y_sg = get_pml(Ny, dy, dt, c0, PML, PML_ALPHA, staggered=True, dimension=2, axisymmetric=True).reshape(1, Ny)

inv_y_sg = 1.0 / ((np.arange(Ny) + 0.5) * dy)[None, :]


def mirror_wswa(w):
    out = np.zeros((Nx, NyE))
    out[:, :Ny] = w
    out[:, Ny + 1:2 * Ny] = -w[:, 1:][:, ::-1]
    out[:, 2 * Ny:3 * Ny] = -w
    out[:, 3 * Ny + 1:] = w[:, 1:][:, ::-1]
    return out


def mirror_hahs(w):
    return np.concatenate([w, w[:, ::-1], -w, -w[:, ::-1]], axis=1)


def mirror_hsha(w):
    return np.concatenate([w, -w[:, ::-1], -w, w[:, ::-1]], axis=1)


def pressure_gradients(p):
    p_k = kappa * np.fft.fft2(mirror_wswa(p))
    dpdx = np.real(np.fft.ifft2(ddx_pos * p_k))[:, :Ny]
    dpdy = np.real(np.fft.ifft2(ddy_k * y_shift_pos * p_k))[:, :Ny]
    return dpdx, dpdy


mask = np.zeros((Nx, Ny), dtype=bool)
mask[:, 8] = True
mask_idx = np.flatnonzero(mask)

p = np.zeros((Nx, Ny))
ux = np.zeros((Nx, Ny))
uy = np.zeros((Nx, Ny))
rhox = np.zeros((Nx, Ny))
rhoy = np.zeros((Nx, Ny))
p_ts = np.zeros((mask_idx.size, NSTEPS), dtype=np.float64)

for t in range(NSTEPS):
    dpdx, dpdy = pressure_gradients(p)
    ux = pml_x_sg * (pml_x_sg * ux - dt / rho0 * dpdx)
    uy = pml_y_sg * (pml_y_sg * uy - dt / rho0 * dpdy)

    ux_k = kappa * np.fft.fft2(mirror_wswa(ux))
    duxdx = np.real(np.fft.ifft2(ddx_neg * ux_k))[:, :Ny]
    uy_k = ddy_k * np.fft.fft2(mirror_hahs(uy)) + np.fft.fft2(mirror_hsha(inv_y_sg * uy))
    duydy = np.real(np.fft.ifft2(kappa * y_shift_neg * uy_k))[:, :Ny]

    rhox = pml_x * (pml_x * rhox - dt * rho0 * duxdx)
    rhoy = pml_y * (pml_y * rhoy - dt * rho0 * duydy)
    p = c0**2 * (rhox + rhoy)

    if t == 0:
        p = p0.copy()
        rhox = p0 / (2 * c0**2)
        rhoy = p0 / (2 * c0**2)
        dpdx, dpdy = pressure_gradients(p)
        ux = dt / rho0 * dpdx / 2
        uy = dt / rho0 * dpdy / 2

    p_ts[:, t] = p.ravel()[mask_idx]

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_as.h5")
with h5py.File(ref, "w") as f:
    for k, v in [("Nx", Nx), ("Ny", Ny), ("dx", dx), ("dy", dy), ("c0", c0), ("rho0", rho0),
                 ("dt", dt), ("Nt", NSTEPS), ("pml", PML)]:
        f.create_dataset(k, data=np.float32(v))
    f.create_dataset("p0", data=p0.astype(np.float32))
    f.create_dataset("mask", data=mask.astype(np.float32))
    f.create_dataset("p_ts", data=p_ts.astype(np.float32))
    f.create_dataset("p_final", data=p.astype(np.float32))

print("wrote", ref, "max|p_ts|", float(np.max(np.abs(p_ts))),
      "max|p_final|", float(np.max(np.abs(p))))

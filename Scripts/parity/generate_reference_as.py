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


source_kappa = np.cos(arg)

mask = np.zeros((Nx, Ny), dtype=bool)
mask[:, 8] = True
mask_idx = np.flatnonzero(mask)

# Time-varying source: a short disc on the axis, driven by a windowed tone burst.
src_mask = np.zeros((Nx, Ny), dtype=bool)
src_mask[Nx // 4, 0:3] = True
src_idx = np.flatnonzero(src_mask)
tt = np.arange(NSTEPS)
sig = (np.sin(2 * np.pi * 2e6 * tt * dt) * np.exp(-((tt - 30.0) ** 2) / 200.0)).astype(np.float64)


def run(p0_init, p_mode, absorb_tau=None, bona=None):
    """One AS run: p0 IVP (p_mode None) or a time-varying pressure source; optional Stokes
    absorption (tau = -2*db2neper(alpha,2)*c0) and B/A nonlinearity."""
    p = np.zeros((Nx, Ny)); ux = np.zeros((Nx, Ny)); uy = np.zeros((Nx, Ny))
    rhox = np.zeros((Nx, Ny)); rhoy = np.zeros((Nx, Ny))
    ts = np.zeros((mask_idx.size, NSTEPS))
    for t in range(NSTEPS):
        dpdx, dpdy = pressure_gradients(p)
        ux = pml_x_sg * (pml_x_sg * ux - dt / rho0 * dpdx)
        uy = pml_y_sg * (pml_y_sg * uy - dt / rho0 * dpdy)

        ux_k = kappa * np.fft.fft2(mirror_wswa(ux))
        duxdx = np.real(np.fft.ifft2(ddx_neg * ux_k))[:, :Ny]
        uy_k = ddy_k * np.fft.fft2(mirror_hahs(uy)) + np.fft.fft2(mirror_hsha(inv_y_sg * uy))
        duydy = np.real(np.fft.ifft2(kappa * y_shift_neg * uy_k))[:, :Ny]

        if bona is None:
            rhox = pml_x * (pml_x * rhox - dt * rho0 * duxdx)
            rhoy = pml_y * (pml_y * rhoy - dt * rho0 * duydy)
        else:
            rho0_plus_rho = 2 * (rhox + rhoy) + rho0
            rhox = pml_x * (pml_x * rhox - dt * rho0_plus_rho * duxdx)
            rhoy = pml_y * (pml_y * rhoy - dt * rho0_plus_rho * duydy)

        # Pressure source (kspaceFirstOrder_scaleSourceTerms, N = 2 splits, uniform grid).
        if p_mode == "dirichlet":
            val = sig[t] / (2 * c0**2)
            rhox.ravel()[src_idx] = val
            rhoy.ravel()[src_idx] = val
        elif p_mode == "additive":
            mat = np.zeros((Nx, Ny))
            mat.ravel()[src_idx] = sig[t] * 2 * dt / (2 * c0 * dx)
            mat = np.real(np.fft.ifft2(source_kappa * np.fft.fft2(mirror_wswa(mat))))[:, :Ny]
            rhox = rhox + mat
            rhoy = rhoy + mat

        rho_total = rhox + rhoy
        eos = rho_total.copy()
        if absorb_tau is not None:
            eos = eos + absorb_tau * rho0 * (duxdx + duydy)
        if bona is not None:
            eos = eos + bona * rho_total**2 / (2 * rho0)
        p = c0**2 * eos

        if t == 0 and p0_init is not None:
            p = p0_init.copy()
            rhox = p0_init / (2 * c0**2)
            rhoy = p0_init / (2 * c0**2)
            dpdx, dpdy = pressure_gradients(p)
            ux = dt / rho0 * dpdx / 2
            uy = dt / rho0 * dpdy / 2

        ts[:, t] = p.ravel()[mask_idx]
    return ts, p


from kwave.utils.conversion import db2neper

ALPHA_STOKES = 10.0   # dB/(MHz^2 cm)
BONA = 10.0
P0_AMP = 5e6          # [Pa] — large amplitude so the B/A term is significant

tau = -2 * db2neper(ALPHA_STOKES, 2) * c0
ts_ivp, pf_ivp = run(p0, None)
ts_add, pf_add = run(None, "additive")
ts_dir, pf_dir = run(None, "dirichlet")
ts_stokes, pf_stokes = run(p0, None, absorb_tau=tau)
ts_nl, pf_nl = run(P0_AMP * p0, None, bona=BONA)
print("stokes effect rel-l2:", np.linalg.norm(ts_stokes - ts_ivp) / np.linalg.norm(ts_ivp))
print("nonlin effect rel-l2:", np.linalg.norm(ts_nl / P0_AMP - ts_ivp) / np.linalg.norm(ts_ivp))

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_as.h5")
with h5py.File(ref, "w") as f:
    for k, v in [("Nx", Nx), ("Ny", Ny), ("dx", dx), ("dy", dy), ("c0", c0), ("rho0", rho0),
                 ("dt", dt), ("Nt", NSTEPS), ("pml", PML)]:
        f.create_dataset(k, data=np.float32(v))
    f.create_dataset("p0", data=p0.astype(np.float32))
    f.create_dataset("mask", data=mask.astype(np.float32))
    f.create_dataset("src_mask", data=src_mask.astype(np.float32))
    f.create_dataset("sig", data=sig.astype(np.float32))
    f.create_dataset("alpha_stokes", data=np.float32(ALPHA_STOKES))
    f.create_dataset("bona", data=np.float32(BONA))
    f.create_dataset("p0_amp", data=np.float32(P0_AMP))
    for name, (ts, pf) in [("ivp", (ts_ivp, pf_ivp)), ("add", (ts_add, pf_add)),
                           ("dir", (ts_dir, pf_dir)), ("stokes", (ts_stokes, pf_stokes)),
                           ("nl", (ts_nl, pf_nl))]:
        f.create_dataset(f"p_ts_{name}", data=ts.astype(np.float32))
        f.create_dataset(f"p_final_{name}", data=pf.astype(np.float32))

print("wrote", ref, "|ivp|", float(np.max(np.abs(ts_ivp))),
      "|add|", float(np.max(np.abs(ts_add))), "|dir|", float(np.max(np.abs(ts_dir))),
      "|stokes|", float(np.max(np.abs(ts_stokes))), "|nl|", float(np.max(np.abs(ts_nl))))

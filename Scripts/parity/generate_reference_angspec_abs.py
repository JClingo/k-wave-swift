#!/usr/bin/env python3
"""Reference for the absorbing branch of broadband `angular_spectrum`.

The upstream absorbing path is dead code (it reads `medium.alpha_coeff` as an attribute, but
`medium` is a dict by that point → AttributeError), so this script reproduces the function's core
loop verbatim — including Eq. 11 `H *= exp(-alpha_Np*z*k/kz)` with complex kz — with that one-line
access fixed. Odd Nt only (the even-Nt rebuild has its own upstream off-by-one).

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_angspec_abs.py`
"""
import os
import numpy as np
import h5py

from kwave.utils.conversion import db2neper
from kwave.utils.filters import next_pow2

Nx = Ny = 16
Nt = 21
dx = 1e-4
dt = 2e-8
c0 = 1500.0
alpha_coeff, alpha_power = 0.75, 1.5
z = 1e-3

# Same smooth input as the lossless reference.
x = (np.arange(Nx) - Nx // 2)[:, None, None]
y = (np.arange(Ny) - Ny // 2)[None, :, None]
t = np.arange(Nt)[None, None, :]
f0 = 5e6
input_plane = (np.exp(-(x**2 + y**2) / 18.0)
               * np.sin(2 * np.pi * f0 * t * dt) * np.exp(-((t - Nt / 2) ** 2) / 40.0))

# --- core loop of angular_spectrum, absorbing, verbatim except medium["..."] access ------------
fft_length = int(2 ** (next_pow2(max(Nx, Ny)) + 1))
N = fft_length
k_vec = np.arange(-N // 2, N // 2) * 2 * np.pi / (N * dx)
k_vec[N // 2] = 0
k_vec = np.fft.ifftshift(k_vec)
ky, kx = np.meshgrid(k_vec, k_vec, indexing="ij")
sqrt_kx2_ky2 = np.sqrt(kx**2 + ky**2)

spec = np.fft.fft(input_plane, axis=2)
num_unique_pts = int(np.ceil((Nt + 1) / 2))
spec = spec[:, :, :num_unique_pts]
f_vec = np.arange(num_unique_pts) / (dt * Nt)
f_vec_prop = f_vec[f_vec < (c0 / (2 * dx))]

P = np.zeros((Nx, Ny, num_unique_pts), dtype=np.complex128)
for fi in range(len(f_vec_prop)):
    k = 2 * np.pi * f_vec[fi] / c0
    kz = np.sqrt((k**2 - (kx**2 + ky**2)).astype(complex))
    H = np.conj(np.exp(1j * z * kz))
    alpha_Np = db2neper(alpha_coeff, alpha_power) * (2 * np.pi * f_vec[fi]) ** alpha_power
    if alpha_Np != 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            H = H * np.exp(-alpha_Np * z * k / kz)
        H[np.isnan(H) | np.isinf(H)] = 0  # kz == 0 ring (does not hit grid points here)
    D = (fft_length - 1) * dx
    kc = k * np.sqrt(0.5 * D**2 / (0.5 * D**2 + z**2))
    H[sqrt_kx2_ky2 > kc] = 0
    xy_fft = np.fft.fft2(spec[:, :, fi], s=(fft_length, fft_length))
    step = np.fft.ifft2(xy_fft * H, s=(fft_length, fft_length))
    P[:, :, fi] = step[:Nx, :Ny] * np.exp(1j * 2 * np.pi * f_vec[fi] * z / c0)

P_exp = np.concatenate((P, np.flip(np.conj(P[:, :, 1:]), axis=2)), axis=2)  # odd Nt
series = np.real(np.fft.ifft(P_exp, axis=2))
pmax = np.max(series, axis=2)

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_angspec_abs.h5")
with h5py.File(ref, "w") as f:
    for kk, v in [("dx", dx), ("dt", dt), ("c0", c0), ("z", z),
                  ("alpha_coeff", alpha_coeff), ("alpha_power", alpha_power)]:
        f.create_dataset(kk, data=np.float32(v))
    f.create_dataset("input", data=input_plane.astype(np.float32))
    f.create_dataset("pmax", data=pmax.astype(np.float32))
    f.create_dataset("ptime", data=series.astype(np.float32))

print("wrote", ref, "pmax max", round(float(pmax.max()), 6))

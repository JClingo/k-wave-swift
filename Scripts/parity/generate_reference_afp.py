#!/usr/bin/env python3
"""Reference for `acousticFieldPropagator` (exact CW Green's-function propagation).

k-wave-python has no port of MATLAB acousticFieldPropagator.m, so this script reproduces the
MATLAB algorithm verbatim in NumPy float64 (the non-first-order path: expanded grid, exact
propagator with singular-bin patches, single FFT). The Swift port implements the same published
formulas independently; agreement cross-checks both.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_afp.py`
"""
import os
import numpy as np
import h5py

MACHINE_PRECISION = np.finfo(float).eps * 10


def largest_prime_factor(n):
    best, m, d = 1, n, 2
    while d * d <= m:
        while m % d == 0:
            best = max(best, d)
            m //= d
        d += 1
    return max(best, m if m > 1 else 1)


def optimal_grid_dim(n, search_range):
    facs = [largest_prime_factor(n + i) for i in range(search_range + 1)]
    return n + int(np.argmin(facs))


def afp(amp_in, phase_in, dx, f0, c0, use_ramp=True,
        time_exp_factor=1.5, grid_exp_factor=1.1, grid_search_range=50):
    sz = amp_in.shape
    w0 = 2 * np.pi * f0
    t = time_exp_factor * dx * np.sqrt(sum(s**2 for s in sz)) / c0
    expansion = int(np.ceil(grid_exp_factor * t * c0 / dx))
    sz_ex = tuple(optimal_grid_dim(s + expansion, grid_search_range) for s in sz)

    # |k| on the expanded grid (FFT-natural order).
    k_sq = np.zeros(sz_ex)
    for axis, N in enumerate(sz_ex):
        kv = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        shape = [1] * len(sz_ex)
        shape[axis] = N
        k_sq = k_sq + kv.reshape(shape) ** 2
    k = np.sqrt(k_sq)

    eps = k.max() * MACHINE_PRECISION
    with np.errstate(divide="ignore", invalid="ignore"):
        if not use_ramp:
            prop = (1j * w0 * c0 * k * (np.exp(1j * w0 * t) - np.cos(c0 * k * t))
                    + w0**2 * np.sin(c0 * k * t)) / ((c0 * k) ** 3 - c0 * k * w0**2) \
                + np.sin(c0 * k * t) / (c0 * k)
            prop[np.abs(k - w0 / c0) < eps] = (w0 * t * np.exp(1j * w0 * t) + np.sin(w0 * t)) / (2 * w0)
            prop[k == 0] = (1j - 1j * np.exp(1j * w0 * t)) / w0
        else:
            prop = (-2j * np.exp(1j * w0 * t) * w0 * (16 * c0**4 * k**4 - 40 * c0**2 * k**2 * w0**2 + 9 * w0**4)
                    - 3j * w0**3 * (4 * c0**2 * k**2 + w0**2) * np.cos(c0 * k * t)
                    - 3j * w0**3 * (4 * c0**2 * k**2 + w0**2) * np.cos(c0 * k * (t - 2 * np.pi / w0))
                    + c0 * k * w0**2 * (4 * c0**2 * k**2 + 11 * w0**2)
                    * (np.sin(c0 * k * t) + np.sin(c0 * k * (t - 2 * np.pi / w0)))) \
                / (-32 * c0**6 * k**6 + 112 * c0**4 * k**4 * w0**2 - 98 * c0**2 * k**2 * w0**4 + 18 * w0**6)
            prop[np.abs(k - w0 / c0) < eps] = \
                (-1j - 15 * np.exp(2j * w0 * t) * (1j + 2 * np.pi - 2 * w0 * t)) / (np.exp(1j * w0 * t) * 60 * w0)
            prop[np.abs(k - w0 / (2 * c0)) < eps] = \
                -(16j * np.exp(1j * w0 * t) + 3 * np.exp(1j * w0 * t / 2) * np.pi) / (12 * w0)
            prop[np.abs(k - 3 * w0 / (2 * c0)) < eps] = \
                (16j * np.exp(1j * w0 * t) - 5 * np.exp(3j * w0 * t / 2) * np.pi) / (20 * w0)
            prop[k == 0] = -1j * (1 + 3 * np.exp(1j * w0 * t)) / (3 * w0)

    src = np.zeros(sz_ex, dtype=complex)
    src[tuple(slice(0, s) for s in sz)] = amp_in * np.exp(1j * phase_in)
    p = np.fft.ifftn(prop * np.fft.fftn(src))
    p = p[tuple(slice(0, s) for s in sz)] * 2 * c0 / dx
    return p, sz_ex


dx, f0, c0 = 1e-4, 2e6, 1500.0

# 2D: focused arc-ish phase profile.
Nx, Ny = 64, 48
amp2 = np.zeros((Nx, Ny))
amp2[4, 8:40] = 1.0
y = np.arange(8, 40)
phase2 = np.zeros((Nx, Ny))
phase2[4, 8:40] = 2 * np.pi * f0 / c0 * (np.sqrt((40 * 1e-4) ** 2 + ((y - 24) * dx) ** 2))
p2, sz_ex2 = afp(amp2, phase2, dx, f0, c0)
p2_nr, _ = afp(amp2, phase2, dx, f0, c0, use_ramp=False)

# 3D: small plane source.
N3 = (24, 20, 18)
amp3 = np.zeros(N3)
amp3[3, 6:14, 5:13] = 1.0
phase3 = np.zeros(N3)
p3, sz_ex3 = afp(amp3, phase3, dx, f0, c0)

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_afp.h5")
with h5py.File(ref, "w") as f:
    for k_, v in [("dx", dx), ("f0", f0), ("c0", c0)]:
        f.create_dataset(k_, data=np.float32(v))
    f.create_dataset("amp2", data=amp2.astype(np.float32))
    f.create_dataset("phase2", data=phase2.astype(np.float32))
    f.create_dataset("p2_re", data=np.real(p2).astype(np.float32))
    f.create_dataset("p2_im", data=np.imag(p2).astype(np.float32))
    f.create_dataset("p2nr_re", data=np.real(p2_nr).astype(np.float32))
    f.create_dataset("p2nr_im", data=np.imag(p2_nr).astype(np.float32))
    f.create_dataset("amp3", data=amp3.astype(np.float32))
    f.create_dataset("p3_re", data=np.real(p3).astype(np.float32))
    f.create_dataset("p3_im", data=np.imag(p3).astype(np.float32))

print("wrote", ref, "sz_ex2", sz_ex2, "sz_ex3", sz_ex3,
      "max|p2|", round(float(np.abs(p2).max()), 4), "max|p3|", round(float(np.abs(p3).max()), 4))

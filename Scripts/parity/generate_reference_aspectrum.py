#!/usr/bin/env python3
"""Reference for CW angular-spectrum projection (k-wave-python `angular_spectrum_cw`): a disc piston
source projected to several parallel planes. The Swift `angularSpectrumCW` must match (complex).

`z_pos`'s type hint rejects arrays, so we call once per scalar z and stack.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_aspectrum.py`
"""
import os
import numpy as np
import h5py
from kwave.data import Vector
from kwave.utils.mapgen import make_disc
from kwave.utils.angular_spectrum_cw import angular_spectrum_cw

Nx = Ny = 32
dx = 2e-4
f0 = 1_000_000
c0 = 1500.0
zpos = [0.0, 4e-3, 8e-3]

inp = make_disc(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), 5).astype(np.float64)
sl = [np.asarray(angular_spectrum_cw(inp, dx, float(z), f0, {"sound_speed": c0},
                                     angular_restriction=True))[:, :, 0] for z in zpos]
out = np.stack(sl, axis=2)

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_aspectrum.h5")
with h5py.File(ref, "w") as f:
    f.create_dataset("input", data=inp.astype(np.float32))
    f.create_dataset("out_re", data=np.real(out).astype(np.float32))
    f.create_dataset("out_im", data=np.imag(out).astype(np.float32))
    for k, v in [("Nx", Nx), ("Ny", Ny), ("dx", dx), ("f0", f0), ("c0", c0), ("Nz", len(zpos))]:
        f.create_dataset(k, data=np.float32(v))
    f.create_dataset("zpos", data=np.array(zpos, np.float32))

print("wrote", ref, "out shape", out.shape, "max|out|", float(np.max(np.abs(out))))

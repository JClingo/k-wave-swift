#!/usr/bin/env python3
"""Reference for `makeDTTDim` (DTT wavenumber vectors + implied periods), all 8 transform types.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_dttdim.py`
"""
import os
import numpy as np
import h5py
from kwave.kgrid import kWaveGrid
from kwave.enums import DiscreteCosine, DiscreteSine

N, dx = 9, 1e-4
g = kWaveGrid([N, N], [dx, dx])

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_dttdim.h5")
with h5py.File(ref, "w") as f:
    f.create_dataset("N", data=np.float32(N))
    f.create_dataset("dx", data=np.float32(dx))
    for prefix, enum in (("dct", DiscreteCosine), ("dst", DiscreteSine)):
        for t in (1, 2, 3, 4):
            kvec, M = g.kx_vec_dtt(enum(t))
            f.create_dataset(f"{prefix}{t}_k", data=np.asarray(kvec, np.float32))
            f.create_dataset(f"{prefix}{t}_M", data=np.float32(M))
            print(f"{prefix}{t}: len={len(kvec)} M={M} k0={kvec[0]:.3f} kEnd={kvec[-1]:.3f}")
print("wrote", ref)

#!/usr/bin/env python3
"""Reference for `cart2grid` binary-mask output (Cartesian points → nearest grid nodes), 2D and 3D.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_cart2grid.py`
"""
import os
import warnings
import numpy as np
import h5py

warnings.filterwarnings("ignore")
from kwave.kgrid import kWaveGrid
from kwave.data import Vector
from kwave.utils.conversion import cart2grid
from kwave.utils.mapgen import make_cart_circle

g2 = kWaveGrid(Vector([50, 50]), Vector([1e-4, 1e-4]))
c2 = make_cart_circle(15 * 1e-4, 40).astype(np.float64)
m2, _, _ = cart2grid(g2, c2, order="C")

g3 = kWaveGrid(Vector([30, 30, 30]), Vector([1e-4, 1e-4, 1e-4]))
ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
c3 = np.vstack([10e-4 * np.cos(ang), 10e-4 * np.sin(ang), np.zeros(20)]).astype(np.float64)
m3, _, _ = cart2grid(g3, c3, order="C")

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_cart2grid.h5")
with h5py.File(ref, "w") as f:
    f.create_dataset("c2", data=c2.astype(np.float32))
    f.create_dataset("m2", data=m2.astype(np.float32))
    f.create_dataset("c3", data=c3.astype(np.float32))
    f.create_dataset("m3", data=m3.astype(np.float32))

print("wrote", ref, "m2 sum", int(m2.sum()), "m3 sum", int(m3.sum()))

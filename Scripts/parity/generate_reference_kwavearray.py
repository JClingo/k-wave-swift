#!/usr/bin/env python3
"""Reference for `KWaveArray` 2D element grid weights (arc, line, rect) and the array sum.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_kwavearray.py`
"""
import os
import warnings
import numpy as np
import h5py

warnings.filterwarnings("ignore")
from kwave.kgrid import kWaveGrid
from kwave.data import Vector
from kwave.utils.kwave_array import kWaveArray

g = kWaveGrid(Vector([48, 40]), Vector([1e-4, 1e-4]))
arr = kWaveArray(bli_tolerance=0.1, upsampling_rate=10)
arr.add_arc_element(position=[-1.2e-3, 0.0], radius=2e-3, diameter=1.5e-3, focus_pos=[1e-3, 0.0])
arr.add_line_element(start_point=[0.4e-3, -1.1e-3], end_point=[1.3e-3, 0.7e-3])
arr.add_rect_element(position=[-0.5e-3, 1.2e-3], Lx=0.8e-3, Ly=0.4e-3, theta=25.0)

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_kwavearray.h5")
with h5py.File(ref, "w") as f:
    for i, name in enumerate(["arc", "line", "rect"]):
        w = np.asarray(arr.get_element_grid_weights(g, i))
        f.create_dataset(f"w_{name}", data=w.astype(np.float32))
        print(name, "max", round(float(np.max(np.abs(w))), 4), "sum", round(float(w.sum()), 4))
    W = np.asarray(arr.get_array_grid_weights(g))
    f.create_dataset("w_all", data=W.astype(np.float32))

print("wrote", ref)

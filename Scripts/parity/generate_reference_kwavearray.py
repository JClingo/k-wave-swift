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

mask = np.asarray(arr.get_array_binary_mask(g))
n_pts = int(mask.sum())

# Per-element signals; distribute onto grid points (C order, matching the Swift convention).
Nt = 50
t = np.arange(Nt)
sig = np.vstack([np.sin(2 * np.pi * t / 20.0 + ph) for ph in (0.0, 0.7, -1.1)])
dist = np.asarray(arr.get_distributed_source_signal(g, sig, order="C"))

# Synthetic per-grid-point sensor data; combine back to elements (C order).
rng = np.random.default_rng(3)
sensor_data = rng.standard_normal((n_pts, Nt))
combined = np.asarray(arr.combine_sensor_data(g, sensor_data, order="C"))

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_kwavearray.h5")
with h5py.File(ref, "w") as f:
    for i, name in enumerate(["arc", "line", "rect"]):
        w = np.asarray(arr.get_element_grid_weights(g, i))
        f.create_dataset(f"w_{name}", data=w.astype(np.float32))
        print(name, "max", round(float(np.max(np.abs(w))), 4), "sum", round(float(w.sum()), 4))
    W = np.asarray(arr.get_array_grid_weights(g))
    f.create_dataset("w_all", data=W.astype(np.float32))
    f.create_dataset("mask", data=mask.astype(np.float32))
    f.create_dataset("sig", data=sig.astype(np.float32))
    f.create_dataset("dist", data=dist.astype(np.float32))
    f.create_dataset("sensor_data", data=sensor_data.astype(np.float32))
    f.create_dataset("combined", data=combined.astype(np.float32))

print("wrote", ref, "mask pts", n_pts, "dist", dist.shape, "combined", combined.shape)

# --- disc / bowl samplers + elements ---------------------------------------
from kwave.utils.mapgen import make_cart_disc, make_cart_bowl

disc2 = make_cart_disc(np.array([0.3e-3, -0.2e-3]), 0.6e-3, None, 60, False, False)
disc3 = make_cart_disc(np.array([0.2e-3, -0.1e-3, 0.4e-3]), 0.5e-3,
                       np.array([1.0e-3, 0.8e-3, -0.5e-3]), 60, False, False)
bowl = make_cart_bowl(np.array([-1.0e-3, 0.0, 0.0]), 2.0e-3, 1.4e-3,
                      np.array([1.0e-3, 0.2e-3, 0.1e-3]), 80)

g3 = kWaveGrid(Vector([32, 28, 24]), Vector([1e-4, 1e-4, 1e-4]))
arr2 = kWaveArray(bli_tolerance=0.1, upsampling_rate=10)
arr2.add_disc_element(position=[0.3e-3, -0.2e-3], diameter=1.2e-3)
w_disc2 = np.asarray(arr2.get_element_grid_weights(g, 0))

arr3 = kWaveArray(bli_tolerance=0.1, upsampling_rate=10)
arr3.add_bowl_element(position=[-1.0e-3, 0.0, 0.0], radius=2.0e-3, diameter=1.4e-3,
                      focus_pos=[1.0e-3, 0.2e-3, 0.1e-3])
w_bowl = np.asarray(arr3.get_element_grid_weights(g3, 0))

with h5py.File(ref, "a") as f:
    f.create_dataset("disc2", data=np.asarray(disc2, np.float32))
    f.create_dataset("disc3", data=np.asarray(disc3, np.float32))
    f.create_dataset("bowl", data=np.asarray(bowl, np.float32))
    f.create_dataset("w_disc2", data=w_disc2.astype(np.float32))
    f.create_dataset("w_bowl", data=w_bowl.astype(np.float32))

print("disc2", disc2.shape, "disc3", disc3.shape, "bowl", bowl.shape,
      "w_disc2 sum", round(float(w_disc2.sum()), 4), "w_bowl sum", round(float(w_bowl.sum()), 4))

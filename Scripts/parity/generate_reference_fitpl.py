#!/usr/bin/env python3
"""Reference for `fit_power_law_params` (Nelder-Mead fit of absorption parameters).

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_fitpl.py`
"""
import os
import warnings
import numpy as np
import h5py

warnings.filterwarnings("ignore")
from kwave.utils.mapgen import fit_power_law_params

cases = {
    "c1": (0.75, 1.5, 1500.0, 0.5e6, 3e6),
    "c2": (0.5, 1.1, 1540.0, 1e6, 5e6),
}

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_fitpl.h5")
with h5py.File(ref, "w") as f:
    for key, (a0, y, c0, fmin, fmax) in cases.items():
        a0_fit, y_fit = fit_power_law_params(a0, y, c0, fmin, fmax)
        f.create_dataset(key + "_in", data=np.array([a0, y, c0, fmin, fmax], np.float32))
        f.create_dataset(key + "_out", data=np.array([float(a0_fit), float(y_fit)], np.float32))
        print(key, "a0_fit", float(a0_fit), "y_fit", float(y_fit))

print("wrote", ref)

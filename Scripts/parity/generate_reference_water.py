#!/usr/bin/env python3
"""Reference for water material-property functions (water_sound_speed/density/absorption/
non_linearity). The Swift ports must match these scalar values.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_water.py`
"""
import os
import numpy as np
import h5py
from kwave.utils.mapgen import water_sound_speed, water_density, water_absorption, water_non_linearity

ss_t = [0, 10, 20, 37, 60, 95]
de_t = [5, 15, 25, 37]
ab_t = [0, 20, 37, 60]
ab_f = 2.0
nl_t = [0, 25, 50, 100]

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_water.h5")
with h5py.File(ref, "w") as f:
    f.create_dataset("ss_temp", data=np.array(ss_t, np.float32))
    f.create_dataset("ss", data=np.array([water_sound_speed(t) for t in ss_t], np.float32))
    f.create_dataset("de_temp", data=np.array(de_t, np.float32))
    f.create_dataset("de", data=np.array([water_density(t) for t in de_t], np.float32))
    f.create_dataset("ab_temp", data=np.array(ab_t, np.float32))
    f.create_dataset("ab_f", data=np.float32(ab_f))
    f.create_dataset("ab", data=np.array([water_absorption(ab_f, t) for t in ab_t], np.float32))
    f.create_dataset("nl_temp", data=np.array(nl_t, np.float32))
    f.create_dataset("nl", data=np.array([water_non_linearity(t) for t in nl_t], np.float32))
print("wrote", ref)

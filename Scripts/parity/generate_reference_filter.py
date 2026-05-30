#!/usr/bin/env python3
"""Generate apply_filter / filter_time_series references for the Swift parity tests.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_filter.py`
"""
import os
import numpy as np
import h5py

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.utils.filters import apply_filter, filter_time_series

fs = 10e6
n = 128
t = np.arange(n) / fs
signal = (np.sin(2 * np.pi * 0.5e6 * t) + 0.5 * np.sin(2 * np.pi * 3e6 * t)).reshape(1, n)

lp = np.asarray(apply_filter(signal, fs, 1e6, "LowPass")).squeeze()
hp = np.asarray(apply_filter(signal, fs, 1e6, "HighPass")).squeeze()
bp = np.asarray(apply_filter(signal, fs, [0.3e6, 1.5e6], "BandPass")).squeeze()

# filter_time_series
kgrid = kWaveGrid([64, 64], [1e-4, 1e-4])
medium = kWaveMedium(sound_speed=1500.0)
kgrid.makeTime(medium.sound_speed, cfl=0.3)
sig2 = np.sin(2 * np.pi * 2e6 * (np.arange(kgrid.Nt) * kgrid.dt)).reshape(1, kgrid.Nt)
fts = np.asarray(filter_time_series(kgrid, medium, sig2.copy())).squeeze()

out_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(out_dir, "reference_filter.h5")
with h5py.File(ref_path, "w") as f:
    f.create_dataset("signal", data=signal.squeeze().astype(np.float32))
    f.create_dataset("lp", data=lp.astype(np.float32))
    f.create_dataset("hp", data=hp.astype(np.float32))
    f.create_dataset("bp", data=bp.astype(np.float32))
    f.create_dataset("fs", data=np.float32(fs))
    f.create_dataset("sig2", data=sig2.squeeze().astype(np.float32))
    f.create_dataset("fts", data=fts.astype(np.float32))
    f.create_dataset("dt", data=np.float32(kgrid.dt))
    f.create_dataset("Nt", data=np.float32(kgrid.Nt))
print("wrote", ref_path, "lp0", float(lp[0]), "hp0", float(hp[0]), "fts max", float(np.max(np.abs(fts))))

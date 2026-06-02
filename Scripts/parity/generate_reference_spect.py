#!/usr/bin/env python3
"""Reference for spectral utilities `spect` (single-sided amplitude/phase) and `extract_amp_phase`.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_spect.py`
"""
import os
import numpy as np
import h5py
from kwave.utils.filters import spect, extract_amp_phase

fs = 10e6
n = 200
t = np.arange(n) / fs
sig = (1.3 * np.sin(2 * np.pi * 1e6 * t + 0.4) + 0.7 * np.sin(2 * np.pi * 2.5e6 * t - 1.1)).astype(np.float64)

f, a, p = spect(sig, fs)                                       # rectangular window (default)
amp, ph, fr = extract_amp_phase(sig.reshape(1, -1), fs, 1e6)   # Hanning, fft_padding=3

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_spect.h5")
with h5py.File(ref, "w") as h:
    h.create_dataset("sig", data=sig.astype(np.float32))
    h.create_dataset("fs", data=np.float32(fs))
    h.create_dataset("f", data=np.asarray(f, np.float32))
    h.create_dataset("amp", data=np.asarray(a, np.float32))
    h.create_dataset("phase", data=np.asarray(p, np.float32))
    h.create_dataset("eap_freq", data=np.float32(1e6))
    h.create_dataset("eap_amp", data=np.float32(float(amp)))
    h.create_dataset("eap_phase", data=np.float32(float(ph)))
    h.create_dataset("eap_f", data=np.float32(float(fr)))

print("wrote", ref, "amp bins", len(a), "extract amp", float(amp), "phase", float(ph))

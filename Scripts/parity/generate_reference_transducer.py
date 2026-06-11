#!/usr/bin/env python3
"""Reference for `KWaveTransducer` (linear-array transducer model) vs k-wave-python
`kWaveTransducerSimple`/`NotATransducer`: masks, beamforming delays, delay mask, apodization,
input-signal padding, scan-line beamforming, and sensor-data combination.

Ordering note: python's combine_sensor_data pairs rows in MATLAB F-order of the x = 0 plane;
the Swift port uses this codebase's C-order convention. The per-voxel series here are keyed by
voxel coordinates and exported in both orders so the two implementations see identical data.

Run: `.venv-kwave/bin/python Scripts/parity/generate_reference_transducer.py`
"""
import os
import warnings
import numpy as np
import h5py

warnings.filterwarnings("ignore")
from kwave.kgrid import kWaveGrid
from kwave.data import Vector
from kwave.ktransducer import kWaveTransducerSimple, NotATransducer
from kwave.utils.signals import tone_burst

Nx, Ny, Nz = 12, 64, 16
dx = 1e-4
kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dx, dx]))
kgrid.setTime(300, 2e-8)

geometry = kWaveTransducerSimple(
    kgrid, number_elements=8, element_width=4, element_length=12, element_spacing=2,
    position=[1, 5, 3],          # 1-based (MATLAB convention)
)

active = np.ones((8, 1))
active[0] = 0
active[1] = 0                    # first two elements inactive → renumbering exercised

fs = 1 / 2e-8
signal = np.squeeze(tone_burst(fs, 1e6, 3)).astype(np.float64)

t = NotATransducer(
    geometry, kgrid, active_elements=active,
    focus_distance=30e-3, elevation_focus_distance=20e-3,
    receive_apodization="Hanning", transmit_apodization="Hanning",
    sound_speed=1540, input_signal=signal, steering_angle=10.0,
)

n_active = int(t.number_active_elements)
all_mask = np.asarray(t.all_elements_mask, dtype=np.float32).reshape(Nx, Ny, Nz)
act_mask = np.asarray(t.active_elements_mask, dtype=np.float32).reshape(Nx, Ny, Nz)
idx_act = np.asarray(t.indexed_active_elements_mask, dtype=np.float32).reshape(Nx, Ny, Nz)
bf = np.asarray(t.beamforming_delays, dtype=np.float32)
elev = np.asarray(t.elevation_beamforming_delays, dtype=np.float32).squeeze()
dmask = np.asarray(t.delay_mask(), dtype=np.float32).reshape(Nx, Ny, Nz)
apod_mask = np.asarray(t.transmit_apodization_mask, dtype=np.float32).reshape(Nx, Ny, Nz)
padded = np.asarray(t.input_signal, dtype=np.float32).squeeze()

# Per-voxel deterministic series keyed by (y, z) so row ordering is irrelevant.
Nt = 60
tt = np.arange(Nt)


def series(y, z):
    return np.sin(0.13 * y + 0.31 * z + 0.05 * tt).astype(np.float64)


# Python row order: F-order of the x=0 plane (mask[0].T flattened), active voxels only.
plane_idx = np.asarray(t.indexed_active_elements_mask)[0]      # [Ny, Nz]
py_rows = [(y, z) for z in range(Nz) for y in range(Ny) if plane_idx[y, z] > 0]
sd_py = np.vstack([series(y, z) for (y, z) in py_rows])
combined = np.asarray(t.combine_sensor_data(sd_py.copy()), dtype=np.float32)

# Swift row order: C-flat ascending over the full grid (x plane is at index 0 here).
act_flat = act_mask.ravel()
sw_rows = []
for flat in np.flatnonzero(act_flat):
    x, rem = divmod(flat, Ny * Nz)
    y, z = divmod(rem, Nz)
    sw_rows.append((y, z))
sd_swift = np.vstack([series(y, z) for (y, z) in sw_rows]).astype(np.float32)

# Scan line on deterministic per-element data.
sd_elements = np.vstack([np.cos(0.21 * e + 0.07 * tt) for e in range(n_active)]).astype(np.float64)
line = np.asarray(t.scan_line(sd_elements.copy()), dtype=np.float32)

ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_transducer.h5")
with h5py.File(ref, "w") as f:
    f.create_dataset("all_mask", data=all_mask)
    f.create_dataset("act_mask", data=act_mask)
    f.create_dataset("idx_act", data=idx_act)
    f.create_dataset("bf", data=bf)
    f.create_dataset("elev", data=elev)
    f.create_dataset("dmask", data=dmask)
    f.create_dataset("apod_mask", data=apod_mask)
    f.create_dataset("padded", data=padded)
    f.create_dataset("signal", data=signal.astype(np.float32))
    f.create_dataset("sd_swift", data=sd_swift)
    f.create_dataset("combined", data=combined)
    f.create_dataset("sd_elements", data=sd_elements.astype(np.float32))
    f.create_dataset("line", data=line)

print("wrote", ref, "n_active", n_active, "bf", bf.astype(int).tolist(),
      "padded len", len(padded), "(signal", len(signal), ")")

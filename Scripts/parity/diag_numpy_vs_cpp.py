#!/usr/bin/env python3
"""Diagnostic: run the k-wave-python NUMPY solver (kspace_solver.py) on the SAME heterogeneous
inputs used to build reference_2d_hetero.h5 (which was produced by the C++ engine), and compare.

Hypothesis: the Swift port mirrors the numpy solver line-for-line, so numpy-vs-C++ should reproduce
the ~2% L2 / ~7% max gap the Swift parity test sees. If so, the gap is numpy-formulation vs the
C++ engine, not a Swift bug.
"""
import os
import numpy as np
import h5py

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.solvers.kspace_solver import Simulation

here = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(here, "reference_2d_hetero.h5")
with h5py.File(ref_path, "r") as f:
    Nx = int(f["Nx"][()]); Ny = int(f["Ny"][()])
    dx = float(f["dx"][()]); dy = float(f["dy"][()])
    dt = float(f["dt"][()]); Nt = int(f["Nt"][()])
    pml = int(f["pml_size"][()])
    c0 = np.array(f["c0"], dtype=np.float32)
    rho0 = np.array(f["rho0"], dtype=np.float32)
    p0 = np.array(f["p0"], dtype=np.float32)
    p_final_cpp = np.array(f["p_final"], dtype=np.float32)

kgrid = kWaveGrid([Nx, Ny], [dx, dy])
kgrid.setTime(Nt, dt)
medium = kWaveMedium(sound_speed=c0, density=rho0)
source = kSource(); source.p0 = p0
sensor = kSensor(); sensor.mask = np.ones((Nx, Ny), dtype=bool); sensor.record = ["p_final"]

sim = Simulation(kgrid, medium, source, sensor, device="cpu", smooth_p0=True,
                 pml_size=(pml, pml), pml_alpha=(2.0, 2.0), dtype=np.float32)
sim.setup()
print("numpy solver pml_sizes:", getattr(sim, "pml_sizes", None))
res = sim.run()
p_final_np = np.asarray(res["p_final"], dtype=np.float32)

print("cpp shape", p_final_cpp.shape, "np shape", p_final_np.shape)
if p_final_np.shape != p_final_cpp.shape:
    # numpy strips the PML; slice the C++ full-grid ref to the same interior region.
    ps = sim.pml_sizes
    sl = tuple(slice(s, N - s if s else None) for s, N in zip(ps, p_final_cpp.shape))
    p_final_cpp = p_final_cpp[sl]
    print("sliced cpp to interior:", p_final_cpp.shape)

diff = np.abs(p_final_np - p_final_cpp)
refmax = np.max(np.abs(p_final_cpp))
l2 = np.sqrt(np.mean(diff**2)); refl2 = np.sqrt(np.mean(p_final_cpp**2))
print("--- INTERIOR (PML-stripped) numpy vs cpp ---")
print(f"max|cpp|={refmax:.6e}  maxErr={diff.max():.6e}  maxErr/refMax={diff.max()/refmax:.4e}")
print(f"l2/refL2={l2/refl2:.4e}")
yi, xi = np.unravel_index(np.argmax(diff), diff.shape)
print(f"peak err at interior ({yi},{xi})  cpp={p_final_cpp[yi,xi]:.4e} np={p_final_np[yi,xi]:.4e}")

# FULL-grid numpy field (sim.p) vs full C++ grid — should match the Swift test's full-grid gap.
with h5py.File(ref_path, "r") as f:
    cpp_full = np.array(f["p_final"], dtype=np.float32)
np_full = np.asarray(sim.p, dtype=np.float32)
d2 = np.abs(np_full - cpp_full)
rm = np.max(np.abs(cpp_full))
fl2 = np.sqrt(np.mean(d2**2)); frefl2 = np.sqrt(np.mean(cpp_full**2))
print("--- FULL GRID (incl. PML) numpy vs cpp [compare to Swift test] ---")
print(f"maxErr/refMax={d2.max()/rm:.4e}  l2/refL2={fl2/frefl2:.4e}")

#!/usr/bin/env python3
"""Performance benchmark for k-wave-python. Mirrors Sources/KWaveBench/main.swift —
scenario names, grids, step counts, and physics must stay in sync so compare.py can
join the results.

Run from the k-wave-python checkout's environment, e.g.:
    cd ../k-wave-python && uv run python ../k-wave-swift/Scripts/bench/bench_python.py --out bench.json

Backend "python" (NumPy, float32) is the default comparison target. Pass --backend cpp
to benchmark the C++ OMP binary instead (downloads on first use; sensor.record limits apply).
"""

import argparse
import json
import math
import platform
import time

import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.angular_spectrum import angular_spectrum

# The beartype hint on angular_spectrum says z_pos: float, but the implementation handles
# vectors (np.atleast_1d). Unwrap so we can project all planes in one call like the Swift side.
angular_spectrum = getattr(angular_spectrum, "__wrapped__", angular_spectrum)
from kwave.utils.mapgen import make_ball, make_disc
from kwave.utils.signals import tone_burst

DX = 1e-4
C0 = 1500.0
RHO0 = 1000.0


def time_it(repeats, warmup, body):
    for _ in range(warmup):
        body()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        body()
        out.append(time.perf_counter() - t0)
    return out


def step_map(shape, base):
    """Half-plane step map matching KWaveBench.stepMap."""
    m = np.full(shape, base, dtype=np.float32)
    m[shape[0] // 2 :] = base * 1.2
    return m


def make_grid(n_per_dim, nt):
    kgrid = kWaveGrid([n for n in n_per_dim], [DX] * len(n_per_dim))
    kgrid.makeTime(C0, cfl=0.3)
    kgrid.setTime(nt, float(kgrid.dt))
    return kgrid


def run_solver(kgrid, medium, source, sensor, pml, args):
    return kspaceFirstOrder(
        kgrid, medium, source, sensor,
        pml_size=pml, pml_inside=True, smooth_p0=getattr(source, "p0", None) is not None,
        backend=args.backend, device="cpu", dtype=np.float32, quiet=True,
    )


def bench_2d(n, nt, args, hetero=False, absorbing=False):
    kgrid = make_grid([n, n], nt)
    if hetero:
        medium = kWaveMedium(sound_speed=step_map((n, n), C0), density=step_map((n, n), RHO0))
    elif absorbing:
        medium = kWaveMedium(sound_speed=C0, density=RHO0, alpha_coeff=0.75, alpha_power=1.5)
    else:
        medium = kWaveMedium(sound_speed=C0, density=RHO0)

    source = kSource()
    source.p0 = make_disc(Vector([n, n]), Vector([n // 2, n // 2]), n / 16).astype(np.float32)
    sensor = kSensor()
    sensor.mask = np.ones((n, n), dtype=bool)
    sensor.record = ["p_final"]

    seconds = time_it(args.repeats, args.warmup,
                      lambda: run_solver(kgrid, medium, source, sensor, 20, args))
    name = f"kspace2d_{n}" + ("_hetero" if hetero else "") + ("_absorbing" if absorbing else "")
    return dict(name=name, seconds=seconds, meta={"grid": f"{n}x{n}", "steps": str(nt)})


def bench_2d_source(n, nt, args):
    kgrid = make_grid([n, n], nt)
    medium = kWaveMedium(sound_speed=C0, density=RHO0)

    source = kSource()
    p_mask = np.zeros((n, n), dtype=bool)
    p_mask[20, :] = True
    source.p_mask = p_mask
    source.p = tone_burst(1.0 / float(kgrid.dt), 1e6, 5).squeeze().astype(np.float32).reshape(1, -1)

    sensor = kSensor()
    mask = np.zeros((n, n), dtype=bool)
    mask[n - 20, :] = True
    sensor.mask = mask
    sensor.record = ["p"]

    seconds = time_it(args.repeats, args.warmup,
                      lambda: run_solver(kgrid, medium, source, sensor, 20, args))
    return dict(name=f"kspace2d_{n}_source", seconds=seconds,
                meta={"grid": f"{n}x{n}", "steps": str(nt)})


def bench_3d(n, nt, args):
    kgrid = make_grid([n, n, n], nt)
    medium = kWaveMedium(sound_speed=C0, density=RHO0)
    source = kSource()
    source.p0 = make_ball(Vector([n, n, n]), Vector([n // 2] * 3), n // 8).astype(np.float32)
    sensor = kSensor()
    sensor.mask = np.ones((n, n, n), dtype=bool)
    sensor.record = ["p_final"]

    seconds = time_it(args.repeats, args.warmup,
                      lambda: run_solver(kgrid, medium, source, sensor, 10, args))
    return dict(name=f"kspace3d_{n}", seconds=seconds,
                meta={"grid": f"{n}x{n}x{n}", "steps": str(nt)})


def bench_angular_spectrum(n, nt, nz, args):
    dt = 2e-8
    disc = np.asarray(make_disc(Vector([n, n]), Vector([n // 2, n // 2]), n / 8), dtype=np.float32)
    burst = tone_burst(1.0 / dt, 1e6, 5).squeeze().astype(np.float32)
    burst = np.pad(burst, (0, max(0, nt - len(burst))))[:nt]
    plane = disc[:, :, None] * burst[None, None, :]
    z_pos = np.arange(1, nz + 1) * DX

    # data_cast="single" is broken upstream (exec references an undefined `single`),
    # so this runs in float64 — k-wave-python's only working precision here.
    seconds = time_it(args.repeats, args.warmup,
                      lambda: angular_spectrum(plane, DX, dt, z_pos, C0))
    return dict(name=f"angular_spectrum_{n}", seconds=seconds,
                meta={"grid": f"{n}x{n}x{nt}", "steps": f"{nz} planes"})


# --- AFP: k-wave-python has no acoustic_field_propagator; this NumPy implementation is
# --- lifted from Scripts/parity/generate_reference_afp.py (mirrors MATLAB k-Wave).

def _largest_prime_factor(n):
    best, m, d = 1, n, 2
    while d * d <= m:
        while m % d == 0:
            best = max(best, d)
            m //= d
        d += 1
    return max(best, m if m > 1 else 1)


def _optimal_grid_dim(n, search_range):
    facs = [_largest_prime_factor(n + i) for i in range(search_range + 1)]
    return n + int(np.argmin(facs))


def afp_numpy(amp_in, phase_in, dx, f0, c0, use_ramp=True,
              time_exp_factor=1.5, grid_exp_factor=1.1, grid_search_range=50):
    sz = amp_in.shape
    w0 = 2 * np.pi * f0
    t = time_exp_factor * dx * np.sqrt(sum(s**2 for s in sz)) / c0
    expansion = int(np.ceil(grid_exp_factor * t * c0 / dx))
    sz_ex = tuple(_optimal_grid_dim(s + expansion, grid_search_range) for s in sz)

    src = np.zeros(sz_ex, dtype=np.complex64)
    src[tuple(slice(0, s) for s in sz)] = amp_in * np.exp(1j * phase_in)

    ks = [(2 * np.pi * np.fft.fftfreq(n, d=dx)).astype(np.float32) for n in sz_ex]
    k2 = np.zeros(sz_ex, dtype=np.float32)
    for axis, kv in enumerate(ks):
        shape = [1] * len(sz_ex)
        shape[axis] = len(kv)
        k2 = k2 + kv.reshape(shape) ** 2
    k = np.sqrt(k2)

    ck = c0 * k
    with np.errstate(divide="ignore", invalid="ignore"):
        if use_ramp:
            num = (w0**2 * np.exp(1j * w0 * t) - ck**2 * np.cos(w0 * t) * np.exp(1j * ck * t)
                   - 1j * w0 * ck * np.sin(w0 * t) * np.exp(1j * ck * t))
            den = w0**2 - ck**2
        else:
            num = 1j * w0 * (np.exp(1j * w0 * t) - np.exp(1j * ck * t))
            den = w0 - ck
        prop = num / den
    sing = ~np.isfinite(prop)
    prop[sing] = 0.5 * (1 + 1j * w0 * t) * np.exp(1j * w0 * t) if use_ramp \
        else 1j * w0 * t * np.exp(1j * w0 * t)

    out = np.fft.ifftn(np.fft.fftn(src) * prop)
    return out[tuple(slice(0, s) for s in sz)]


def bench_afp(n, args):
    amp = make_ball(Vector([n, n, n]), Vector([n // 2] * 3), n // 8).astype(np.float32)
    seconds = time_it(args.repeats, args.warmup,
                      lambda: afp_numpy(amp, 0.0, DX, 1e6, C0))
    return dict(name=f"afp_{n}", seconds=seconds,
                meta={"grid": f"{n}x{n}x{n}", "steps": "1"})


SCENARIOS = [
    ("kspace2d_64", lambda a: bench_2d(64, 256, a)),
    ("kspace2d_128", lambda a: bench_2d(128, 256, a)),
    ("kspace2d_256", lambda a: bench_2d(256, 256, a)),
    ("kspace2d_512", lambda a: bench_2d(512, 256, a)),
    ("kspace2d_1024", lambda a: bench_2d(1024, 256, a)),
    ("kspace2d_256_hetero", lambda a: bench_2d(256, 256, a, hetero=True)),
    ("kspace2d_256_absorbing", lambda a: bench_2d(256, 256, a, absorbing=True)),
    ("kspace2d_256_source", lambda a: bench_2d_source(256, 256, a)),
    ("kspace3d_32", lambda a: bench_3d(32, 128, a)),
    ("kspace3d_64", lambda a: bench_3d(64, 128, a)),
    ("kspace3d_128", lambda a: bench_3d(128, 128, a)),
    ("angular_spectrum_128", lambda a: bench_angular_spectrum(128, 64, 32, a)),
    ("afp_64", lambda a: bench_afp(64, a)),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="python", choices=["python", "cpp"])
    args = ap.parse_args()

    results = []
    for name, fn in SCENARIOS:
        if args.filter and args.filter not in name:
            continue
        r = fn(args)
        secs = sorted(r["seconds"])
        r["best"] = secs[0]
        r["median"] = secs[len(secs) // 2]
        results.append(r)
        times = ", ".join(f"{s:.4f}" for s in r["seconds"])
        print(f"{r['name']}: best {r['best']:.4f}s  median {r['median']:.4f}s  [{times}]")

    if args.out:
        payload = dict(impl="python", backend=args.backend,
                       machine=platform.machine(), repeats=args.repeats, results=results)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print("wrote", args.out)


if __name__ == "__main__":
    main()

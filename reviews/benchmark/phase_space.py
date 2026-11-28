"""Distances between two particle distributions, for what rms sizes and emittances do not see: shape and halo.  Numpy only.

A worker saves its last particle snapshot with ``save_snapshot``; the analysis compares it with the reference's.  Each plane,
(x, x'), (y, y') and (z, delta), of each sample is whitened with its own mean and covariance, so size, emittance and orientation drop
out (the rms gates carry those) and what is left is the shape.

``shape``  sliced Wasserstein-1 distance of the two whitened planes, in standard deviations: the mean over directions of the mean
           distance between the sorted projections.  0 for equal shapes.
``halo``   relative difference of the 99.9th percentile of the whitened radius.
"""
import json
import pathlib
import sys

import numpy as np

KEYS = ("x", "y", "z", "bgx", "bgy", "bgz")


def save_snapshot(x, y, z, bgx, bgy, bgz):
    """Write ``snapshot.npz`` (lab frame, one time: m and beta*gamma) beside the job's result.  Nothing happens outside a job."""
    if "--job" not in sys.argv:
        return
    out = pathlib.Path(json.loads(pathlib.Path(sys.argv[sys.argv.index("--job") + 1]).read_text())["out_dir"])
    np.savez(out / "snapshot.npz", **{k: np.asarray(v, dtype=np.float64) for k, v in zip(KEYS, (x, y, z, bgx, bgy, bgz))})


def planes(snapshot, station_m: float) -> dict:
    """The three phase planes, each (N, 2), after a straight drift that brings the mean z to ``station_m``: codes stop at slightly
    different times, and over those centimetres the momenta hardly change."""
    x, y, z, bgx, bgy, bgz = (np.asarray(snapshot[k], dtype=np.float64) for k in KEYS)
    gamma = np.sqrt(1 + bgx**2 + bgy**2 + bgz**2)
    ct = (z.mean() - station_m) / (bgz / gamma).mean()
    x, y, z = (q - bg / gamma * ct for q, bg in ((x, bgx), (y, bgy), (z, bgz)))
    p = np.sqrt(bgx**2 + bgy**2 + bgz**2)
    return dict(x=np.column_stack([x, bgx / bgz]), y=np.column_stack([y, bgy / bgz]), z=np.column_stack([z, p / p.mean() - 1]))


def whiten(a):
    """Zero mean, unit covariance, by the symmetric inverse square root (it does not favour either axis)."""
    a = np.asarray(a, dtype=np.float64) - np.mean(a, axis=0)
    a = a / a.std(axis=0)   # bring metres and radians to one scale before the eigendecomposition
    w, v = np.linalg.eigh(np.cov(a, rowvar=False, bias=True))
    return a @ (v / np.sqrt(w)) @ v.T


def sliced_wasserstein(a, b, directions: int = 64) -> float:
    """Mean over ``directions`` of the 1-D Wasserstein-1 distance of the projections of two (N, 2) samples (N may differ)."""
    theta = np.arange(directions) * np.pi / directions
    u = np.stack([np.cos(theta), np.sin(theta)])
    pa, pb = np.sort(a @ u, axis=0), np.sort(b @ u, axis=0)
    if len(pa) != len(pb):
        n = max(len(pa), len(pb))
        q = (np.arange(n) + 0.5) / n
        pa, pb = (p[np.minimum((q * len(p)).astype(int), len(p) - 1)] for p in (pa, pb))
    return float(np.abs(pa - pb).mean())


def compare(snapshot, reference, station_m: float) -> dict:
    """``{shape, halo}`` of the transverse planes (the worse of x and y) and ``{shape_z, halo_z}`` of the longitudinal one."""
    a, b = planes(snapshot, station_m), planes(reference, station_m)
    found = {}
    for k in a:
        wa, wb = whiten(a[k]), whiten(b[k])
        tail = [np.quantile(np.hypot(*w.T), 0.999) for w in (wa, wb)]
        found[k] = (sliced_wasserstein(wa, wb), float(abs(tail[0] / tail[1] - 1)))
    return dict(shape=max(found["x"][0], found["y"][0]), halo=max(found["x"][1], found["y"][1]),
                shape_z=found["z"][0], halo_z=found["z"][1])

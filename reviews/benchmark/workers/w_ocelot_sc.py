"""Ocelot free-space PIC drift, using the same frozen bunch and station observables.

Ocelot uses MAD coordinates: transverse momenta/P0 and (E-E0)/(P0*c).
The fixed-time input is projected ballistically to s=0; output bunch lengths
use z=-beta0*tau. No cathode image, radiation or external fields are applied.
"""
import importlib.metadata
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from common import lab_stats, launches, run_worker

CONFIGS = [dict(name=f"cpu-f64-g{g}s{n}", device="cpu", dtype="float64", grid=g, nslice=n)
           for g, n in [(16, 1), (32, 2), (64, 4)]]


def setup():
    import ocelot  # noqa: F401 -- import is timed separately


def build(lat, case, cfg):
    return dict(case=case, lab=launches(case)), None


def track(model, init, turns, cfg):
    from ocelot import ParticleArray, SpaceCharge, Drift, SecondTM
    from ocelot.cpbd.coord_transform import xp_2_xxstg_mad, xxstg_2_xp_mad
    case, lab = model["case"], model["lab"]
    mass = case["mass_eV"]
    gamma = case["energy_eV"] / mass
    # Use Ocelot's own MAD conversion in both directions (momenta in eV/c).
    xp = lab.T.copy()
    xp[3:] *= mass
    p = ParticleArray(n=len(lab))
    p.E = case["energy_eV"] * 1e-9
    p.q_array[:] = abs(case["charge_C"]) / len(lab)
    xp_2_xxstg_mad(xp, p.rparticles, gamma)
    restored = xxstg_2_xp_mad(p.rparticles, np.zeros_like(xp), gamma)
    assert np.allclose(restored, xp, rtol=2e-9, atol=1e-8), "MAD frame round trip"
    sc = SpaceCharge(nmesh_xyz=[cfg["grid"]] * 3, random_mesh=False)
    rows = []

    def record(z):
        state = xxstg_2_xp_mad(p.rparticles, np.zeros_like(xp), gamma)
        d = lab_stats(*state[:3], *(state[3:] / mass), mass_eV=mass)
        rows.append([z] + [d[k] for k in ("sigma_x_m", "sigma_y_m", "sigma_z_m", "eps_nx_m", "eps_ny_m", "ekin_eV", "sigma_E_eV")])

    record(0.0)
    for a, b in zip(case["stations_m"][:-1], case["stations_m"][1:]):
        n = int(np.ceil((b - a) / (0.05 / cfg["nslice"])))
        ds = (b - a) / n
        half = Drift(l=ds / 2, tm=SecondTM)
        for _ in range(n):
            for tm in half.tms:
                tm.apply(p)
            if case["charge_C"] != 0:
                sc.apply(p, ds)
            for tm in half.tms:
                tm.apply(p)
        record(b)
    return np.asarray(rows)


def versions():
    for dist in ("ocelot-collab", "ocelot-desy"):  # the same code under two distribution names (PyPI, DESY)
        try:
            return dict(ocelot=importlib.metadata.version(dist), distribution=dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(ocelot=None)


if __name__ == "__main__":
    run_worker("ocelot", CONFIGS, build, track, versions=versions, setup=setup)

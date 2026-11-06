"""HELIX's fixed-energy linear FODO path, at 3 MeV proton kinetic energy.

Restrict comparison to transverse transport: HELIX mm/mrad slopes are converted
explicitly, and the tiny basis launches isolate the linear limit. This case
does not validate either code's space charge, RF or nonlinear tracking.
"""
import importlib.metadata
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from common import canon_to_slopes, slopes_to_canon, run_worker

CONFIGS = [dict(name="cpu-f64-linear", device="cpu", dtype="float64")]


def setup():
    import linac_gen.tracking.autograd_api  # noqa: F401 -- import is timed separately


def build(lat, case, cfg):
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import Particle
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.tracking.autograd_api import DifferentiableLattice
    ref = ReferenceParticle(Particle(case["mass_eV"] * 1e-6, 1, "proton"),
                            w_kin=(case["energy_eV"] - case["mass_eV"]) * 1e-6, frequency=352.21)
    line = Lattice()
    for e in lat["elements"]:
        if e["kind"] == "drift":
            el = Drift(e["name"], length=e["length"] * 1e3)
        elif e["kind"] == "quad":
            el = Quadrupole(e["name"], length=e["length"] * 1e3, gradient=e["k1"] * ref.brho)
        else:
            raise ValueError(e["kind"])
        line.add(el)
    return DifferentiableLattice(line, ref), None


def track(model, init, turns, cfg):
    p = init.copy()
    p[:, 1], p[:, 3] = canon_to_slopes(p[:, 1], p[:, 3], p[:, 5])
    p[:, :4] *= 1e3
    rec = []
    for _ in range(turns):
        bs = p[:, :4].copy() * 1e-3
        bs[:, 1], bs[:, 3] = slopes_to_canon(bs[:, 1], bs[:, 3], init[:, 5])
        rec.append(bs)
        p = model.track(p).detach().numpy()
    return np.stack(rec, axis=1)


def versions():
    import torch
    return dict(helix=importlib.metadata.version("linac_gen"), torch=torch.__version__)


if __name__ == "__main__":
    run_worker("helix", CONFIGS, build, track, versions=versions, setup=setup)

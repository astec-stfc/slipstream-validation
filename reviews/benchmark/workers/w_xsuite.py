"""Xsuite (xtrack) adapter, CPU and CUDA (cupy) contexts; both are float64.
R1 explicitly selects exact quadrupole splitting and bend-kick-bend bodies, but leaves drifts/sextupoles adaptive.
The separate *-exact controls explicitly select exact Drift/Sextupole models; the original R1 settings are retained.
Bend edges are off to match e1 = e2 = 0."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from common import p0c_eV, run_worker  # noqa: E402

_RUNGS = [("teapot", 2, 4), ("yoshida4", 2, 4), ("yoshida4", 4, 8), ("yoshida4", 8, 16), ("yoshida4", 16, 32)]
CONFIGS = [dict(name=f"cuda-f64-{i[:3]}k{k}b{b}", device="cuda", dtype="float64", integrator=i, n_kicks=k, n_kicks_bend=b)
           for i, k, b in _RUNGS[1:4]]
CONFIGS += [dict(name=f"cpu-f64-{i[:3]}k{k}b{b}", device="cpu", dtype="float64", integrator=i, n_kicks=k, n_kicks_bend=b)
            for i, k, b in _RUNGS]
# Explicit model correction is a separate follow-up, preserving every R1 configuration.
CONFIGS += [dict(c, name=c["name"] + "-exact", explicit_exact=True) for c in CONFIGS
            if c["name"] in ("cuda-f64-yosk8b16", "cpu-f64-yosk8b16", "cpu-f64-yosk16b32")]


def setup():
    import xfields  # noqa: F401
    import xobjects  # noqa: F401
    import xtrack  # noqa: F401


def build(lat, case, cfg):
    import xobjects as xo
    import xtrack as xt

    integrator, nk, nb = cfg["integrator"], cfg["n_kicks"], cfg["n_kicks_bend"]
    els, names = [], []
    for i, e in enumerate(lat["elements"]):
        k, L = e["kind"], e["length"]
        if k == "drift":
            el = xt.Drift(length=L, model="exact" if cfg.get("explicit_exact") else "adaptive")
        elif k == "quad":
            el = xt.Quadrupole(length=L, k1=e["k1"], num_multipole_kicks=nk, integrator=integrator, model="drift-kick-drift-exact")
        elif k == "sext":
            el = xt.Sextupole(length=L, k2=e["k2"], num_multipole_kicks=nk, integrator=integrator,
                              model="drift-kick-drift-exact" if cfg.get("explicit_exact") else "adaptive")
        elif k == "bend":
            el = xt.Bend(length=L, angle=e["angle"], k0=e["angle"] / L, edge_entry_active=False, edge_exit_active=False,
                         num_multipole_kicks=nb, integrator=integrator, model="bend-kick-bend")
        else:
            raise ValueError(k)
        els.append(el)
        names.append(f"{e['name']}_{i}")
    ctx = xo.ContextCupy() if cfg["device"] == "cuda" else xo.ContextCpu(omp_num_threads=1)
    line = xt.Line(elements=els, element_names=names)
    p0c = p0c_eV(case["energy_eV"], case["mass_eV"])
    line.particle_ref = xt.Particles(p0c=p0c, mass0=case["mass_eV"])
    line.build_tracker(_context=ctx)
    inv = dict(n=len(els), length=sum(getattr(x, "length", 0.0) for x in els),
               angle=sum(x.angle for x in els if isinstance(x, xt.Bend)),
               k1L_abs=sum(abs(x.k1) * x.length for x in els if isinstance(x, xt.Quadrupole)),
               k2L_abs=sum(abs(x.k2) * x.length for x in els if isinstance(x, xt.Sextupole)))
    return dict(line=line, ctx=ctx, p0c=p0c, mass=case["mass_eV"]), inv


def track(model, init, turns, cfg):
    line = model["line"]
    p = line.build_particles(x=init[:, 0], px=init[:, 1], y=init[:, 2], py=init[:, 3], delta=init[:, 5], zeta=init[:, 4])
    line.track(p, num_turns=turns, turn_by_turn_monitor=True)  # monitor holds the state at the START of turns 0..T-1
    m = line.record_last_track
    to_np = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)  # noqa: E731  # the monitor may already be host memory
    out = np.stack([to_np(m.x), to_np(m.px), to_np(m.y), to_np(m.py)], axis=-1).astype(float)
    return out


def fd_update(model, lat, case, cfg):
    """Set the strengths of the built line in place (no rebuild, so no C recompilation) for finite differences."""
    el = model["line"].element_dict
    for i, e in enumerate(lat["elements"]):
        if e["kind"] == "quad":
            el[f"{e['name']}_{i}"].k1 = e["k1"]
        elif e["kind"] == "sext":
            el[f"{e['name']}_{i}"].k2 = e["k2"]


def sync(dev):
    if dev == "cuda":
        import cupy
        cupy.cuda.Device().synchronize()


def peak_mem(dev):
    if dev != "cuda":
        return {}
    import cupy
    return dict(cuda_pool_total_bytes=int(cupy.get_default_memory_pool().total_bytes()))  # ponytail: pool size, not true peak


def versions():
    import xfields
    import xobjects
    import xtrack
    return dict(xtrack=xtrack.__version__, xfields=xfields.__version__, xobjects=xobjects.__version__)


if __name__ == "__main__":
    run_worker("xsuite", CONFIGS, build, track, sync, peak_mem, versions, setup, fd_update=fd_update)

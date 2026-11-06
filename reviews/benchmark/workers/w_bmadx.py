"""Bmad-X adapter (torch path: TorchLattice of TorchDrift/TorchQuadrupole/TorchSextupole/TorchSBend).
Bmad-X's coordinates already are BSF: px = Px/P0 (canonical), pz = delta, z = -beta0 c (t - t0) (not used here).
Element models, read from the source: the drift and the sector-bend body are exact; the quadrupole body is an exact linear map per
step (NUM_STEPS only matters off-momentum); the sextupole is ONE explicit-Euler kick-drift per step (first order, not symplectic),
so the sextupole step count is the convergence knob.  Hard-edge fringes are off ("none") to match the e1 = e2 = 0 sector bends."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import p0c_eV, run_worker  # noqa: E402

_RUNGS = [(1, 4), (1, 16), (1, 64), (1, 256)]  # (steps per quadrupole, steps per sextupole)
CONFIGS = [dict(name=f"cuda-f{t[-2:]}-q{q}s{s}", device="cuda", dtype=t, n_quad=q, n_sext=s)
           for t in ("float32", "float64") for q, s in _RUNGS[:3]]
CONFIGS += [dict(name=f"cpu-f64-q{q}s{s}", device="cpu", dtype="float64", n_quad=q, n_sext=s) for q, s in _RUNGS]


def setup():
    import bmadx  # noqa: F401
    import torch  # noqa: F401


def build(lat, case, cfg):
    import torch
    from bmadx.bmad_torch.track_torch import (Beam, TorchDrift, TorchLattice, TorchQuadrupole, TorchSBend,
                                              TorchSextupole)

    dev, dt = torch.device(cfg["device"]), getattr(torch, cfg["dtype"])
    if dev.type == "cpu":
        torch.set_num_threads(1)
    p0c = p0c_eV(case["energy_eV"], case["mass_eV"])
    T = lambda v: torch.tensor(float(v), dtype=dt, device=dev)  # noqa: E731
    els = []
    for e in lat["elements"]:
        k, L = e["kind"], e["length"]
        if k == "drift":
            els.append(TorchDrift(T(L)))
        elif k == "quad":
            els.append(TorchQuadrupole(T(L), T(e["k1"]), NUM_STEPS=cfg["n_quad"]))
        elif k == "sext":
            els.append(TorchSextupole(T(L), T(e["k2"]), NUM_STEPS=cfg["n_sext"]))
        elif k == "bend":
            els.append(TorchSBend(T(L), T(p0c), T(e["angle"] / L), FRINGE_AT="no_end", FRINGE_TYPE="none"))
        else:
            raise ValueError(k)
    lattice = TorchLattice(els).to(dev)
    inv = dict(n=len(els), length=sum(float(x.L) for x in els),
               angle=sum(float(x.G * x.L) for x in els if isinstance(x, TorchSBend)),
               k1L_abs=sum(abs(float(x.K1)) * float(x.L) for x in els if isinstance(x, TorchQuadrupole)),
               k2L_abs=sum(abs(float(x.K2)) * float(x.L) for x in els if isinstance(x, TorchSextupole)))
    return dict(lattice=lattice, dev=dev, dt=dt, p0c=p0c, mc2=case["mass_eV"], Beam=Beam), inv


def track(model, init, turns, cfg):
    import torch

    dev, dt = model["dev"], model["dt"]
    data = torch.as_tensor(init[:, [0, 1, 2, 3, 4, 5]], dtype=dt, device=dev).clone()  # (x, px, y, py, z, pz=delta)
    beam = model["Beam"](data, torch.tensor(model["p0c"], dtype=dt, device=dev), mc2=torch.tensor(model["mc2"], dtype=dt, device=dev))
    rec = []
    with torch.no_grad():
        p = beam
        for _ in range(turns):
            rec.append(torch.stack([p.x, p.px, p.y, p.py], -1).clone())
            p = model["lattice"](p)
    return torch.stack(rec, 1).double().cpu().numpy()


def ad_build(lat, case, cfg, knobs):
    from common import nominal_strengths
    model, _ = build(lat, case, cfg)
    model.update(lat=lat, case=case, cfg=cfg, theta0=nominal_strengths(lat, knobs))
    return model


def ad_jacobian(m, init, turns, cfg, knobs):
    """Reverse-mode Jacobian; knob elements are rebuilt each call as namedtuples (Bmad-X's functional ``track_element``) so that
    the strength tensor stays in the autograd graph; all other elements are the prebuilt torch modules."""
    import torch
    from bmadx import Quadrupole, Sextupole, track_element

    dev, dt = m["dev"], m["dt"]
    fam = {kn["family"]: j for j, kn in enumerate(knobs)}
    mods = m["lattice"].elements
    lat_els = m["lat"]["elements"]
    T = lambda v: torch.tensor(float(v), dtype=dt, device=dev)  # noqa: E731

    def f(theta):
        data = torch.as_tensor(init, dtype=dt, device=dev).clone()
        p = m["Beam"](data, torch.tensor(m["p0c"], dtype=dt, device=dev), mc2=torch.tensor(m["mc2"], dtype=dt, device=dev))
        for _ in range(turns - 1):
            for e, mod in zip(lat_els, mods):
                if e["name"] in fam and e["kind"] == "quad":
                    p = track_element(p, Quadrupole(L=T(e["length"]), K1=theta[fam[e["name"]]], NUM_STEPS=cfg["n_quad"]))
                elif e["name"] in fam and e["kind"] == "sext":
                    p = track_element(p, Sextupole(L=T(e["length"]), K2=theta[fam[e["name"]]], NUM_STEPS=cfg["n_sext"]))
                else:
                    p = mod(p)
        return torch.stack([p.x[0], p.px[0], p.y[0], p.py[0]])

    theta0 = torch.as_tensor(m["theta0"], dtype=dt, device=dev)
    J = torch.autograd.functional.jacobian(f, theta0)
    return f(theta0).detach().double().cpu().numpy(), J.detach().double().cpu().numpy()


def sync(dev):
    if dev == "cuda":
        import torch
        torch.cuda.synchronize()


def peak_mem(dev):
    if dev != "cuda":
        return {}
    import torch
    return dict(cuda_peak_allocated=int(torch.cuda.max_memory_allocated()), cuda_peak_reserved=int(torch.cuda.max_memory_reserved()))


def versions():
    import bmadx
    import torch
    return dict(bmadx_file=bmadx.__file__, torch=torch.__version__, cuda=torch.version.cuda)


if __name__ == "__main__":
    run_worker("bmadx", CONFIGS, build, track, sync, peak_mem, versions, setup, ad=(ad_build, ad_jacobian))

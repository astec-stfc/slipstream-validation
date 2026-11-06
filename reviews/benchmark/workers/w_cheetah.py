"""Cheetah adapter.  Cheetah's sextupole only offers 'linear' and 'second_order', so the nonlinear rungs use the
second-order sextupole with drift-kick-drift quadrupoles/dipoles (the same choice as the version-1 report).
Cheetah has no octupole path needed here (the lattice stores the K3 = 0 octupole as a drift)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from common import cheetah_p_to_delta, delta_to_cheetah_p, run_worker  # noqa: E402,F401

_RUNGS = [("linear", 1), ("second_order", 1), ("drift_kick_drift", 4), ("drift_kick_drift", 16), ("drift_kick_drift", 64)]
CONFIGS = [dict(name=f"cuda-f32-{m[:3]}{n}", device="cuda", dtype="float32", method=m, num_steps=n)
           for m, n in _RUNGS[:4]]
CONFIGS += [dict(name=f"cpu-f64-{m[:3]}{n}", device="cpu", dtype="float64", method=m, num_steps=n) for m, n in _RUNGS]


def setup():
    import cheetah  # noqa: F401
    import torch  # noqa: F401


def build(lat, case, cfg):
    import cheetah
    import torch

    kw = dict(device=cfg["device"], dtype=getattr(torch, cfg["dtype"]))
    if cfg["device"] == "cpu":
        torch.set_num_threads(1)
    T = lambda v: torch.tensor(float(v), **kw)  # noqa: E731
    m, ns, els = cfg["method"], cfg["num_steps"], []
    for e in lat["elements"]:
        k, L = e["kind"], e["length"]
        if k == "drift":
            els.append(cheetah.Drift(T(L), tracking_method=m, **kw))
        elif k == "quad":
            els.append(cheetah.Quadrupole(T(L), k1=T(e["k1"]), num_steps=ns, tracking_method=m, **kw))
        elif k == "sext":
            els.append(cheetah.Sextupole(T(L), k2=T(e["k2"]), tracking_method="linear" if m == "linear" else "second_order", **kw))
        elif k == "bend":
            els.append(cheetah.Dipole(T(L), angle=T(e["angle"]), dipole_e1=T(0.0), dipole_e2=T(0.0), k1=T(0.0),
                                      tracking_method=m, **kw))
        else:
            raise ValueError(k)
    seg = cheetah.Segment(els)
    inv = dict(n=len(els), length=sum(float(x.length) for x in els),
               angle=sum(float(x.angle) for x in els if isinstance(x, cheetah.Dipole)),
               k1L_abs=sum(abs(float(x.k1)) * float(x.length) for x in els if isinstance(x, cheetah.Quadrupole)),
               k2L_abs=sum(abs(float(x.k2)) * float(x.length) for x in els if isinstance(x, cheetah.Sextupole)))
    return dict(seg=seg, kw=kw, E0=case["energy_eV"]), inv


def track(model, init, turns, cfg):
    import cheetah
    import torch

    kw, E0 = model["kw"], model["E0"]
    p = np.zeros((init.shape[0], 7))
    p[:, :4] = init[:, :4]
    p[:, 4] = init[:, 4]
    p[:, 5] = delta_to_cheetah_p(init[:, 5], E0)
    p[:, 6] = 1.0
    beam = cheetah.ParticleBeam(particles=torch.as_tensor(p, **kw), energy=torch.tensor(E0, **kw), **kw)
    rec = []
    with torch.no_grad():
        for _ in range(turns):
            rec.append(beam.particles[:, :4].clone())
            beam = model["seg"](beam)
    return torch.stack(rec, 1).double().cpu().numpy()


def _segment(lat, cfg, theta, knobs):
    """Segment whose knob-family elements all share one tensor (so autograd sums over occurrences)."""
    import cheetah
    import torch

    kw = dict(device=cfg["device"], dtype=getattr(torch, cfg["dtype"]))
    T = lambda v: torch.tensor(float(v), **kw)  # noqa: E731
    fam = {kn["family"]: theta[j] for j, kn in enumerate(knobs)}
    m, ns, els = cfg["method"], cfg["num_steps"], []
    for e in lat["elements"]:
        k, L = e["kind"], e["length"]
        if k == "drift":
            els.append(cheetah.Drift(T(L), tracking_method=m, **kw))
        elif k == "quad":
            els.append(cheetah.Quadrupole(T(L), k1=fam.get(e["name"], T(e["k1"])), num_steps=ns, tracking_method=m, **kw))
        elif k == "sext":
            els.append(cheetah.Sextupole(T(L), k2=fam.get(e["name"], T(e["k2"])), tracking_method="linear" if m == "linear" else "second_order", **kw))
        else:
            els.append(cheetah.Dipole(T(L), angle=T(e["angle"]), dipole_e1=T(0.0), dipole_e2=T(0.0), k1=T(0.0), tracking_method=m, **kw))
    return cheetah.Segment(els), kw


def ad_build(lat, case, cfg, knobs):
    from common import nominal_strengths
    return dict(lat=lat, case=case, theta0=nominal_strengths(lat, knobs))


def ad_jacobian(m, init, turns, cfg, knobs):
    import cheetah
    import torch

    kw0 = dict(device=cfg["device"], dtype=getattr(torch, cfg["dtype"]))
    p0 = np.zeros((1, 7))
    p0[:, :4] = init[:, :4]
    p0[:, 5] = delta_to_cheetah_p(init[:, 5], m["case"]["energy_eV"])
    p0[:, 6] = 1.0

    def f(theta):
        seg, kw = _segment(m["lat"], cfg, theta, knobs)
        beam = cheetah.ParticleBeam(particles=torch.as_tensor(p0, **kw), energy=torch.tensor(m["case"]["energy_eV"], **kw), **kw)
        for _ in range(turns - 1):
            beam = seg(beam)
        return beam.particles[0, :4]

    theta0 = torch.as_tensor(m["theta0"], **kw0)
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
    import cheetah
    import torch
    return dict(cheetah=cheetah.__version__, torch=torch.__version__, cuda=torch.version.cuda)


if __name__ == "__main__":
    run_worker("cheetah", CONFIGS, build, track, sync, peak_mem, versions, setup, ad=(ad_build, ad_jacobian))

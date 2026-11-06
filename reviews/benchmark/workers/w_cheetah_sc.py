"""Cheetah adapter for ``sc_drift``: alternating ``Drift`` and ``SpaceChargeKick`` (integrated-Green-function Poisson solve on a 3-D grid, momentum
kick only) per slice.  Station statistics are computed here from the particle array: sigma_z = beta_ref * sigma_tau, normalised emittance
sqrt(<x^2><px^2> - <x px>^2) * (beta gamma)_ref with px = Px/P0, energy from p = (E - E0)/(p0 c).  The macroparticles start at fixed s
(tau = -z/beta_ref, x_s = x - x' z, ignored at this size: x' z ~ 1e-8 m)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from common import run_worker  # noqa: E402

# (grid points per axis, grid extent in sigma, slices per 0.05 m)
_RUNGS = [(16, 4.0, 1), (32, 4.0, 2), (64, 4.0, 4)]
CONFIGS = [dict(name=f"cuda-f32-g{g}s{s}", device="cuda", dtype="float32", grid=g, extent=e, nslice=s) for g, e, s in _RUNGS[:2]]
CONFIGS += [dict(name=f"cpu-f64-g{g}s{s}", device="cpu", dtype="float64", grid=g, extent=e, nslice=s) for g, e, s in _RUNGS]


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
    n_steps = int(round(case["stations_m"][-1] / 0.05))
    L = 0.05 / cfg["nslice"]
    els = []
    for _ in range(n_steps * cfg["nslice"]):  # half drift, kick, half drift: second-order splitting
        els += [cheetah.Drift(T(L / 2), tracking_method="exact", **kw),
                cheetah.SpaceChargeKick(T(L), grid_shape=(cfg["grid"],) * 3, grid_extent_x=T(cfg["extent"]), grid_extent_y=T(cfg["extent"]),
                                        grid_extent_tau=T(cfg["extent"]), **kw),
                cheetah.Drift(T(L / 2), tracking_method="exact", **kw)]
    return dict(elements=els, kw=kw, per_station=n_steps and cfg["nslice"], case=case, seg=cheetah.Segment(els)), None


def _stats(p, E0, mc2, case):
    p0c = np.sqrt(E0**2 - mc2**2)
    bg0, beta0 = p0c / mc2, p0c / E0
    x, px, y, py, tau, dp = (p[:, i] for i in range(6))
    cov = lambda a, b: np.mean(a * b) - np.mean(a) * np.mean(b)  # noqa: E731
    eps = lambda a, b: np.sqrt(max(cov(a, a) * cov(b, b) - cov(a, b) ** 2, 0.0))  # noqa: E731
    E = E0 + dp * p0c
    return [cov(x, x) ** 0.5, cov(y, y) ** 0.5, beta0 * cov(tau, tau) ** 0.5, eps(x, px) * bg0, eps(y, py) * bg0, E.mean() - mc2, E.std()]


def track(model, init, turns, cfg):
    import cheetah
    import torch

    kw, case = model["kw"], model["case"]
    mc2, E0 = case["mass_eV"], case["energy_eV"]
    p0c = np.sqrt(E0**2 - mc2**2)
    bg0 = p0c / mc2
    beta0 = p0c / E0
    x, y, z, bgx, bgy, bgz = init.T
    E = mc2 * np.sqrt(1 + bgx**2 + bgy**2 + bgz**2)
    parts = np.column_stack([x, bgx / bg0, y, bgy / bg0, -z / beta0, (E - E0) / p0c, np.ones(len(x))])
    n = len(x)
    charges = torch.full((n,), abs(case["charge_C"]) / n, **kw)
    beam = cheetah.ParticleBeam(particles=torch.as_tensor(parts, **kw), energy=torch.tensor(E0, **kw), particle_charges=charges, **kw)
    st = case["stations_m"]
    per = int(round(0.05 / (0.05 / cfg["nslice"])))  # slices per 0.05 m
    out = [[st[0]] + _stats(beam.particles[:, :6].double().cpu().numpy(), E0, mc2, case)]
    els, k = model["elements"], 0
    with torch.no_grad():
        for s in st[1:]:
            target = int(round(s / 0.05)) * per * 3  # elements per slice = 3
            for e in els[k:target]:
                beam = e(beam)
            k = target
            out.append([s] + _stats(beam.particles[:, :6].double().cpu().numpy(), E0, mc2, case))
    return np.array(out)


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
    run_worker("cheetah", CONFIGS, build, track, sync, peak_mem, versions, setup)

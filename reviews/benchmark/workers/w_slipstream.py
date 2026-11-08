"""Slipstream RingMap adapter.  Configs run coarse -> fine; the last is Slipstream's own converged reference."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from common import run_worker  # noqa: E402

# (order, n_kicks per quad/sext, n_kicks_bend); device/dtype are separate rungs so precision is a recorded variable
_RUNGS = [(4, 1, 2), (4, 2, 4), (4, 4, 8), (6, 4, 8), (6, 8, 16), (6, 16, 32)]
CONFIGS = [dict(name=f"cuda-f{t[-2:]}-o{o}k{k}b{b}", device="cuda", dtype=t, order=o, n_kicks=k, n_kicks_bend=b)
           for t in ("float32", "float64") for o, k, b in _RUNGS if (o, k) in [(4, 4), (6, 8), (6, 16)]]
# the converged CPU float64 map is the reference rung, so it comes last
CONFIGS += [dict(name=f"cpu-f64-o{o}k{k}b{b}", device="cpu", dtype="float64", order=o, n_kicks=k, n_kicks_bend=b)
            for o, k, b in _RUNGS]
# Order-8 and cheaper order-6 rungs (7 Nov 2026): 21 kicks per step against 7 for order 6 and 3 for order 4; the converged order-6 rung stays last.
_LAST = CONFIGS.pop()
_O8 = [(8, 1, 2), (8, 2, 4), (8, 4, 8), (6, 2, 4), (6, 4, 8)]   # the order-6 pair gives the cheaper order-6 points to compare with
CONFIGS += [dict(name=f"cuda-f{t[-2:]}-o{o}k{k}b{b}", device="cuda", dtype=t, order=o, n_kicks=k, n_kicks_bend=b)
            for t in ("float32", "float64") for o, k, b in _O8]
CONFIGS += [dict(name=f"cpu-f64-o{o}k{k}b{b}", device="cpu", dtype="float64", order=o, n_kicks=k, n_kicks_bend=b) for o, k, b in _O8]
CONFIGS += [dict(name=f"cuda-f32ds-hybrid-o{o}k{k}b{b}", device="cuda", dtype="float32", order=o, n_kicks=k, n_kicks_bend=b, double_single="hybrid")
            for o, k, b in [(6, 4, 8), (8, 2, 4), (8, 4, 8)]]
CONFIGS.append(_LAST)
# Keep the finest original rung last; this opt-in control reuses native compilation.
CONFIGS.insert(-1, dict(name="cpu-f64-o6k8b16-compiled", device="cpu", dtype="float64", order=6,
                       n_kicks=8, n_kicks_bend=16, compile_cpu=True))
for mode in ('hybrid', 'full'):
    CONFIGS.insert(-1, dict(name=f"cuda-f32ds-{mode}-o6k8b16", device="cuda", dtype="float32", order=6,
                            n_kicks=8, n_kicks_bend=16, double_single=mode))

# Per-kind rungs (slipstream 1.1.0, 7 Nov 2026): o<order>q<quadrupole kicks>s<sextupole kicks>.  Bends are one exact step (exact_dipole), so b8 is inert.
# o6q8s8 is the old uniform o6k8 under the new option: its track must equal the old o6k8b16 bit for bit (defaults unchanged).
_PK = [(6, q, 1) for q in (2, 3, 4, 6, 8)] + [(6, 4, 2), (6, 8, 2), (6, 8, 8)] + [(4, q, 1) for q in (4, 8, 16)] + [(8, q, 1) for q in (1, 2, 3, 4)]


def _pk(prefix, dev, dtype, o, q, s, **extra):
    return dict(name=f"{prefix}-o{o}q{q}s{s}b8", device=dev, dtype=dtype, order=o, n_kicks=q, n_kicks_quad=q, n_kicks_sext=s, n_kicks_bend=8, **extra)


CONFIGS[-1:-1] = [_pk(p, "cuda", t, o, q, s) for p, t in (("cuda-f32", "float32"), ("cuda-f64", "float64")) for o, q, s in _PK]
CONFIGS[-1:-1] = [_pk("cpu-f64", "cpu", "float64", 6, q, 1) for q in (3, 4, 8)]
CONFIGS[-1:-1] = [_pk("cuda-f32ds-hybrid", "cuda", "float32", 6, q, 1, double_single="hybrid") for q in (3, 4, 6)]
# The paper's ring setting (o6q8s1, chosen 7 Nov 2026) in every mode the controls use: compiled CPU and the double-single presets.
CONFIGS[-1:-1] = [dict(_pk("cpu-f64", "cpu", "float64", 6, 8, 1, compile_cpu=True), name="cpu-f64-o6q8s1b8-compiled")]
CONFIGS[-1:-1] = [_pk(f"cuda-f32ds-{m}", "cuda", "float32", 6, 8, 1, double_single=m) for m in ("hybrid", "full")]
# Rank order: per-kind rungs below every uniform rung, so the converged uniform rung stays each device's finest (analyse.py: dq_self, dq_prev).
CONFIGS.sort(key=lambda c: 0 if "q" in c["name"].rsplit("-o", 1)[-1] else 1)


def build(lat, case, cfg):
    import torch
    from slipstream.tracking.native.lattice_import import ElementSpec
    from slipstream.tracking.native.ring import RingMap

    if cfg.get("double_single") and case.get("kind") == "grad":
        raise NotImplementedError("double-single benchmark presets are for ring tracking, not the Jacobian adapter")
    specs = [ElementSpec(e["name"], e["kind"], e["length"], angle=e.get("angle", 0.0), k1=e.get("k1", 0.0), k2=e.get("k2", 0.0))
             for e in lat["elements"]]
    dev, dt = torch.device(cfg["device"]), getattr(torch, cfg["dtype"])
    if dev.type == "cpu":
        torch.set_num_threads(1)  # ponytail: threads are fixed by the launcher env; add a job field if CPU scaling is studied
    ring = RingMap(specs, energy_eV=case["energy_eV"], mc2_eV=case["mass_eV"], device=dev, dtype=dt, n_kicks=cfg["n_kicks"],
                   n_kicks_bend=cfg["n_kicks_bend"], integration_order=cfg["order"],
                   n_kicks_quad=cfg.get("n_kicks_quad"), n_kicks_sext=cfg.get("n_kicks_sext"),
                   compile=dev.type == "cuda" or cfg.get("compile_cpu", False),
                   double_single=cfg.get("double_single", False))
    inv = dict(n=len(specs), length=sum(s.length for s in specs), angle=sum(s.angle for s in specs),
               k1L_abs=sum(abs(s.k1) * s.length for s in specs), k2L_abs=sum(abs(s.k2) * s.length for s in specs))
    return ring, inv


def track(ring, init, turns, cfg):
    import torch

    s = torch.as_tensor(init, dtype=torch.float64 if ring.double_single else ring.dtype, device=ring.device)
    r = ring.track(s, turns, aperture=(1.0, 1.0), record_every=1)  # records the state AFTER turns 1..T
    after = r.tbt.permute(1, 0, 2).double().cpu().numpy()  # (N, T, 4)
    return np.concatenate([init[:, None, [0, 1, 2, 3]], after[:, :-1]], axis=1)  # states at the START of turns 0..T-1


def ad_build(lat, case, cfg, knobs):
    ring, _ = build(lat, case, cfg)
    from common import nominal_strengths
    return dict(ring=ring, theta0=nominal_strengths(lat, knobs))


def ad_jacobian(m, init, turns, cfg, knobs):
    """Reverse-mode Jacobian of the state after ``turns - 1`` turns with respect to the family strengths (``with_strengths``)."""
    import torch

    ring = m["ring"]
    theta0 = torch.as_tensor(m["theta0"], dtype=ring.dtype, device=ring.device)
    s0 = torch.as_tensor(init, dtype=ring.dtype, device=ring.device)

    def f(theta):
        k = {"k1": {}, "k2": {}}
        for j, kn in enumerate(knobs):
            k[kn["kind"]][kn["family"]] = theta[j]
        r = ring.with_strengths(k1=k["k1"], k2=k["k2"])
        s = s0
        for _ in range(turns - 1):
            s = r.turn(s)
        return s[0, :4]

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
    import slipstream
    import torch
    v = dict(torch=torch.__version__, cuda=torch.version.cuda, slipstream_file=slipstream.__file__)
    try:
        import warp
        v["warp"] = warp.__version__
    except Exception:  # noqa: BLE001
        v["warp"] = None
    try:
        import importlib.metadata as md
        v["slipstream"] = md.version("slipstream-accel")
    except Exception:  # noqa: BLE001
        pass
    return v


def setup():
    import torch  # noqa: F401
    import slipstream.tracking.native.ring  # noqa: F401


if __name__ == "__main__":
    run_worker("slipstream", CONFIGS, build, track, sync, peak_mem, versions, setup, ad=(ad_build, ad_jacobian))

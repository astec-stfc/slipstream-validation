"""Model-substitution control for the element attribution (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P1).

element_attribution.py showed that each non-Slipstream code has one or two element maps that deviate from an exact reference, and noted that
the link to that code's ring tune gap was by correspondence only.  This control closes the loop: the native ring map (CPU float64, order 6,
8/16 kicks, the repository's converged setting) is run turn by turn element by element; all elements of one kind are then replaced by the *other
code's own element* (Cheetah and Bmad-X imported in-process, JuTrack through the harness's persistent Julia server, each at the setting that
was probed) and the tunes of the same five launches are re-measured with the harness's one estimator.  The tune shift of the hybrid is
compared with the code's own ring-tune gap against the native ring (both from 1024 turns of the same launches).

    python element_substitution_control.py [--codes cheetah,bmadx,jutrack] [--kinds quad,sext,bend,drift,all]
    python element_substitution_control.py --closure --codes cheetah,bmadx,jutrack   # all-element hybrids against the codes' own ring runs, component-wise

Coordinates: all codes use BSF (x, px/P0, y, py/P0, z, delta); the substituted element receives (x, px, y, py) and returns the four transverse
coordinates; delta is conserved and z does not feed back into a static lattice, so both stay native.
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.name == "nt" and os.environ.setdefault("CXX", "clang-cl")  # Windows torch.compile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "workers"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("BENCH_BMADX", ""))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402

CAMPAIGN = common.RESULTS / "2026-10-26_R1/ring_tunes"
OUT = common.EVIDENCE / "2026-10-26_R1/element_substitution_control.json"
SETTING = dict(cheetah="cpu-f64-dri16", bmadx="cpu-f64-q1s64", jutrack="cpu-f64-n32b64")  # the settings element_attribution probed
DT = torch.float64


def native_ring(case, lat):
    from slipstream.tracking.native.lattice_import import ElementSpec
    from slipstream.tracking.native.ring import RingMap

    torch.set_num_threads(1)
    specs = [ElementSpec(e["name"], e["kind"], e["length"], angle=e.get("angle", 0.0), k1=e.get("k1", 0.0), k2=e.get("k2", 0.0))
             for e in lat["elements"]]
    ring = RingMap(specs, energy_eV=case["energy_eV"], mc2_eV=case["mass_eV"], device="cpu", dtype=DT, n_kicks=8, n_kicks_bend=16,
                   integration_order=6, compile=True)
    assert len(ring.params) == len(lat["elements"])
    return ring


# ---------------------------------------------------------------------------------------- other codes' single-element maps
def cheetah_map(e, case, cfg):
    import cheetah

    kw = dict(device="cpu", dtype=DT)
    T = lambda v: torch.tensor(float(v), **kw)  # noqa: E731
    m, ns, L = cfg["method"], cfg["num_steps"], e["length"]
    if e["kind"] == "drift":
        el = cheetah.Drift(T(L), tracking_method=m, **kw)
    elif e["kind"] == "quad":
        el = cheetah.Quadrupole(T(L), k1=T(e["k1"]), num_steps=ns, tracking_method=m, **kw)
    elif e["kind"] == "sext":
        el = cheetah.Sextupole(T(L), k2=T(e["k2"]), tracking_method="linear" if m == "linear" else "second_order", **kw)
    else:
        el = cheetah.Dipole(T(L), angle=T(e["angle"]), dipole_e1=T(0.0), dipole_e2=T(0.0), k1=T(0.0), tracking_method=m, **kw)
    E0 = torch.tensor(case["energy_eV"], **kw)

    def f(s):
        p = torch.zeros(s.shape[0], 7, **kw)
        p[:, :5] = s[:, :5]
        p[:, 5] = torch.as_tensor(common.delta_to_cheetah_p(s[:, 5].numpy(), case["energy_eV"], case["mass_eV"]), **kw)
        p[:, 6] = 1.0
        out = el(cheetah.ParticleBeam(particles=p, energy=E0, **kw)).particles
        return torch.cat([out[:, :4], s[:, 4:]], 1)
    return f


def bmadx_map(e, case, cfg):
    from bmadx.bmad_torch.track_torch import Beam, TorchDrift, TorchQuadrupole, TorchSBend, TorchSextupole

    T = lambda v: torch.tensor(float(v), dtype=DT)  # noqa: E731
    p0c, L = common.p0c_eV(case["energy_eV"], case["mass_eV"]), e["length"]
    if e["kind"] == "drift":
        el = TorchDrift(T(L))
    elif e["kind"] == "quad":
        el = TorchQuadrupole(T(L), T(e["k1"]), NUM_STEPS=cfg["n_quad"])
    elif e["kind"] == "sext":
        el = TorchSextupole(T(L), T(e["k2"]), NUM_STEPS=cfg["n_sext"])
    else:
        el = TorchSBend(T(L), T(p0c), T(e["angle"] / L), FRINGE_AT="no_end", FRINGE_TYPE="none")
    P0, MC2 = torch.tensor(p0c, dtype=DT), torch.tensor(case["mass_eV"], dtype=DT)

    def f(s):
        b = el(Beam(s.clone(), P0, mc2=MC2))
        return torch.cat([torch.stack([b.x, b.px, b.y, b.py], -1), s[:, 4:]], 1)
    return f


_JULIA_LINE = [None]  # the Julia server holds ONE built line at a time, so a model is rebuilt whenever another element was built since


def jutrack_map(e, case, cfg):
    import w_jutrack as w

    model, _ = w.build(dict(elements=[e]), case, cfg)
    cmd = f"BUILD {(model['dir'] / 'lat.json').as_posix()} {cfg['n_steps']} {cfg['n_steps_bend']} {case['energy_eV']!r} {case['mass_eV']!r}"

    def f(s):
        if _JULIA_LINE[0] is not model:
            w._ask(cmd)
            _JULIA_LINE[0] = model
        out = w.track(model, s.numpy(), 2, cfg)[:, 1, :4]
        return torch.cat([torch.as_tensor(out, dtype=DT), s[:, 4:]], 1)
    _JULIA_LINE[0] = model  # w.build just built this element
    return f


MAPS = dict(cheetah=cheetah_map, bmadx=bmadx_map, jutrack=jutrack_map)


def config_of(code):
    import importlib

    w = importlib.import_module(f"w_{code}")
    return next(c for c in w.CONFIGS if c["name"] == SETTING[code])


# ---------------------------------------------------------------------------------------- hybrid tracking
def track(ring, init, turns, subst):
    """States at the START of turns 0..turns-1 (N, turns, 4); ``subst[i]`` replaces native element i."""
    s = torch.as_tensor(init, dtype=DT)
    rec = []
    with torch.no_grad():
        for _ in range(turns):
            rec.append(s[:, :4].clone())
            for i, p in enumerate(ring.params):
                f = subst.get(i)
                s = f(s) if f else ring._element(s, p, ring._energy_t, ring._mc2_t)[0]
    return torch.stack(rec, 1).numpy()


def substitution(code, kinds, lat, case, cfg):
    """One callable per distinct element (kind, length, k1, k2, angle), shared by every occurrence."""
    cache, subst = {}, {}
    for i, e in enumerate(lat["elements"]):
        if e["kind"] in kinds:
            key = (e["kind"], e["length"], e.get("k1", 0.0), e.get("k2", 0.0), e.get("angle", 0.0))
            if key not in cache:
                cache[key] = MAPS[code](e, case, cfg)
            subst[i] = cache[key]
    return subst


def gap(tbt, case, ref):
    qx, qy = common.tunes(tbt, case)
    return dict(qx=qx.tolist(), qy=qy.tolist(), dqx=(qx - ref[0]).tolist(), dqy=(qy - ref[1]).tolist(),
                max_abs=float(max(np.abs(qx - ref[0]).max(), np.abs(qy - ref[1]).max())))


def closure(a, out, ring, init, case, lat, ref):
    """Replace every element by the other code's own and compare the turn-by-turn state with that code's own ring run, per component."""
    out.setdefault("closure", {})
    for code in a.codes.split(","):
        if code == "jutrack":
            import w_jutrack as w
            w.setup()
        cfg = config_of(code)
        t0 = time.perf_counter()
        tbt = track(ring, init, a.turns, substitution(code, ("drift", "quad", "sext", "bend"), lat, case, cfg))
        own = np.load(CAMPAIGN / code / SETTING[code] / "p1/tbt.npz")["tbt"][:, :a.turns, :4]
        d = np.abs(tbt - own)
        comp = ("x_m", "px", "y_m", "py")
        out["closure"][code] = dict(config=cfg["name"], turns=a.turns, wall_s=time.perf_counter() - t0,
                                    max_residual_all_turns=dict(zip(comp, d.max(axis=(0, 1)).tolist())), max_residual_first_turn=dict(zip(comp, d[:, 1].max(axis=0).tolist())),
                                    tune=gap(tbt, case, ref), own_tune_gap=out["code_gap"].get(code, {}).get("max_abs"))
        OUT.write_text(json.dumps(out, indent=2) + "\n")
        c = out["closure"][code]
        print(f"{code:8s} all-element hybrid vs own ring: first-turn residual {max(c['max_residual_first_turn'].values()):.2e}, all turns {max(c['max_residual_all_turns'].values()):.2e}; "
              f"tune shift {c['tune']['max_abs']:.3e} vs own gap {c['own_tune_gap']:.3e}  ({c['wall_s']:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="cheetah,bmadx,jutrack")
    ap.add_argument("--kinds", default="quad,sext,bend,drift,all")
    ap.add_argument("--turns", type=int, default=1024)
    ap.add_argument("--closure", action="store_true", help="run only the all-element hybrids and compare them component-wise with the code's own ring tbt")
    a = ap.parse_args()
    case = common.load(HERE / "cases/ring_tunes.json")
    lat = common.load(HERE / "cases" / case["lattice"])
    init = common.launches(case)
    ring = native_ring(case, lat)
    out = json.loads(OUT.read_text()) if OUT.exists() else dict(runs=[])
    out["design"] = dict(note="Native ring (CPU float64, order 6, 8/16 kicks, compiled per element) with all elements of one kind replaced by the other code's own element; "
                              "the five ring_tunes launches, 1024 turns, harness NAFF estimator. Gap = the code's own ring tbt tunes minus the native baseline tunes.",
                         settings=SETTING, turns=a.turns, case_sha256=common.sha256(HERE / "cases/ring_tunes.json"), analysis_sha256=common.sha256(__file__))

    t0 = time.perf_counter()
    base = track(ring, init, a.turns, {})
    qx, qy = common.tunes(base, case)
    ref = (qx, qy)
    out["baseline"] = dict(qx=qx.tolist(), qy=qy.tolist(), wall_s=time.perf_counter() - t0)
    z = np.load(CAMPAIGN / "slipstream/cuda-f64-o6k8b16/p1/tbt.npz")["tbt"]
    out["baseline"]["vs_campaign_native_tune"] = gap(z, case, ref)["max_abs"]
    print("baseline", out["baseline"]["wall_s"], "s; vs campaign native tune", out["baseline"]["vs_campaign_native_tune"], flush=True)

    out.setdefault("code_gap", {})  # earlier invocations may have covered other codes
    for code in a.codes.split(","):
        out["code_gap"][code] = gap(np.load(CAMPAIGN / code / SETTING[code] / "p1/tbt.npz")["tbt"], case, ref)
        print(code, "own ring gap", out["code_gap"][code]["max_abs"], flush=True)

    if a.closure:
        return closure(a, out, ring, init, case, lat, ref)
    done = {(r["code"], r["kinds"]) for r in out["runs"]}
    for code in a.codes.split(","):
        if code == "jutrack":
            import w_jutrack as w
            w.setup()
        cfg = config_of(code)
        for kind in a.kinds.split(","):
            kinds = ("drift", "quad", "sext", "bend") if kind == "all" else (kind,)
            if (code, kind) in done:
                continue
            if code == "jutrack" and kind == "all":
                continue  # 100 000 Julia round trips; the per-kind runs carry the evidence
            t0 = time.perf_counter()
            sub = substitution(code, kinds, lat, case, cfg)
            tbt = track(ring, init, a.turns, sub)
            row = dict(code=code, config=cfg["name"], kinds=kind, n_elements=len(sub), wall_s=time.perf_counter() - t0, **gap(tbt, case, ref))
            out["runs"].append(row)
            OUT.write_text(json.dumps(out, indent=2) + "\n")
            print(f"{code:8s} {kind:6s} n={len(sub):3d}  hybrid-minus-native tune shift {row['max_abs']:.3e}   ({row['wall_s']:.0f} s)", flush=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

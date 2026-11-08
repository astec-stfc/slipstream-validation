"""Matrix-kick quadrupole prototype (7 November 2026): accuracy and GPU time against Slipstream's drift-kick step, same Warp kernel.

    python prototypes/mkm_prototype.py validate                      # one-quad check of x, px, y, py, tau against a fine drift-kick reference
    python prototypes/mkm_prototype.py tunes --dtype float64 --out results/2026-11-07_mkm_prototype/tunes_f64.json

Both variants run through ring_warp_mkm.py (``mkm_quads`` False is the package kernel unchanged), so the time ratio is like-for-like.  The tune test is
the benchmark's own (cases/ring_tunes.json, 1024 turns, five launches) against the corrected elegant reference.  Needs the slipstream package on
PYTHONPATH and a CUDA GPU with Warp; GPU memory use is a few MiB.
"""
import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
B = HERE.parent
sys.path.insert(0, str(B))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import ring_warp_mkm as rw  # noqa: E402
from slipstream.tracking.native.lattice_import import ElementSpec  # noqa: E402
from slipstream.tracking.native.ring import RingMap  # noqa: E402

REF = B / "results/2026-10-27_momentum/ring_tunes/elegant/cpu-f64-o6s16b32/p0/tbt.npz"


def specs_of(lat):
    return [ElementSpec(e["name"], e["kind"], e["length"], angle=e.get("angle", 0.0), k1=e.get("k1", 0.0), k2=e.get("k2", 0.0)) for e in lat["elements"]]


def packed_ring(specs, case, order, k, kb, dtype, mkm, kq=None):
    """``k`` steps per quadrupole and sextupole, or ``kq`` for the quadrupoles alone when given (the sextupoles keep ``k``)."""
    dev = torch.device("cuda")
    ring = RingMap(specs, energy_eV=case["energy_eV"], mc2_eV=case["mass_eV"], device=dev, dtype=dtype, n_kicks=k, n_kicks_bend=kb, integration_order=order)
    if kq is not None:
        ring.params = [p._replace(n_kicks=kq) if (p.angle == 0.0 and _f(p.k1) != 0.0 and _f(p.k2) == 0.0) else p for p in ring.params]
    packed = rw.pack(ring.params, dev, dtype, mkm_quads=mkm)
    assert packed is not None
    return packed


def _f(v):
    return float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)


def track(packed, init, turns, case, dtype):
    """States at the START of turns 0..T-1, shape (N, T, 6), float64 on the host; also the seconds taken (device sync, host copy included)."""
    dev = torch.device("cuda")
    buf = torch.as_tensor(np.ascontiguousarray(init[:, :6]), dtype=dtype, device=dev).contiguous()
    prepared = rw.prepare(buf, packed)
    rec = torch.empty((turns, buf.shape[0], 6), dtype=dtype, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(turns):
        rec[t].copy_(buf)
        if t < turns - 1:
            rw.turn(buf, packed, case["energy_eV"], case["mass_eV"], prepared=prepared)
    out = rec.permute(1, 0, 2).double().cpu().numpy()
    return out, time.perf_counter() - t0


def validate():
    case = common.load(B / "cases/ring_tunes.json")
    lat = common.load(B / "cases" / case["lattice"])
    quad = next(e for e in lat["elements"] if e["kind"] == "quad")
    qd = next(e for e in lat["elements"] if e["kind"] == "quad" and e["k1"] * quad["k1"] < 0)
    rng = np.random.default_rng(1)
    init = np.zeros((6, 6))
    init[:, :4] = rng.normal(scale=[1e-3, 1e-3, 1e-3, 1e-3], size=(6, 4))
    init[:, 5] = [0, 0, 5e-3, -5e-3, 1e-2, -1e-2]
    init[0, :4] = 0.0
    init[0, 5] = 0.0
    dt = torch.float64
    out = {}
    for label, el in (("focusing quad", quad), ("defocusing quad", qd)):
        specs = [ElementSpec("Q", "quad", el["length"], k1=el["k1"])]
        ref, _ = track(packed_ring(specs, case, 6, 128, 1, dt, False), init, 2, case, dt)
        ref = ref[:, 1]
        rows = []
        for name, mkm, order, k in (("drift-kick o4 k1", False, 4, 1), ("drift-kick o4 k4", False, 4, 4), ("drift-kick o6 k2", False, 6, 2),
                                    ("matrix-kick o4 k1", True, 4, 1), ("matrix-kick o4 k4", True, 4, 4), ("matrix-kick o6 k1", True, 6, 1)):
            got, _ = track(packed_ring(specs, case, order, k, 1, dt, mkm), init, 2, case, dt)
            err = np.abs(got[:, 1] - ref).max(axis=0)
            rows.append(dict(name=name, **{c: float(v) for c, v in zip(("x", "px", "y", "py", "tau", "delta"), err)}))
        out[label] = rows
        print(f"{label}: K1 L = {el['k1'] * el['length']:.4f}, length {el['length']} m  (max over six particles, delta up to 1e-2; reference: drift-kick o6 k128)")
        for r in rows:
            print(f"  {r['name']:20s}" + "".join(f" {c} {r[c]:9.2e}" for c in ("x", "px", "y", "py", "tau")))
    return out


def tunes(args):
    case = common.load(B / "cases" / f"{args.case}.json")
    lat = common.load(B / "cases" / case["lattice"])
    specs = specs_of(lat)
    tune_case = common.load(B / "cases/ring_tunes.json")
    qr = common.tunes(np.load(REF)["tbt"], tune_case) if args.case == "ring_tunes" else None   # other cases (particle scaling) are timed only
    init = common.launches(case)
    turns = int(case["turns"])
    dtype = getattr(torch, args.dtype)
    rows = []
    for tag in args.configs:
        mt = re.fullmatch(r"([dm])(\d)(?:k(\d+)|q(\d+)s(\d+))", tag)   # d4k4: order 4, 4 steps for quads and sextupoles; m6q1s2: matrix-kick quads, 1 and 2 steps
        mkm, o = mt[1] == "m", int(mt[2])
        kq, k = (int(mt[3]), int(mt[3])) if mt[3] else (int(mt[4]), int(mt[5]))
        packed = packed_ring(specs, case, o, k, 2 * k, dtype, mkm, kq=kq)
        times = []
        for _ in range(args.repeats + 1):
            tbt, dt = track(packed, init, turns, case, dtype)
            times.append(dt)
        dq = None if qr is None else max(float(np.abs(a - b).max()) for a, b in zip(common.tunes(tbt[:, :, :4], tune_case), qr))
        row = dict(config=tag, mkm=mkm, order=o, n_kicks=k, n_kicks_quad=kq, dtype=args.dtype, dq_ref=dq, cold_s=times[0], warm_s=float(np.median(times[1:])), warm_all_s=times[1:],
                   finite=bool(np.isfinite(tbt).all()))
        rows.append(row)
        print(f"{tag:6s} {args.dtype}: warm {row['warm_s']:.3f} s  cold {times[0]:.2f} s  dq {dq if dq is None else format(dq, '.2e')}", flush=True)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(dtype=args.dtype, rows=rows, torch=torch.__version__, sha256=common.sha256(__file__), kernel_sha256=common.sha256(HERE / "ring_warp_mkm.py")), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("validate", "tunes"))
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--case", default="ring_tunes", help="ring_tunes (tune error and time) or ring_scale_n<N> (time only)")
    ap.add_argument("--configs", nargs="+", default=["d4k4", "m4k4"], help="d<order>k<n> = package drift-kick, m<order>k<n> = matrix-kick quads, n steps for quads and sextupoles; "
                    "d6q2s4 / m6q1s4 = q steps for the quadrupoles and s for the sextupoles")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(B / "results/2026-11-07_mkm_prototype/tunes.json"))
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.4)   # the 8 GiB card also runs the desktop; never spill to host RAM
    if a.mode == "validate":
        validate()
    else:
        tunes(a)


if __name__ == "__main__":
    main()

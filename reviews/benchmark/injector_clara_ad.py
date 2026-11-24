"""Reduced-CLARA validation of ``Injector.track_differentiable`` (plans/NATIVE_INJECTOR_DIFFERENTIABLE_PLAN.md,
Phase 5): the real 5 pC deck from the cathode to ``--z-stop`` at one fixed time step.

Modes (any combination, ``--modes``):

- ``compare``: endpoint statistics of ``track()`` (``mesh="fixed"`` and ``"adaptive"``, same fixed step,
  ``dt_adaptive`` off) against ``track_differentiable`` (bunch-fitted box, and ``--pinned-box`` if given)
  at the same step count, with the cathode image off and/or on (``--image``). The differentiable image
  runs for as many steps as ``track()``'s fixed mode keeps it (centroid within ``image_threshold_m``).
- ``grad``: warm forward without a graph, then forward+backward of ``emit_x`` w.r.t. the knobs with
  ``checkpoint_every`` (default ``sqrt(n_steps)``); wall time and, on CUDA, peak memory.
- ``fd``: autograd vs central differences for every knob at each ``--fd-h`` (relative to the knob's
  scale), on ``--fd-device`` (default CPU float64: bit-repeatable; CUDA ``scatter_add`` is not).

Example (Windows, from reviews/benchmark, with SSV_WORKBENCH set)::

    $env:CXX = "clang-cl"
    python injector_clara_ad.py --modes compare grad --image off on --json clara_autograd.json
    python injector_clara_ad.py --modes fd --fd-particles 1024 --fd-h 1e-4 1e-6 1e-7
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
import warnings
from pathlib import Path

import torch

from slipstream.tracking.native.injector import Injector

import common


def deck() -> Path:
    """The CLARA 5 pC ASTRA deck (injector400.in, its field maps and laser_input.astra) of the workbench checkout, ``SSV_WORKBENCH``."""
    if common.WORKBENCH is None:
        raise SystemExit("set SSV_WORKBENCH to the neg_coating_rwi checkout: the CLARA 5 pC deck is slipstream_dev/examples/clara_linac/ASTRA/5pC")
    return common.WORKBENCH / "slipstream_dev" / "examples" / "clara_linac" / "ASTRA" / "5pC" / "injector400.in"


STATS = ("Ekin_MeV", "sigma_x_mm", "sigma_z_mm", "emit_x_mm_mrad", "emit_can_x_mm_mrad", "emit_z_kev_mm",
         "dE_rms_keV")
KNOBS = ("cavities[0].phase_deg", "cavities[0].e_max_V_per_m", "solenoids[0].b_max_T")
KNOB_SCALE = {"cavities[0].phase_deg": 1.0}  # FD step = h * scale; others scale with their own value (every phase_deg: 1 degree)


def _machine(args, image: bool) -> Injector:
    inj = Injector.from_astra(args.deck)
    inj.auto_phase(verbose=False)
    inj.space_charge.dt_adaptive = False
    inj.space_charge.cathode_image = image
    inj.space_charge.nz = args.nz
    if args.dt_s:
        inj.dt_s = args.dt_s
    return inj


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _image_steps(inj: Injector, beam: dict, ref: dict) -> int:
    """Steps until ``track()``'s fixed-mode centroid leaves ``image_threshold_m`` (where it drops the image)."""
    tr = ref["trajectory"]
    t0 = min(0.0, float(beam["emission_time_s"].min()))
    z_cut = inj._reference_particle()[0] + inj.space_charge.image_threshold_m
    t_cut = next(t for z, t in zip(tr["z_m"], tr["t_ns"]) if z >= z_cut) * 1e-9
    return int(round((t_cut - t0) / inj.dt_s))


def nominal_image_steps(inj: Injector, args, n_particles: int) -> int:
    """Steps of the nominal ``track()`` (fixed mesh, CPU float64) that keep the cathode image; 0 when ``inj`` has it off.

    ``track_differentiable`` takes the image as a step count, ``sc_image_steps``. The derivative runs hold this nominal
    count fixed, like the step schedule and the emission times: its dependence on a knob is a jump, not a slope."""
    if not inj.space_charge.cathode_image:
        return 0
    torch.manual_seed(args.seed)
    beam = inj.initial_beam(n_particles, "cpu", torch.float64)
    torch.manual_seed(args.seed)
    ref = copy.deepcopy(inj).track(n_particles=n_particles, device="cpu", dtype=torch.float64, mesh="fixed", use_compile=False,
                                   z_stop_m=2 * inj.space_charge.image_threshold_m + 0.1, record_every=1, verbose=False)
    return _image_steps(inj, beam, ref)


def compare(args, out: dict) -> None:
    dtype = getattr(torch, args.dtype)
    for image in args.image:
        on = image == "on"
        base = _machine(args, on)
        torch.manual_seed(args.seed)
        beam = base.initial_beam(args.particles, args.device, dtype)
        rows = {}
        for mesh in ("fixed", "adaptive"):  # warm-up: keep track()'s compile out of its timing
            copy.deepcopy(base).track(n_particles=args.particles, device=args.device, dtype=dtype, mesh=mesh,
                                      z_stop_m=0.01, record_every=0, verbose=False)
        for mesh in ("fixed", "adaptive"):
            torch.manual_seed(args.seed)
            _sync(args.device)
            t0 = time.time()
            ref = copy.deepcopy(base).track(n_particles=args.particles, device=args.device, dtype=dtype, mesh=mesh,
                                            z_stop_m=args.z_stop, record_every=10, verbose=False)
            rows[f"track {mesh}"] = {k: ref["trajectory"][k][-1] for k in STATS} | {
                "time_s": time.time() - t0, "n_steps": ref["n_steps"]}
            if mesh == "fixed":
                n_steps, ref_fixed = ref["n_steps"], ref
        image_steps = _image_steps(base, beam, ref_fixed) if on else 0
        boxes = {"diff fitted": None} | ({"diff pinned": tuple(args.pinned_box)} if args.pinned_box else {})
        for label, box in boxes.items():
            for rep in range(2):  # the first pays the compile
                _sync(args.device)
                t0 = time.time()
                with torch.no_grad():
                    res = base.track_differentiable(n_steps=n_steps, beam=beam, device=args.device, dtype=dtype,
                                                    use_compile=args.compile, sc_box_m=box, sc_image_steps=image_steps)
                _sync(args.device)
            rows[label] = {k: float(res[k]) for k in STATS} | {
                "time_s": time.time() - t0, "n_steps": n_steps, "sc_box_overflow": res["sc_box_overflow"],
                "sc_image_steps": image_steps}
        out["compare"][f"image {image}"] = rows
        print(f"\nimage {image} (z_stop {args.z_stop} m, {args.particles} particles, {n_steps} steps"
              + (f", image for {image_steps} steps" if on else "") + ")")
        print(f"  {'':16s}" + "".join(f"{k[:14]:>15s}" for k in STATS) + f"{'time_s':>9s}")
        for label, r in rows.items():
            print(f"  {label:16s}" + "".join(f"{r[k]:15.5f}" for k in STATS) + f"{r['time_s']:9.1f}")


def grad(args, out: dict) -> None:
    dtype = getattr(torch, args.dtype)
    inj = _machine(args, False)
    torch.manual_seed(args.seed)
    beam = inj.initial_beam(args.particles, args.device, dtype)
    n_steps = args.n_steps or _steps_to(inj, beam, args)
    every = args.checkpoint_every or int(math.sqrt(n_steps))
    kw = dict(n_steps=n_steps, beam=beam, device=args.device, dtype=dtype, use_compile=args.compile)
    nominal = _nominal(inj, args.knobs)
    for rep in range(2):
        _sync(args.device)
        t0 = time.time()
        with torch.no_grad():
            inj.track_differentiable(**kw)
        _sync(args.device)
        t_fwd = time.time() - t0
    res = {"n_steps": n_steps, "checkpoint_every": every, "forward_no_grad_s": t_fwd}
    for rep in range(2):
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            m0 = torch.cuda.memory_allocated()
        t0 = time.time()
        knobs = {k: torch.tensor(v, dtype=torch.float64, device=args.device, requires_grad=True)
                 for k, v in nominal.items()}
        o = inj.track_differentiable(knobs, checkpoint_every=every, **kw)
        g = torch.autograd.grad(o["emit_x_mm_mrad"], list(knobs.values()))
        _sync(args.device)
        res[f"forward_backward_s_rep{rep}"] = time.time() - t0
        if args.device == "cuda":
            res["peak_MiB"] = (torch.cuda.max_memory_allocated() - m0) / 2**20
    res["d_emit_x"] = dict(zip(nominal, (float(v) for v in g)))
    out["grad"] = res
    print("\ngrad:", json.dumps(res, indent=2))


def fd(args, out: dict) -> None:
    dtype = getattr(torch, args.fd_dtype)
    inj = _machine(args, args.image[0] == "on")  # fd takes the first of --image
    torch.manual_seed(args.seed)
    beam = inj.initial_beam(args.fd_particles, args.fd_device, dtype)
    n_steps = args.n_steps or _steps_to(inj, beam, args)
    image_steps = nominal_image_steps(inj, args, args.fd_particles)
    print(f"cathode image {args.image[0]}: {image_steps} of {n_steps} steps")
    kw = dict(n_steps=n_steps, beam=beam, device=args.fd_device, dtype=dtype, use_compile=args.compile, sc_image_steps=image_steps)
    nominal = _nominal(inj, args.knobs)
    objs = ("emit_x_mm_mrad", "sigma_x_mm", "sigma_z_mm", "emit_z_kev_mm")
    knob_list = args.fd_knobs or list(nominal)
    t0 = time.time()
    ad_file = Path(args.ad_json) if args.ad_json else None
    if ad_file and ad_file.exists():  # the autograd pass of the full line needs ~12 GiB of host memory: take it once, share it between the knob shards
        ad = json.loads(ad_file.read_text())["autograd"]
    else:
        knobs = {k: torch.tensor(v, dtype=torch.float64, device=args.fd_device, requires_grad=True)
                 for k, v in nominal.items()}
        o = inj.track_differentiable(knobs, checkpoint_every=int(math.sqrt(n_steps)), **kw)
        ad = {ob: [float(v) for v in torch.autograd.grad(o[ob], list(knobs.values()), retain_graph=True)] for ob in objs}
        if ad_file:
            ad_file.write_text(json.dumps({"n_steps": n_steps, "particles": args.fd_particles, "image_steps": image_steps, "knobs": list(nominal), "autograd": ad}))
    print(f"\nfd: autograd {time.time() - t0:.0f}s ({args.fd_particles} particles, {n_steps} steps, {args.fd_device})")
    if args.ad_only:
        return
    rows = []
    for i, k in enumerate(nominal):
        if k not in knob_list:
            continue
        for h_rel in args.fd_h:
            h = h_rel * (1.0 if k.endswith("phase_deg") else abs(nominal[k]))
            with torch.no_grad():
                fp, fm = (inj.track_differentiable({**nominal, k: nominal[k] + s * h}, **kw) for s in (1, -1))
            for ob in objs:
                fdv = (float(fp[ob]) - float(fm[ob])) / (2 * h)
                rows.append({"knob": k, "h": h, "objective": ob, "autograd": ad[ob][i], "fd": fdv,
                             "rel": abs(ad[ob][i] - fdv) / abs(ad[ob][i])})
                print(f"  {k:28s} h={h:9.2e} {ob:16s} AD {ad[ob][i]: .6e} FD {fdv: .6e} rel {rows[-1]['rel']:.2e}")
    out["fd"] = {"n_steps": n_steps, "particles": args.fd_particles, "image_steps": image_steps, "rows": rows}


def _nominal(inj: Injector, which: str = "gun") -> dict:
    """``gun``: the three gun knobs of the original check; ``all``: phase and field of every cavity, every solenoid
    and the bunch charge (``charge_scale``)."""
    if which == "gun":
        return {KNOBS[0]: inj.cavities[0].phase_deg, KNOBS[1]: inj.cavities[0].e_max_V_per_m,
                KNOBS[2]: inj.solenoids[0].b_max_T}
    out = {}
    for i, c in enumerate(inj.cavities):
        if c.e_max_V_per_m:
            out[f"cavities[{i}].phase_deg"] = c.phase_deg
            out[f"cavities[{i}].e_max_V_per_m"] = c.e_max_V_per_m
    out.update({f"solenoids[{i}].b_max_T": s.b_max_T for i, s in enumerate(inj.solenoids) if s.b_max_T})
    out["charge_scale"] = 1.0
    return out


def _steps_to(inj: Injector, beam: dict, args) -> int:
    """Step count at which ``track()``'s centroid first reaches ``--z-stop`` (cheap: no space charge)."""
    probe = copy.deepcopy(inj)
    probe.space_charge.enabled = False
    torch.manual_seed(args.seed)
    return probe.track(n_particles=64, device="cpu", dtype=torch.float64, use_compile=False, z_stop_m=args.z_stop,
                       record_every=0, verbose=False)["n_steps"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--deck", type=Path, default=None, help="default: the workbench's CLARA 5 pC deck")
    p.add_argument("--modes", nargs="+", default=["compare", "grad"], choices=["compare", "grad", "fd"])
    p.add_argument("--z-stop", type=float, default=1.0)
    p.add_argument("--n-steps", type=int, default=0, help="grad/fd step count (default: reach --z-stop)")
    p.add_argument("--particles", type=int, default=16384)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--nz", type=int, default=512)
    p.add_argument("--image", nargs="+", default=["off"], choices=["off", "on"])
    p.add_argument("--pinned-box", type=float, nargs=2, metavar=("R_MAX_M", "Z_HALF_M"))
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--fd-particles", type=int, default=1024)
    p.add_argument("--fd-device", default="cpu")
    p.add_argument("--fd-dtype", default="float64")
    p.add_argument("--fd-h", type=float, nargs="+", default=[1e-5, 1e-7])
    p.add_argument("--knobs", choices=["gun", "all"], default="gun")
    p.add_argument("--fd-knobs", nargs="+", default=None, help="fd: only these knobs (a shard of --knobs)")
    p.add_argument("--ad-json", type=Path, default=None, help="fd: file of the autograd derivatives: read when it exists, else written after the autograd pass")
    p.add_argument("--ad-only", action="store_true", help="fd: stop after the autograd pass (needs --ad-json)")
    p.add_argument("--dt-s", type=float, default=0.0, help="fixed step of the differentiable run (default: the machine's own)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    args.deck = args.deck or deck()
    warnings.simplefilter("ignore")
    out = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "env": {"torch": torch.__version__, "platform": platform.platform(),
                   "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
           "compare": {}}
    for mode in args.modes:
        {"compare": compare, "grad": grad, "fd": fd}[mode](args, out)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()




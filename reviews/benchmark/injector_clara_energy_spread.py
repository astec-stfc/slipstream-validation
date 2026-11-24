"""Autograd against central differences for the rms energy spread at the end of the line (CLARA 5 pC deck, CPU float64, six knobs of ``--knobs all``).

The energy spread is the one output of the sensitivity profile that the main ladder (``injector_clara_ad --modes fd``: emittance, size, length and
longitudinal emittance) does not cover; this checks it at two relative steps. Same machine, particles, fixed step and checkpointed autograd as that ladder.

    python injector_clara_energy_spread.py --json reports/Slipstream/benchmarks/results/2026-11-03_injector_fd_energy_spread.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import injector_clara_ad as b



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deck", type=Path, default=None)
    p.add_argument("--particles", type=int, default=1024)
    p.add_argument("--dt-s", type=float, default=5e-13)
    p.add_argument("--z-stop", type=float, default=3.37147)
    p.add_argument("--fd-h", type=float, nargs="+", default=[1e-6, 1e-7])
    p.add_argument("--image", choices=["off", "on"], default="off", help="cathode image in the space-charge solve (the nominal step count is held fixed)")
    p.add_argument("--json", type=Path, required=True)
    a = p.parse_args()
    a.deck = a.deck or b.deck()
    args = argparse.Namespace(deck=a.deck, nz=512, dt_s=a.dt_s, seed=0, n_steps=0, z_stop=a.z_stop, compile=False)
    inj = b._machine(args, a.image == "on")
    image_steps = b.nominal_image_steps(inj, args, a.particles)
    print(f"cathode image {a.image}: {image_steps} steps")
    nominal = b._nominal(inj, "all")
    dtype = torch.float64
    torch.manual_seed(0)
    beam = inj.initial_beam(a.particles, "cpu", dtype)
    n_steps = b._steps_to(inj, beam, args)
    kw = dict(n_steps=n_steps, beam=beam, device="cpu", dtype=dtype, use_compile=False, sc_image_steps=image_steps)
    t0 = time.time()
    knobs = {k: torch.tensor(v, dtype=dtype, requires_grad=True) for k, v in nominal.items()}
    o = inj.track_differentiable(knobs, checkpoint_every=max(1, int(math.sqrt(n_steps))), **kw)
    ad = dict(zip(nominal, (float(v) for v in torch.autograd.grad(o["dE_rms_keV"], list(knobs.values())))))
    print(f"autograd {time.time() - t0:.0f} s, {n_steps} steps, dE_rms {float(o['dE_rms_keV']):.4f} keV")
    rows = []
    for k in nominal:
        for h_rel in a.fd_h:
            h = h_rel * (1.0 if k.endswith("phase_deg") else abs(nominal[k]))
            with torch.no_grad():
                fp, fm = (float(inj.track_differentiable({**nominal, k: nominal[k] + s * h}, **kw)["dE_rms_keV"]) for s in (1, -1))
            fd = (fp - fm) / (2 * h)
            rows.append({"knob": k, "h_rel": h_rel, "h": h, "autograd": ad[k], "fd": fd, "rel": abs(ad[k] - fd) / abs(ad[k])})
            print(f"  {k:28s} h_rel {h_rel:.0e} AD {ad[k]: .6e} FD {fd: .6e} rel {rows[-1]['rel']:.2e}")
            a.json.write_text(json.dumps({"n_steps": n_steps, "particles": a.particles, "z_stop_m": a.z_stop, "dt_s": a.dt_s, "image": a.image, "image_steps": image_steps, "rows": rows,
                                          "env": {"torch": torch.__version__}}, indent=1))


if __name__ == "__main__":
    main()

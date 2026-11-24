"""Where along the line each knob acts: autograd sensitivities of the beam at a series of stations (CLARA 5 pC deck, CPU float64).

For every station ``z_stop`` the injector is tracked to that plane with ``Injector.track_differentiable`` (one fixed step, checkpointed autograd, bunch-fitted
box) and the derivative of each beam statistic with respect to the six knobs of ``injector_clara_ad --knobs all`` is taken with one backward pass per
statistic. The statistics and their derivatives are written to --json; ``reports/Slipstream/benchmarks/injector_ad_profile_analysis.py`` summarises them.

    python injector_clara_profile.py --json reports/Slipstream/benchmarks/results/2026-11-03_injector_ad_profile.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import injector_clara_ad as b

STATIONS = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.3, 1.7, 2.2, 2.7, 3.37147)
OBJECTIVES = ("emit_x_mm_mrad", "sigma_x_mm", "sigma_z_mm", "emit_z_kev_mm", "Ekin_MeV", "dE_rms_keV")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deck", type=Path, default=None)
    p.add_argument("--particles", type=int, default=1024)
    p.add_argument("--dt-s", type=float, default=5e-13)
    p.add_argument("--stations", type=float, nargs="+", default=list(STATIONS))
    p.add_argument("--image", choices=["off", "on"], default="off", help="cathode image in the space-charge solve (the nominal step count is held fixed)")
    p.add_argument("--json", type=Path, required=True)
    a = p.parse_args()
    a.deck = a.deck or b.deck()
    args = argparse.Namespace(deck=a.deck, nz=512, dt_s=a.dt_s, seed=0, n_steps=0, z_stop=0.0, compile=False)
    inj = b._machine(args, a.image == "on")
    image_steps = b.nominal_image_steps(inj, args, a.particles)
    print(f"cathode image {a.image}: {image_steps} steps")
    nominal = b._nominal(inj, "all")
    dtype = torch.float64
    rows = []
    for z_stop in a.stations:
        args.z_stop = z_stop
        torch.manual_seed(0)
        beam = inj.initial_beam(a.particles, "cpu", dtype)
        n_steps = b._steps_to(inj, beam, args)
        t0 = time.time()
        knobs = {k: torch.tensor(v, dtype=dtype, requires_grad=True) for k, v in nominal.items()}
        o = inj.track_differentiable(knobs, n_steps=n_steps, beam=beam, device="cpu", dtype=dtype, use_compile=False,
                                     sc_image_steps=image_steps, checkpoint_every=max(1, int(math.sqrt(n_steps))))
        row = {"z_stop_m": z_stop, "n_steps": n_steps, "z_reached_m": float(o["z_m"]), "values": {}, "grad": {}}
        for ob in OBJECTIVES:
            row["values"][ob] = float(o[ob])
            g = torch.autograd.grad(o[ob], list(knobs.values()), retain_graph=True)
            row["grad"][ob] = dict(zip(nominal, (float(v) for v in g)))
        row["seconds"] = time.time() - t0
        rows.append(row)
        print(f"z_stop {z_stop:6.3f} reached {row['z_reached_m']:.3f} m, {n_steps} steps, {row['seconds']:.0f} s")
        a.json.write_text(json.dumps({"nominal": nominal, "particles": a.particles, "dt_s": a.dt_s, "image": a.image, "image_steps": image_steps, "rows": rows,
                                      "env": {"torch": torch.__version__}}, indent=1))


if __name__ == "__main__":
    main()

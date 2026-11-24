"""Jacobian tolerance budget of the CLARA 5 pC injector against Monte Carlo (CPU float64, ``Injector.track_differentiable``, one fixed step, 1024 particles).

Six knobs (phase and field of both cavities, the gun solenoid field, the bunch charge) are jittered with independent Gaussian errors of the standard deviations in
``SIGMA`` times a scale; five end-of-line outputs are recorded. The Jacobian prediction is ``sigma_out^2 = sum_i (d out / d knob_i * sigma_i)^2`` from one autograd
pass; ``oat`` evaluates each knob alone at plus and minus ``OAT_SCALE`` sigma (the one-at-a-time budget, and the curvature of each response); ``mc`` runs samples.
Samples are split between processes (``--part k --parts m``) and every process appends to its own file.

    python injector_clara_tolerance.py --mode jacobian --json jac.json
    python injector_clara_tolerance.py --mode mc --part 0 --parts 3 --json mc0.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import injector_clara_ad as b

OUTPUTS = ("emit_x_mm_mrad", "sigma_x_mm", "sigma_z_mm", "Ekin_MeV", "dE_rms_keV")
KNOBS = ["cavities[0].phase_deg", "cavities[0].e_max_V_per_m", "cavities[1].phase_deg", "cavities[1].e_max_V_per_m", "solenoids[0].b_max_T", "charge_scale"]
# standard deviation of each knob at scale 1: degrees for the phases, a relative error for the others
SIGMA = {"cavities[0].phase_deg": 0.1, "cavities[0].e_max_V_per_m": 1e-3, "cavities[1].phase_deg": 0.1, "cavities[1].e_max_V_per_m": 1e-3,
         "solenoids[0].b_max_T": 1e-4, "charge_scale": 1e-2}
OAT_SCALE = 10.0
SCALES = (1.0, 5.0, 20.0)


def absolute_sigma(knob, nominal):
    return SIGMA[knob] if knob.endswith("phase_deg") else SIGMA[knob] * abs(nominal[knob])


class Setup:
    def __init__(self, dt_s, image=False):
        args = argparse.Namespace(deck=b.deck(), nz=512, dt_s=dt_s, seed=0, n_steps=0, z_stop=3.37147, compile=False)
        self.inj = b._machine(args, image)
        self.nominal = b._nominal(self.inj, "all")
        torch.manual_seed(0)
        self.beam = self.inj.initial_beam(1024, "cpu", torch.float64)
        self.n_steps = b._steps_to(self.inj, self.beam, args)
        self.image_steps = b.nominal_image_steps(self.inj, args, 1024)  # held fixed in every run, like the step schedule
        self.kw = dict(n_steps=self.n_steps, beam=self.beam, device="cpu", dtype=torch.float64, use_compile=False, sc_image_steps=self.image_steps)
        self.sigma = {k: absolute_sigma(k, self.nominal) for k in KNOBS}

    def forward(self, values):
        with torch.no_grad():
            o = self.inj.track_differentiable({k: float(v) for k, v in values.items()}, **self.kw)
        return {ob: float(o[ob]) for ob in OUTPUTS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["jacobian", "mc"], required=True)
    p.add_argument("--dt-s", type=float, default=5e-13)
    p.add_argument("--scales", type=float, nargs="+", default=list(SCALES))
    p.add_argument("--n", type=int, default=40, help="samples per scale (over all parts)")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--image", choices=["off", "on"], default="off", help="cathode image in the space-charge solve")
    p.add_argument("--json", type=Path, required=True)
    a = p.parse_args()
    s = Setup(a.dt_s, a.image == "on")
    res = {"mode": a.mode, "n_steps": s.n_steps, "dt_s": a.dt_s, "particles": 1024, "image": a.image, "image_steps": s.image_steps, "nominal": s.nominal, "sigma_abs": s.sigma, "sigma_rel_or_deg": SIGMA,
           "env": {"torch": torch.__version__}}
    t0 = time.time()
    if a.mode == "jacobian":
        res["f0"] = s.forward(s.nominal)
        print("nominal", res["f0"], flush=True)
        knobs = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in s.nominal.items()}
        o = s.inj.track_differentiable(knobs, checkpoint_every=max(1, int(math.sqrt(s.n_steps))), **s.kw)
        res["grad"] = {ob: dict(zip(KNOBS, (float(g) for g in torch.autograd.grad(o[ob], [knobs[k] for k in KNOBS], retain_graph=True)))) for ob in OUTPUTS}
        res["autograd_s"] = time.time() - t0
        a.json.write_text(json.dumps(res, indent=1))
        print(f"autograd {res['autograd_s']:.0f} s", flush=True)
        res["oat"] = {}
        for k in KNOBS:
            res["oat"][k] = {}
            for sign in (+1, -1):
                v = dict(s.nominal)
                v[k] = s.nominal[k] + sign * OAT_SCALE * s.sigma[k]
                res["oat"][k]["plus" if sign > 0 else "minus"] = s.forward(v)
            print("oat", k, flush=True)
            res["oat_scale"] = OAT_SCALE
            res["oat_s"] = time.time() - t0
            a.json.write_text(json.dumps(res, indent=1))
        return
    res["samples"] = []
    for scale in a.scales:
        rng = np.random.default_rng(1000 + int(round(10 * scale)))
        z = rng.standard_normal((a.n, len(KNOBS)))
        for j in range(a.n):
            if j % a.parts != a.part:
                continue
            v = {k: s.nominal[k] + scale * s.sigma[k] * z[j, i] for i, k in enumerate(KNOBS)}
            out = s.forward(v)
            res["samples"].append({"scale": scale, "index": j, "z": [float(x) for x in z[j]], "out": out, "t_s": time.time() - t0})
            a.json.write_text(json.dumps(res, indent=1))
            print(f"scale {scale} sample {j} {time.time() - t0:.0f} s emit {out['emit_x_mm_mrad']:.5f}", flush=True)


if __name__ == "__main__":
    main()

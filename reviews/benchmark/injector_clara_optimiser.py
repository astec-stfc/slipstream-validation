"""Five-knob tuning of the CLARA 5 pC injector at equal wall time: L-BFGS-B with autograd, Nelder-Mead and Bayesian optimisation (CPU float64).

Knobs (offsets from the deck, in natural units): gun phase [deg], gun field [%], gun solenoid field [%], laser spot size [%], linac-structure phase [deg].
Objective: J = emittance_n,x [mm mrad] + 0.05 (E_kin - 30 MeV)^2 at the end of the line (3.37 m), so that the linac phase matters (through the energy) as well as
the gun, the solenoid and the spot. Each evaluation is one ``Injector.track_differentiable`` run, with its backward pass for the autograd method. One method per
process (``--method``); the wall time of every evaluation, including the backward pass or the surrogate fit, is recorded, and a method stops when its budget is spent.

    python injector_clara_optimiser.py --method lbfgs_ad --json out/lbfgs_ad.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch

import injector_clara_ad as b

NAMES = ["gun phase [deg]", "gun field [%]", "solenoid [%]", "spot size [%]", "linac phase [deg]"]
START = np.array([-3.0, -2.0, 3.0, 8.0, 3.0])
BOUND = 8.0                     # every knob is searched within +-BOUND natural units
E_TARGET_MEV, E_WEIGHT = 30.0, 0.05


class Budget(Exception):
    pass


class Problem:
    def __init__(self, dt_s, particles, image=False, z_stop=3.37147):
        args = argparse.Namespace(deck=b.deck(), nz=512, dt_s=dt_s, seed=0, n_steps=0, z_stop=z_stop, compile=False)
        self.inj = b._machine(args, image)
        self.nom = b._nominal(self.inj, "all")
        torch.manual_seed(0)
        self.beam = self.inj.initial_beam(particles, "cpu", torch.float64)
        self.n_steps = b._steps_to(self.inj, self.beam, args)
        self.image_steps = b.nominal_image_steps(self.inj, args, particles)  # held fixed in every run, like the step schedule
        self.kw = dict(n_steps=self.n_steps, beam=self.beam, device="cpu", dtype=torch.float64, use_compile=False, sc_image_steps=self.image_steps,
                       checkpoint_every=max(1, int(math.sqrt(self.n_steps))))

    def knobs(self, x):
        n = self.nom
        return {"cavities[0].phase_deg": n["cavities[0].phase_deg"] + x[0],
                "cavities[0].e_max_V_per_m": n["cavities[0].e_max_V_per_m"] * (1 + 0.01 * x[1]),
                "solenoids[0].b_max_T": n["solenoids[0].b_max_T"] * (1 + 0.01 * x[2]),
                "spot_scale": 1 + 0.01 * x[3],
                "cavities[1].phase_deg": n["cavities[1].phase_deg"] + x[4],
                "cavities[1].e_max_V_per_m": n["cavities[1].e_max_V_per_m"]}

    def value(self, x):
        with torch.no_grad():
            o = self.inj.track_differentiable(self.knobs(torch.as_tensor(x, dtype=torch.float64)), **self.kw)
        return float(self._j(o)), float(o["emit_x_mm_mrad"]), float(o["Ekin_MeV"])

    def value_grad(self, x):
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        o = self.inj.track_differentiable(self.knobs(xt), **self.kw)
        j = self._j(o)
        g = torch.autograd.grad(j, xt)[0]
        return float(j), g.numpy().copy(), float(o["emit_x_mm_mrad"]), float(o["Ekin_MeV"])

    @staticmethod
    def _j(o):
        return o["emit_x_mm_mrad"] + E_WEIGHT * (o["Ekin_MeV"] - E_TARGET_MEV) ** 2


class Recorder:
    def __init__(self, budget_s):
        self.t0, self.budget, self.trace, self.best = time.perf_counter(), budget_s, [], math.inf

    def add(self, x, j, eps, ekin, kind):
        self.best = min(self.best, j)
        self.trace.append({"t_s": time.perf_counter() - self.t0, "n_eval": len(self.trace) + 1, "J": j, "J_best": self.best, "eps": eps, "Ekin": ekin,
                           "x": [float(v) for v in x], "kind": kind})
        print(f"{self.trace[-1]['t_s']:7.0f} s  eval {len(self.trace):3d}  J {j:.5f}  best {self.best:.5f}  eps {eps:.4f}  Ekin {ekin:.3f}", flush=True)

    def check(self):
        if time.perf_counter() - self.t0 > self.budget:
            raise Budget


def run_lbfgs(pb, rec):
    from scipy.optimize import minimize

    def fun(x):
        rec.check()
        j, g, eps, ekin = pb.value_grad(x)
        rec.add(x, j, eps, ekin, "value+grad")
        return j, g

    minimize(fun, START, jac=True, method="L-BFGS-B", bounds=[(-BOUND, BOUND)] * 5, options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-7})


def run_nelder_mead(pb, rec):
    from scipy.optimize import minimize

    def fun(x):
        rec.check()
        j, eps, ekin = pb.value(x)
        rec.add(x, j, eps, ekin, "value")
        return j

    simplex = np.vstack([START] + [START + 2.0 * np.eye(5)[i] for i in range(5)])
    minimize(fun, START, method="Nelder-Mead", bounds=[(-BOUND, BOUND)] * 5,
             options={"initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-8, "maxfev": 100000, "adaptive": True})


def run_bayes(pb, rec, seed):
    from botorch.acquisition import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    lo, hi = -BOUND, BOUND

    def to_u(x):
        return (np.asarray(x) - lo) / (hi - lo)

    sobol = torch.quasirandom.SobolEngine(5, scramble=True, seed=seed).draw(5).numpy()
    initial = [START] + [(s - 0.5) * 1.2 * BOUND for s in sobol]   # the start, then 5 scrambled Sobol points within +-0.6 of the bound
    X, Y = [], []
    for x in initial:
        rec.check()
        j, eps, ekin = pb.value(x)
        rec.add(x, j, eps, ekin, "initial")
        X.append(to_u(x))
        Y.append(-j)
    while True:
        rec.check()
        tx = torch.tensor(np.array(X), dtype=torch.float64)
        ty = torch.tensor(np.array(Y), dtype=torch.float64).unsqueeze(-1)
        model = SingleTaskGP(tx, ty, outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        acq = LogExpectedImprovement(model, best_f=ty.max())
        cand, _ = optimize_acqf(acq, bounds=torch.tensor([[0.0] * 5, [1.0] * 5], dtype=torch.float64), q=1, num_restarts=8, raw_samples=256)
        u = cand[0].numpy()
        x = lo + u * (hi - lo)
        rec.check()
        j, eps, ekin = pb.value(x)
        rec.add(x, j, eps, ekin, "bayes")
        X.append(u)
        Y.append(-j)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["lbfgs_ad", "neldermead", "bayes"], required=True)
    p.add_argument("--budget-min", type=float, default=40.0)
    p.add_argument("--dt-s", type=float, default=1e-12)
    p.add_argument("--particles", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=float, nargs=5, default=None, help="starting point in natural units (default: the first comparison's)")
    p.add_argument("--image", choices=["off", "on"], default="off", help="cathode image in the space-charge solve")
    p.add_argument("--threads", type=int, default=5, help="torch threads: the paper runs three methods at once, five threads each, on a 16-core CPU")
    p.add_argument("--json", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)  # as in the paper's runs: the desktop stays usable, all three methods alike
    if a.start is not None:
        START[:] = a.start
    pb = Problem(a.dt_s, a.particles, a.image == "on")
    print(f"{a.method}: {pb.n_steps} steps, budget {a.budget_min} min, start {START.tolist()}", flush=True)
    rec = Recorder(a.budget_min * 60.0)
    try:
        {"lbfgs_ad": lambda: run_lbfgs(pb, rec), "neldermead": lambda: run_nelder_mead(pb, rec), "bayes": lambda: run_bayes(pb, rec, a.seed)}[a.method]()
        stopped = "converged"
    except Budget:
        stopped = "budget"
    out = {"method": a.method, "stopped": stopped, "budget_s": rec.budget, "seed": a.seed, "dt_s": a.dt_s, "particles": a.particles, "image": a.image, "image_steps": pb.image_steps, "n_steps": pb.n_steps,
           "start": START.tolist(), "knobs": NAMES, "bound": BOUND, "e_target_MeV": E_TARGET_MEV, "e_weight": E_WEIGHT, "trace": rec.trace,
           "wall_s": time.perf_counter() - rec.t0, "env": {"torch": torch.__version__}}
    a.json.write_text(json.dumps(out, indent=1))
    print(f"{a.method} {stopped}: best J {rec.best:.5f} in {len(rec.trace)} evaluations, {out['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()

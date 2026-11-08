"""Equal-wall-clock optics matching: autograd against finite differences and derivative-free search (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, item 30).

The knob-count ladder (fd_ad_knob_ladder.py) times Jacobians.  This control asks the practical question: with the same wall-clock budget, which
optimiser restores a perturbed ring's optics, and does the answer change with the number of knobs?  Every optimiser minimises the library's own
differentiable ``joint_optics_loss`` (tune and beta-function errors of the linear optics at the closed orbit, scales 1e-3 and 1%) of the 48 m ring
with random relative gradient errors (sigma 1%) on each of its 24 quadrupoles; the knobs are relative strength factors in [0.97, 1.03]
(5 families, 12 pairs, or all 24 quadrupoles).  The unperturbed ring is the target, so with 24 knobs the loss can reach zero.

    L-BFGS-B, autograd gradient     jac from ``torch.autograd`` (the objective's own backward pass)
    L-BFGS-B, central FD gradient   2 n evaluations per gradient, relative step 1e-6
    Powell, Nelder-Mead             scipy, unit-box search as in ``analysis.nonlinear_opt.gradient_free``

Each run has a wall-clock budget; every objective evaluation (including those inside an FD gradient) is timed and the best loss so far is traced.
One CPU thread, eager order-4 map (the map accuracy is irrelevant to the comparison, both sides use it), three error seeds per knob count.

    python optimisation_budget_control.py --budget 40 --seeds 3 --campaign 2026-10-29_opt_budget
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

KNOB_COUNTS = (5, 12, 24)
METHODS = ('lbfgs_ad', 'lbfgs_fd', 'powell', 'neldermead')
LO, HI = 0.97, 1.03   # the ring is linearly unstable at about +-5% on every quadrupole at once, so the box stays well inside it
PENALTY = 1e6
SIGMA = 0.01
TARGETS = (1e-2, 1e-4, 1e-8)   # loss levels for time-to-target


class Stop(Exception):
    pass


def setup():
    import torch
    from slipstream.analysis import nonlinear as nl
    from slipstream.analysis.nonlinear_opt import joint_optics_loss
    from slipstream.tracking.native.lattice_import import ElementSpec
    from slipstream.tracking.native.ring import RingMap

    torch.set_num_threads(1)
    case = common.load(HERE / 'cases/ring_tunes.json')
    lat = common.load(HERE / 'cases' / case['lattice'])
    specs, quads, count = [], [], {}
    for e in lat['elements']:
        name = e['name']
        if e['kind'] == 'quad':
            count[name] = count.get(name, 0) + 1
            name = f'{name}_{count[name]:02d}'
            quads.append(dict(name=name, family=e['name'], k1=e['k1']))
        specs.append(ElementSpec(name, e['kind'], e['length'], angle=e.get('angle', 0.0), k1=e.get('k1', 0.0), k2=e.get('k2', 0.0)))
    ring = RingMap(specs, energy_eV=case['energy_eV'], mc2_eV=case['mass_eV'], device='cpu', dtype=torch.float64, n_kicks=4, n_kicks_bend=8, integration_order=4)
    target = nl.linear_optics(ring)
    return torch, ring, quads, target, joint_optics_loss


def groups(quads, n):
    """Knob index of each quadrupole: 5 families, 12 neighbouring pairs, or one knob per quadrupole."""
    if n == 5:
        fams = sorted({q['family'] for q in quads})
        return [fams.index(q['family']) for q in quads]
    if n == 12:
        return [i // 2 for i in range(len(quads))]
    return list(range(len(quads)))


def run_one(ctx, n, seed, method, budget):
    torch, ring, quads, target, joint_optics_loss = ctx
    rng = np.random.default_rng(1000 + seed)
    k_err = np.array([q['k1'] for q in quads]) * (1 + SIGMA * rng.standard_normal(len(quads)))
    g = groups(quads, n)
    names = [q['name'] for q in quads]
    gi = np.array(g)

    def loss_t(theta):
        k1 = {nm: float(k_err[i]) * theta[gi[i]] for i, nm in enumerate(names)}
        return joint_optics_loss(ring.with_strengths(k1=k1), target)

    t_start = time.perf_counter()
    trace, best = [], [np.inf, None]
    n_eval = [0]
    unstable = [0]

    def record(x, val):
        n_eval[0] += 1
        if val < best[0]:
            best[:] = [val, np.array(x, float)]
        trace.append((time.perf_counter() - t_start, best[0]))

    def check():
        if time.perf_counter() - t_start > budget:
            raise Stop

    def f_only(x):
        check()
        try:
            with torch.no_grad():
                v = float(loss_t(torch.as_tensor(x, dtype=torch.float64)))
        except ValueError:   # linearly unstable ring: a large finite penalty, counted
            unstable[0] += 1
            v = PENALTY
        record(x, v)
        return v

    def f_grad(x):
        check()
        th = torch.as_tensor(x, dtype=torch.float64).clone().requires_grad_(True)
        try:
            v = loss_t(th)
            gr, = torch.autograd.grad(v, th)
        except ValueError:
            unstable[0] += 1
            record(x, PENALTY)
            return PENALTY, np.zeros(len(x))
        record(x, float(v))
        return float(v), gr.numpy().copy()

    def jac_fd(x, h=1e-6):
        out = np.zeros(len(x))
        for j in range(len(x)):
            up, dn = np.array(x, float), np.array(x, float)
            up[j] += h
            dn[j] -= h
            out[j] = (f_only(up) - f_only(dn)) / (2 * h)
        return out

    from scipy.optimize import minimize
    x0 = np.ones(n)
    status = 'budget'
    try:
        if method == 'lbfgs_ad':
            r = minimize(f_grad, x0, jac=True, method='L-BFGS-B', bounds=[(LO, HI)] * n, options=dict(maxiter=10**6, maxfun=10**6, ftol=1e-30, gtol=1e-14))
            status = 'converged'
        elif method == 'lbfgs_fd':
            r = minimize(f_only, x0, jac=jac_fd, method='L-BFGS-B', bounds=[(LO, HI)] * n, options=dict(maxiter=10**6, maxfun=10**6, ftol=1e-30, gtol=1e-14))
            status = 'converged'
        else:
            u0 = (x0 - LO) / (HI - LO)
            opt = dict(maxfev=10**6, maxiter=10**6)
            if method == 'neldermead':
                opt['initial_simplex'] = np.vstack([u0, u0 + np.diag(np.where(u0 < 0.5, 0.25, -0.25))])
                opt.update(xatol=1e-10, fatol=1e-30)
            else:
                opt['xtol'] = 1e-6
            r = minimize(lambda u: f_only(LO + (HI - LO) * np.asarray(u)), u0, method='Powell' if method == 'powell' else 'Nelder-Mead', bounds=[(0, 1)] * n, options=opt)
            status = 'converged'
    except Stop:
        pass
    wall = time.perf_counter() - t_start
    tr = np.array(trace)
    row = dict(n_knobs=n, seed=seed, method=method, status=status, wall_s=wall, evaluations=n_eval[0], unstable_evaluations=unstable[0], final_loss=float(best[0]),
               loss_start=float(tr[0, 1]) if len(tr) else None)
    for tgt in TARGETS:
        hit = np.nonzero(tr[:, 1] <= tgt)[0] if len(tr) else []
        row[f'time_to_{tgt:g}'] = float(tr[hit[0], 0]) if len(hit) else None
    keep = np.unique(np.linspace(0, len(tr) - 1, min(200, len(tr))).astype(int))
    row['trace'] = [[round(float(tr[i, 0]), 3), float(tr[i, 1])] for i in keep]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=40.0)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--campaign', default='2026-10-29_opt_budget')
    ap.add_argument('--knobs', type=int, nargs='*', default=list(KNOB_COUNTS))
    ap.add_argument('--out')
    a = ap.parse_args()
    ctx = setup()
    # warm: one evaluation of each path so imports are not charged to the first run
    run_one(ctx, 5, 99, 'lbfgs_ad', 3.0)
    rows = []
    for n in a.knobs:
        for seed in range(a.seeds):
            for method in METHODS:
                r = run_one(ctx, n, seed, method, a.budget)
                rows.append(r)
                print(f"n={n:2d} seed {seed} {method:11s} {r['status']:9s} loss {r['loss_start']:.3g} -> {r['final_loss']:.3g}  evals {r['evaluations']:5d}  wall {r['wall_s']:.1f}  t(1e-4) {r['time_to_0.0001']}", flush=True)
    out = dict(design=dict(date='2026-10-29', budget_s=a.budget, seeds=a.seeds, sigma=SIGMA, bounds=[LO, HI], knob_counts=a.knobs, methods=METHODS,
                           objective='analysis.nonlinear_opt.joint_optics_loss (tune scale 1e-3, beta scale 1%) of the 48 m ring; target = unperturbed ring', threads=1),
               analysis_sha256=common.sha256(__file__), rows=rows)
    path = Path(a.out) if a.out else common.EVIDENCE / '2026-10-26_R1/optimisation_budget_control.json'
    path.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()

"""Time to converge: optics matching run to a loss target, not to a wall-clock budget (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, item 33).

Item 30 gave every optimiser 40 s.  Here each optimiser runs until the loss of ``analysis.nonlinear_opt.joint_optics_loss`` (same problem as item 30: 48 m ring,
24 quadrupoles with 1% random gradient errors, knobs = relative strength factors in [0.97, 1.03], unperturbed ring = target) falls below ``--target`` (1e-9,
absolute; the starting loss is 650 to 8800) or an overall wall-clock cap is reached.  A run that terminates by its own tolerances without reaching the
target is restarted from its best point (up to 10 times; stopped as 'stalled' if a restart gains less than 10%).  Knob sets are contiguous blocks of
neighbouring quadrupoles (``--knobs`` counts; 12 and 24 are the pairs and the single quadrupoles of item 30).

Stages (each worker is one single-thread process pinned to its own physical core, up to ``--parallel`` at a time):

    --stage run       all four optimisers on every case (family A: the 24-knob problem of item 30, three seeds; family B: errors shared by blocks of quadrupoles,
                      so that an exact solution exists for 4, 6, 8 and 12 knobs as well)
    --stage analyse   merge the task files into data/2026-10-26_R1/optimisation_convergence_control.json

    A first probe (autograd only, family A independent errors, 4/6/8/12 knobs, bounds +-3%) is kept in <campaign>/probe: the target is unreachable
    except with one knob per quadrupole (and for one of three seeds with 12 knobs), which is why family B exists.

    python optimisation_convergence_control.py --stage run --campaign 2026-10-29_opt_converge --parallel 12
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

LO, HI = 0.97, 1.03
SIGMA = 0.01
PENALTY = 1e6
METHODS = ('lbfgs_ad', 'lbfgs_fd', 'powell', 'neldermead')
MAX_RESTARTS = 10
LO_HI_MARGIN = 0.0299   # the needed correction must lie inside the bounds


class Stop(Exception):
    pass


def groups(n, nq=24):
    """Contiguous blocks of neighbouring quadrupoles (n = 12: pairs, n = 24: single quadrupoles, as in item 30)."""
    return [i * n // nq for i in range(nq)]


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
            quads.append(dict(name=name, k1=e['k1']))
        specs.append(ElementSpec(name, e['kind'], e['length'], angle=e.get('angle', 0.0), k1=e.get('k1', 0.0), k2=e.get('k2', 0.0)))
    ring = RingMap(specs, energy_eV=case['energy_eV'], mc2_eV=case['mass_eV'], device='cpu', dtype=torch.float64, n_kicks=4, n_kicks_bend=8, integration_order=4)
    return torch, ring, quads, nl.linear_optics(ring), joint_optics_loss


def problem(family, n, seed, k0):
    """Perturbed quadrupole strengths and the knob block of each quadrupole.
    A: independent 1% errors on all 24 quadrupoles (item 30; an exact solution needs one knob per quadrupole).
    B: one 1% error per knob block, shared by the quadrupoles of the block, so an exact solution exists for every knob count."""
    rng = np.random.default_rng(1000 + seed)
    gi = np.array(groups(n))
    if family == 'A':
        return k0 * (1 + SIGMA * rng.standard_normal(len(k0))), gi
    eps = SIGMA * rng.standard_normal(n)
    assert np.abs(1 / (1 + eps) - 1).max() < LO_HI_MARGIN
    return k0 * (1 + eps[gi]), gi


def run_task(family, n, seed, method, target, cap):
    from scipy.optimize import minimize
    torch, ring, quads, tgt, jol = setup()
    k_err, gi = problem(family, n, seed, np.array([q['k1'] for q in quads]))
    names = [q['name'] for q in quads]

    def loss_t(theta):
        return jol(ring.with_strengths(k1={nm: float(k_err[i]) * theta[gi[i]] for i, nm in enumerate(names)}), tgt)

    t0 = time.perf_counter()
    best = [np.inf, np.ones(n)]
    counts = dict(evals=0, grads=0, unstable=0)
    events = []            # (elapsed, evaluations, best) at every improvement
    last = {}

    def record(x, v):
        counts['evals'] += 1
        if v < best[0]:
            best[:] = [v, np.array(x, float)]
            events.append((time.perf_counter() - t0, counts['evals'], v))
            if v <= target:
                raise Stop('target')

    def check():
        if time.perf_counter() - t0 > cap:
            raise Stop('cap')

    def f_only(x):
        check()
        try:
            with torch.no_grad():
                v = float(loss_t(torch.as_tensor(x, dtype=torch.float64)))
        except ValueError:
            counts['unstable'] += 1
            v = PENALTY
        last['x'], last['v'] = np.array(x, float), v
        record(x, v)
        return v

    def f_grad(x):
        check()
        th = torch.as_tensor(x, dtype=torch.float64).clone().requires_grad_(True)
        counts['grads'] += 1
        try:
            v = loss_t(th)
            g, = torch.autograd.grad(v, th)
        except ValueError:
            counts['unstable'] += 1
            record(x, PENALTY)
            return PENALTY, np.zeros(len(x))
        record(x, float(v.detach()))
        return float(v.detach()), g.numpy().copy()

    def jac_fd(x, h=1e-6):
        counts['grads'] += 1
        if 'x' in last and np.array_equal(last['x'], x) and last['v'] >= PENALTY:   # no gradient at an unstable trial point
            return np.zeros(len(x))
        out = np.zeros(len(x))
        for j in range(len(x)):
            up, dn = np.array(x, float), np.array(x, float)
            up[j] += h
            dn[j] -= h
            out[j] = (f_only(up) - f_only(dn)) / (2 * h)
        return out

    def one_pass(x0):
        if method == 'lbfgs_ad':
            minimize(f_grad, x0, jac=True, method='L-BFGS-B', bounds=[(LO, HI)] * n, options=dict(maxiter=10**6, maxfun=10**6, ftol=1e-30, gtol=1e-14))
        elif method == 'lbfgs_fd':
            minimize(f_only, x0, jac=jac_fd, method='L-BFGS-B', bounds=[(LO, HI)] * n, options=dict(maxiter=10**6, maxfun=10**6, ftol=1e-30, gtol=1e-14))
        else:
            u0 = (x0 - LO) / (HI - LO)
            opt = dict(maxfev=10**7, maxiter=10**7)
            if method == 'neldermead':
                step = np.where(u0 < 0.5, 1.0, -1.0) * 0.25 * 0.3 ** state['r']   # the item 30 simplex first, smaller after each restart
                opt['initial_simplex'] = np.vstack([u0, u0 + np.diag(step)])
                opt.update(xatol=1e-13, fatol=1e-30)
                meth = 'Nelder-Mead'
            else:
                opt.update(xtol=1e-12, ftol=1e-15)
                meth = 'Powell'
            minimize(lambda u: f_only(LO + (HI - LO) * np.asarray(u)), u0, method=meth, bounds=[(0, 1)] * n, options=opt)

    state = dict(r=0)
    status, restarts, prev = 'cap', 0, None
    try:
        x0 = np.ones(n)
        while True:
            one_pass(x0)                      # returns only by its own tolerances; Stop ends the run
            restarts += 1
            state['r'] = restarts
            if restarts > MAX_RESTARTS or (prev is not None and best[0] > 0.9 * prev):
                status = 'stalled'
                break
            prev, x0 = best[0], best[1].copy()
    except Stop as e:
        status = 'reached' if str(e) == 'target' else 'cap'
    wall = time.perf_counter() - t0
    ev = np.array(events)
    keep = np.unique(np.linspace(0, len(ev) - 1, min(1500, len(ev))).astype(int)) if len(ev) else []
    return dict(family=family, n_knobs=n, seed=seed, method=method, target=target, cap_s=cap, status=status, reached=status == 'reached', wall_s=wall,
                time_to_target_s=wall if status == 'reached' else None, evaluations=counts['evals'], gradient_calls=counts['grads'], unstable_evaluations=counts['unstable'],
                restarts=restarts, final_loss=float(best[0]), loss_start=float(ev[0, 2]) if len(ev) else None, trace=[[round(float(ev[i, 0]), 4), int(ev[i, 1]), float(ev[i, 2])] for i in keep])


def launch(tasks, root, args):
    """Run (n, seed, method) tasks, each in its own pinned single-thread process, at most ``args.parallel`` at a time."""
    root.mkdir(parents=True, exist_ok=True)
    py = common.SLIPSTREAM_SRC
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(py))
    pending = [t for t in tasks if not (root / f'{t[0]}_n{t[1]}_s{t[2]}_{t[3]}.json').exists()]
    running, free = [], list(range(args.parallel))
    while pending or running:
        for r in list(running):
            if r[0].poll() is not None:
                running.remove(r)
                free.append(r[1])
                print('done', r[2], 'rc', r[0].returncode, flush=True)
        while pending and free:
            family, n, seed, method = pending.pop(0)
            slot = free.pop(0)
            cap = args.cap
            cmd = [sys.executable, str(Path(__file__).resolve()), '--task', f'{family},{n},{seed},{method}', '--cpu', str(2 * (slot + 2)), '--target', str(args.target), '--cap', str(cap),
                   '--out', str(root / f'{family}_n{n}_s{seed}_{method}.json')]
            running.append((subprocess.Popen(cmd, env=env), slot, (family, n, seed, method)))
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['run', 'analyse'])
    ap.add_argument('--campaign', default='2026-10-29_opt_converge')
    ap.add_argument('--knobs-b', type=int, nargs='*', default=[4, 6, 8, 12], help='family B knob counts (family A is the 24-knob problem of item 30)')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--target', type=float, default=1e-9)
    ap.add_argument('--cap', type=float, default=1800.0, help='wall-clock cap [s] per run')
    ap.add_argument('--parallel', type=int, default=8)
    ap.add_argument('--task')
    ap.add_argument('--cpu', type=int)
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.task:
        import psutil
        psutil.Process().cpu_affinity([a.cpu])
        family, n, seed, method = a.task.split(',')
        res = run_task(family, int(n), int(seed), method, a.target, a.cap)
        res['cpu'] = a.cpu
        Path(a.out).write_text(json.dumps(res))
        print(family, method, n, seed, res['status'], res['wall_s'])
        return
    root = common.RESULTS / a.campaign
    cases = [('A', 24, s) for s in range(a.seeds)] + [('B', n, s) for n in a.knobs_b for s in range(a.seeds)]
    if a.stage == 'run':
        # long (derivative-free) tasks first so they overlap with the short ones
        order = {'powell': 0, 'neldermead': 1, 'lbfgs_fd': 2, 'lbfgs_ad': 3}
        launch(sorted([(f, n, s, m) for f, n, s in cases for m in METHODS], key=lambda t: (-t[1], order[t[3]], t[2])), root, a)
    else:
        rows = [json.loads(p.read_text()) for p in sorted(root.glob('[AB]_n*_s*_*.json'))]
        probe = [json.loads(p.read_text()) for p in sorted((root / 'probe').glob('n*_s*_lbfgs_ad.json'))]
        for r in probe:
            r.pop('trace', None)
        out = dict(design=dict(date='2026-10-29', target=a.target, cap_s=a.cap, bounds=[LO, HI], sigma=SIGMA, restarts_max=MAX_RESTARTS, knob_counts_b=a.knobs_b, seeds=a.seeds, methods=METHODS,
                               parallel=a.parallel, note='single-thread workers pinned to distinct physical cores, up to --parallel concurrent; eager order-4 map; family A = item 30 problem (24 knobs), family B = block-matched errors (exact solution for every knob count)'),
                   analysis_sha256=common.sha256(__file__), rows=rows, probe_family_a=probe)
        (common.EVIDENCE / '2026-10-26_R1/optimisation_convergence_control.json').write_text(json.dumps(out) + chr(10))
        for r in rows:
            print(f"{r['family']} n={r['n_knobs']:2d} s{r['seed']} {r['method']:11s} {r['status']:8s} wall {r['wall_s']:7.1f} evals {r['evaluations']:7d} grads {r['gradient_calls']:5d} restarts {r['restarts']} final {r['final_loss']:.2e}")


if __name__ == '__main__':
    main()

"""Measured finite-difference versus automatic-differentiation cost as the number of knobs grows (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, item 27).

Item 17 timed the five-knob endpoint Jacobian and *projected* the break-even knob count assuming AD costs the same for any number of knobs.
Here that assumption is tested.  Every quadrupole (24) and nonzero sextupole (16) of the 48 m ring gets its own name, so each is its own
knob (40 knobs).  For n = 1, 2, 3, 5, 10, 20, 40 knobs spread evenly along the ring, one process times, on the same machine state and one
CPU thread (compiled regional CPU preset, warm):

    forward     one endpoint evaluation (39 turns; no graph)
    fd1         central difference at one relative step 1e-5            (2 n forward evaluations)
    fd2         Richardson pair from central differences at 2e-4, 1e-4  (4 n forward evaluations; fourth-order)
    ad          torch.autograd.functional.jacobian of the same endpoint  (the adapter's AD path)

and compares every Jacobian with AD in the dimensionless units of gradient_acceptance.py (S_ij = theta_j dy_i/dtheta_j / sigma_i; limits declared
in that file).  A second block takes eight zero-nominal knobs (horizontal quadrupole offsets, scale 100 um) and compares AD with central
differences at an absolute-step ladder, because a knob with zero nominal value has no multiplicative scale (open item of item 24).

    python fd_ad_knob_ladder.py --campaign 2026-10-29_knob_ladder
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
sys.path.insert(0, str(HERE / 'workers'))
import common  # noqa: E402

CONFIG = 'cpu-f64-o6q8s1b8-compiled'
LADDER = [1, 2, 3, 5, 10, 20, 40]
H1 = 1e-5
H2 = (2e-4, 1e-4)
ALIGN_SCALE = 1e-4      # m, declared scale of a quadrupole offset knob
ALIGN_STEPS = (1e-1, 1e-2, 1e-3, 1e-4)   # relative to ALIGN_SCALE


def unique_lattice(lat):
    """Rename every occurrence of a quadrupole or nonzero sextupole to a unique name; returns (lattice, knob list in ring order)."""
    import copy
    out, knobs, count = copy.deepcopy(lat), [], {}
    for e in out['elements']:
        kind = 'k1' if e['kind'] == 'quad' else 'k2' if e['kind'] == 'sext' else None
        if kind and e.get(kind, 0.0) != 0.0:
            count[e['name']] = count.get(e['name'], 0) + 1
            e['name'] = f"{e['name']}_{count[e['name']]:02d}"
            knobs.append(dict(family=e['name'], kind=kind, value=e[kind]))
    return out, knobs


def worker(out):
    import torch
    import w_slipstream as w
    from gradient_acceptance import LIMITS, metrics, native_sigma, scaled

    cfg = next(c for c in w.CONFIGS if c['name'] == CONFIG)
    case = common.load(HERE / 'cases/ring_grad.json')
    lat0 = common.load(HERE / 'cases' / case['lattice'])
    lat, allknobs = unique_lattice(lat0)
    turns = case['turns']
    init = common.launches(case)[:1]
    w.setup()
    ring, _ = w.build(lat, case, cfg)
    dtype, dev = ring.dtype, ring.device
    s0 = torch.as_tensor(init, dtype=dtype, device=dev)
    sigma = native_sigma(case)
    res = dict(config=CONFIG, threads=torch.get_num_threads(), turns_endpoint=turns - 1, n_knobs_available=len(allknobs), rows=[])

    def peak_wset():
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 2**20
        except Exception:  # noqa: BLE001
            return None

    def endpoint(theta, names, kinds, extra=None, grad=False):
        k = {'k1': {}, 'k2': {}}
        for j, (nm, kd) in enumerate(zip(names, kinds)):
            k[kd][nm] = theta[j]
        r = ring.with_strengths(k1=k['k1'], k2=k['k2'], **(extra(theta) if extra else {}))
        s = s0
        with torch.set_grad_enabled(grad):
            for _ in range(turns - 1):
                s = r.turn(s)
        return s[0, :4]

    def median_time(fn, n):
        fn()  # warm: compiled regions and caches
        t = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = fn()
            t.append(time.perf_counter() - t0)
        return float(np.median(t)), r

    for n in LADDER:
        idx = np.unique(np.round(np.linspace(0, len(allknobs) - 1, n)).astype(int))
        sel = [allknobs[i] for i in idx]
        names, kinds = [s['family'] for s in sel], [s['kind'] for s in sel]
        k0 = np.array([s['value'] for s in sel])
        theta0 = torch.as_tensor(k0, dtype=dtype, device=dev)
        reps = 3 if n <= 5 else 2

        def f(theta):
            return endpoint(theta, names, kinds)

        def central(h):
            J = np.zeros((4, n))
            for j in range(n):
                up, dn = np.ones(n), np.ones(n)
                up[j] += h
                dn[j] -= h
                yu = f(theta0 * torch.as_tensor(up, dtype=dtype, device=dev)).double().cpu().numpy()
                yd = f(theta0 * torch.as_tensor(dn, dtype=dtype, device=dev)).double().cpu().numpy()
                J[:, j] = (yu - yd) / (2 * h * k0[j])
            return J

        def fd2():
            Ja, Jb = central(H2[0]), central(H2[1])     # h and h/2: R = (4 D(h/2) - D(h)) / 3
            return (4 * Jb - Ja) / 3

        def f_graph(theta):
            return endpoint(theta, names, kinds, grad=True)

        def ad():
            return torch.autograd.functional.jacobian(f_graph, theta0).detach().double().cpu().numpy()

        row = dict(n_knobs=n, knobs=names[:3] + (['...'] if n > 3 else []))
        row['forward_s'], _ = median_time(lambda: f(theta0), 3)
        row['fd1_s'], J1 = median_time(lambda: central(H1), reps)
        row['fd2_s'], J2 = median_time(fd2, reps)
        row['peak_wset_before_ad_mb'] = peak_wset()
        row['ad_s'], Jad = median_time(ad, reps)
        row['peak_wset_after_ad_mb'] = peak_wset()
        S = {k: scaled(J, k0, sigma) for k, J in (('fd1', J1), ('fd2', J2), ('ad', Jad))}
        for key in ('fd1', 'fd2'):
            m = metrics(S[key], S['ad'])
            row[f'{key}_vs_ad_e_abs'], row[f'{key}_vs_ad_e_rel'] = m['e_abs'], m['e_rel']
        row['fd1_vs_fd2_e_abs'] = metrics(S['fd1'], S['fd2'])['e_abs']
        row['evals_fd1'], row['evals_fd2'] = 2 * n, 4 * n
        row['fd1_over_ad'], row['fd2_over_ad'] = row['fd1_s'] / row['ad_s'], row['fd2_s'] / row['ad_s']
        row['ad_over_forward'] = row['ad_s'] / row['forward_s']
        res['rows'].append(row)
        print(f"n={n:3d} fwd {row['forward_s']:.3f} fd1 {row['fd1_s']:.2f} fd2 {row['fd2_s']:.2f} ad {row['ad_s']:.2f} "
              f"e_abs fd1 {row['fd1_vs_ad_e_abs']:.1e} fd2 {row['fd2_vs_ad_e_abs']:.1e}", flush=True)

    # zero-nominal knobs: horizontal offsets of eight quadrupoles
    quads = [s for s in allknobs if s['kind'] == 'k1']
    qs = [quads[i] for i in np.unique(np.round(np.linspace(0, len(quads) - 1, 8)).astype(int))]
    qn = [s['family'] for s in qs]
    nq = len(qn)

    def fa(theta):
        k = {n_: theta[j] for j, n_ in enumerate(qn)}
        r = ring.with_strengths(dx=k)
        s = s0
        with torch.set_grad_enabled(theta.requires_grad):
            for _ in range(turns - 1):
                s = r.turn(s)
        return s[0, :4]

    zero = torch.zeros(nq, dtype=dtype, device=dev)
    # a launch exactly on the closed orbit has zero response to nothing, so use the case launch; offsets are small perturbations
    Jad = torch.autograd.functional.jacobian(fa, zero).detach().double().cpu().numpy()
    kscale = np.full(nq, ALIGN_SCALE)
    Sad = scaled(Jad, kscale, sigma)
    align = dict(knobs=qn, scale_m=ALIGN_SCALE, rows=[])
    for rel in ALIGN_STEPS:
        h = rel * ALIGN_SCALE
        J = np.zeros((4, nq))
        for j in range(nq):
            e = torch.zeros(nq, dtype=dtype, device=dev)
            e[j] = h
            J[:, j] = (fa(e) - fa(-e)).double().cpu().numpy() / (2 * h)
        S = scaled(J, kscale, sigma)
        m = metrics(S, Sad)
        align['rows'].append(dict(rel_step=rel, step_m=h, e_abs_vs_ad=m['e_abs'], e_rel_vs_ad=m['e_rel'], n_below_floor=m['n_below_floor']))
    align['max_abs_S'] = float(np.abs(Sad).max())
    align['limits'] = LIMITS
    res['alignment'] = align
    (out / 'result.json').write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign')
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.worker:
        worker(Path(a.out))
        return
    root = common.RESULTS / a.campaign
    py = common.SLIPSTREAM_SRC
    for k in range(a.procs):
        out = root / f'p{k}'
        if (out / 'result.json').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', **common.cxx_env(), PYTHONPATH=str(py))
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, __file__, '--worker', '--out', str(out)], env=env, capture_output=True, text=True)
        print(f'p{k} rc {r.returncode} {time.perf_counter() - t0:.0f} s', flush=True)
        if r.returncode:
            print(r.stderr[-2500:])
        else:
            print(r.stdout)
    rs = [json.loads(p.read_text()) for p in sorted(root.glob('p*/result.json'))]
    rows = []
    for i, n in enumerate(LADDER):
        rr = [r['rows'][i] for r in rs]
        row = dict(n_knobs=n, n_ok=len(rr))
        for key in rr[0]:
            if key.endswith('_s') or key.endswith('_over_ad') or key.endswith('_over_forward') or key.startswith('peak'):
                vals = [x[key] for x in rr if x.get(key) is not None]
                row[key] = float(np.median(vals)) if vals else None
                row[key + '_range'] = [min(vals), max(vals)] if vals else None
            elif key.endswith('_e_abs') or key.endswith('_e_rel'):
                row[key] = max(x[key] for x in rr)
        for key in ('evals_fd1', 'evals_fd2'):
            row[key] = rr[0][key]
        rows.append(row)
    align = dict(knobs=rs[0]['alignment']['knobs'], scale_m=ALIGN_SCALE, max_abs_S=max(r['alignment']['max_abs_S'] for r in rs),
                 rows=[dict(rel_step=a_['rel_step'], step_m=a_['step_m'],
                            e_abs_vs_ad=max(r['alignment']['rows'][j]['e_abs_vs_ad'] for r in rs),
                            e_rel_vs_ad=max(r['alignment']['rows'][j]['e_rel_vs_ad'] for r in rs),
                            n_below_floor=rs[0]['alignment']['rows'][j]['n_below_floor']) for j, a_ in enumerate(rs[0]['alignment']['rows'])])
    # break-even: smallest ladder n where AD is faster than each FD procedure (measured), and the model-based interpolation
    be = {}
    for key in ('fd1', 'fd2'):
        ns = [r['n_knobs'] for r in rows if r[f'{key}_s'] > r['ad_s']]
        be[key] = ns[0] if ns else None
    out = dict(design=dict(date='2026-11-07', procs=len(rs), config=CONFIG, ladder=LADDER, fd1_step=H1, fd2_steps=H2,
                           note='Measured knob-count ladder: one CPU thread, compiled regional CPU preset, medians of warm calls; Jacobians compared in the dimensionless units of gradient_acceptance.py'),
               analysis_sha256=common.sha256(__file__), rows=rows, alignment=align, first_ladder_n_where_ad_faster=be)
    (common.EVIDENCE / '2026-10-26_R1/fd_ad_knob_ladder.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()

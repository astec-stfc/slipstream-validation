"""Equal-output timing of Slipstream finite-difference and automatic-differentiation Jacobians (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P1).

The ring_grad adapter's finite-difference path rebuilds the model for every perturbation and tracks 40 turns recording every turn
(one turn unused), whereas its AD path reuses one model and runs 39 turns with ``with_strengths``.  This control times the same
Jacobian output (state at the start of turn 39, five knobs) three ways in one process, each on the identical machine state:

    fd_adapter   one central-difference Jacobian at one step, exactly as the adapter does it (10 rebuilds + 40-turn recorded tracks)
    fd_endpoint  the same 10 evaluations through ``with_strengths`` and 39 ``turn`` calls, no rebuild, no recording
    ad           ``torch.autograd.functional.jacobian`` of that endpoint function (the adapter's AD path)
    forward      one endpoint evaluation (no autograd); ``forward_graph`` the same with the autograd graph recorded

and checks the Jacobians agree.  Three independent worker processes per configuration, medians of warm calls.

    python fd_ad_endpoint_control.py --campaign 2026-10-27_fd_ad_endpoint
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'workers'))
import common  # noqa: E402

CONFIGS = {'cpu-f64-o6q8s1b8': dict(skip_ad=True, repeats=1), 'cpu-f64-o6q8s1b8-compiled': dict(skip_ad=False, repeats=3)}
H = 1e-3   # one relative step, the middle of the case ladder


def worker(config, out):
    import torch
    import w_slipstream as w

    cfg = next(c for c in w.CONFIGS if c['name'] == config)
    case = common.load(HERE / 'cases/ring_grad.json')
    lat = common.load(HERE / 'cases' / case['lattice'])
    knobs, turns = case['knobs'], case['turns']
    init = common.launches(case)[:1]
    k0 = common.nominal_strengths(lat, knobs)
    spec = CONFIGS[config]
    w.setup()
    ring, _ = w.build(lat, case, cfg)
    dtype, dev = ring.dtype, ring.device
    s0 = torch.as_tensor(init, dtype=dtype, device=dev)
    # warm the compiled regions once (cold cost is not this control's question)
    am = w.ad_build(lat, case, cfg, knobs)

    def endpoint(theta, grad=False):
        k = {'k1': {}, 'k2': {}}
        for j, kn in enumerate(knobs):
            k[kn['kind']][kn['family']] = theta[j]
        r = ring.with_strengths(k1=k['k1'], k2=k['k2'])
        s = s0
        with torch.set_grad_enabled(grad):
            for _ in range(turns - 1):
                s = r.turn(s)
        return s[0, :4]

    def median_time(fn, n):
        fn()
        t = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            t.append(time.perf_counter() - t0)
        return float(np.median(t))

    theta0 = torch.as_tensor(k0, dtype=dtype, device=dev)

    def fd_endpoint():
        J = np.zeros((4, len(k0)))
        for j in range(len(k0)):
            up, dn = np.ones(len(k0)), np.ones(len(k0))
            up[j] += H
            dn[j] -= H
            yu = endpoint(theta0 * torch.as_tensor(up, dtype=dtype, device=dev)).double().cpu().numpy()
            yd = endpoint(theta0 * torch.as_tensor(dn, dtype=dtype, device=dev)).double().cpu().numpy()
            J[:, j] = (yu - yd) / (2 * H * k0[j])
        return J

    def fd_adapter():
        J = np.zeros((4, len(k0)))
        for j in range(len(k0)):
            up, dn = np.ones(len(k0)), np.ones(len(k0))
            up[j] += H
            dn[j] -= H
            ys = []
            for f in (up, dn):
                m, _ = w.build(common.scaled_lattice(lat, knobs, f), case, cfg)
                ys.append(w.track(m, init, turns, cfg)[0, -1, :4])
            J[:, j] = (ys[0] - ys[1]) / (2 * H * k0[j])
        return J

    rep = spec['repeats']
    res = dict(config=config, repeats=rep, threads=torch.get_num_threads())
    res['fd_adapter_s'] = median_time(fd_adapter, rep)
    res['fd_endpoint_s'] = median_time(fd_endpoint, rep)
    res['forward_s'] = median_time(lambda: endpoint(theta0), max(rep, 3))
    th = theta0.clone().requires_grad_(True)
    res['forward_graph_s'] = median_time(lambda: endpoint(th, grad=True), max(rep, 3))
    J_fd = fd_endpoint()
    if not spec['skip_ad']:
        res['ad_s'] = median_time(lambda: w.ad_jacobian(am, init, turns, cfg, knobs), rep)
        _, J_ad = w.ad_jacobian(am, init, turns, cfg, knobs)
        res['ad_vs_fd_endpoint_fro'] = float(np.linalg.norm(J_ad - J_fd) / np.linalg.norm(J_fd))
    res['fd_adapter_vs_endpoint_fro'] = float(np.linalg.norm(fd_adapter() - J_fd) / np.linalg.norm(J_fd))
    res['turns_adapter'] = turns
    res['turns_endpoint'] = turns - 1
    (out / 'result.json').write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign')
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--worker')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.worker:
        worker(a.worker, Path(a.out))
        return
    root = common.RESULTS / a.campaign
    py = common.SLIPSTREAM_SRC
    for config, spec in CONFIGS.items():
        for k in range(1 if config == 'cpu-f64-o6q8s1b8' else a.procs):
            out = root / config / f'p{k}'
            if (out / 'result.json').exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            import os
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', **common.cxx_env(), PYTHONPATH=str(py))
            t0 = time.perf_counter()
            r = subprocess.run([sys.executable, __file__, '--worker', config, '--out', str(out)], env=env, capture_output=True, text=True)
            print(config, f'p{k}', 'rc', r.returncode, f'{time.perf_counter() - t0:.0f} s', flush=True)
            if r.returncode:
                print(r.stderr[-1500:])
    rows = []
    for config in CONFIGS:
        rs = [json.loads(p.read_text()) for p in sorted((root / config).glob('p*/result.json'))]
        if not rs:
            continue
        row = dict(config=config, n_ok=len(rs))
        for key in rs[0]:
            if key.endswith('_s'):
                row[key] = float(np.median([r[key] for r in rs if key in r]))
        for key in ('ad_vs_fd_endpoint_fro', 'fd_adapter_vs_endpoint_fro'):
            vals = [r[key] for r in rs if key in r]
            if vals:
                row[key] = max(vals)
        rows.append(row)
    # accuracy of one central difference against the best Richardson FD and AD, from the saved primary-campaign Jacobians
    from analyse_grad import best_fd
    steps_err = {}
    for z in (np.load(p) for p in sorted((common.RESULTS / '2026-10-26_R1/ring_grad/slipstream/cpu-f64-o6q8s1b8').glob('p*/grad.npz'))):
        best, label = best_fd(z['J_fd'], z['steps'])
        for h, J in zip(z['steps'], z['J_fd']):
            steps_err[str(h)] = max(steps_err.get(str(h), 0.0), float(np.linalg.norm(J - best) / np.linalg.norm(best)))
        steps_err['ad'] = max(steps_err.get('ad', 0.0), float(np.linalg.norm(z['J_ad'] - best) / np.linalg.norm(best)))
    for row in rows:
        row['single_step_fd_error_vs_best'] = steps_err
        break
    out = dict(design=dict(date='2026-11-07', procs=a.procs, step=H, note='Same Jacobian output timed through the adapter FD path, an endpoint-only FD path and AD; one CPU thread; medians of warm calls'),
               analysis_sha256=common.sha256(__file__), rows=rows)
    (common.EVIDENCE / '2026-10-26_R1/fd_ad_endpoint_control.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()

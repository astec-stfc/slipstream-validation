"""Combine the same-session GPU scaling runs into data/2026-10-26_R1/xsuite_gpu_scaling.json (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, item 29).

Inputs: results/2026-10-29_xsuite_gpu/p*.json (xsuite_gpu_scaling.py), results/2026-10-29_scaling_gpu (run.py: Slipstream GPU float32 and float64 on the
ring_scale_n* cases) and, for the one-thread CPU context, the stored particle_scaling_control.json (27 October).  Final states of the first five
particles are compared with the stored elegant run at the same N (separate x, px, y, py residuals; positions and momenta are never mixed).
"""
import json
from pathlib import Path

import numpy as np

import common

HERE = Path(__file__).resolve().parent
NS = (5, 50, 500, 5000, 50000)
GPU = common.RESULTS / '2026-10-29_scaling_gpu'
XS = common.RESULTS / '2026-10-29_xsuite_gpu'
OLD = common.RESULTS / '2026-10-27_scaling'


def slip(config, n):
    rs = [common.load(p) for p in sorted((GPU / f'ring_scale_n{n}' / 'slipstream' / config).glob('p*/result.json'))]
    ok = [r for r in rs if r['status'] == 'ok']
    if not ok:
        return None
    state = None
    p0 = GPU / f'ring_scale_n{n}' / 'slipstream' / config / 'p0' / 'tbt.npz'
    if p0.exists():
        state = np.load(p0)['tbt'][:5, -1, :4]
    return dict(n_ok=len(ok), warm_s=float(np.median([r['timing']['warm_s']['median'] for r in ok])), warm_range_s=[min(r['timing']['warm_s']['median'] for r in ok), max(r['timing']['warm_s']['median'] for r in ok)],
                cold_s=float(np.median([r['timing']['cold_s'] for r in ok])), state=state)


def elegant_state(n):
    p = OLD / f'ring_scale_n{n}' / 'elegant' / 'cpu-f64-o6s8b16' / 'p0' / 'tbt.npz'
    return np.load(p)['tbt'][:5, -1, :4] if p.exists() else None


def diff(a, b):
    return None if a is None or b is None else dict(zip(('x_m', 'px', 'y_m', 'py'), np.abs(np.asarray(a) - np.asarray(b)).max(axis=0).tolist()))


def main():
    xs = [json.loads(p.read_text()) for p in sorted(XS.glob('p*.json'))]
    rows = []
    for n in NS:
        ref = elegant_state(n)
        row = dict(n=n)
        for label, cfg in (('slipstream_gpu_f32', 'cuda-f32-o6q8s1b8'), ('slipstream_gpu_f64', 'cuda-f64-o6q8s1b8')):
            r = slip(cfg, n)
            if r:
                st = r.pop('state')
                row[label] = dict(r, per_particle_turn_us=1e6 * r['warm_s'] / (n * 64), first5_vs_elegant=diff(st, ref))
        got = [next((x for x in p['rows'] if x.get('n') == n and 'warm_median_s' in x), None) for p in xs]
        got = [g for g in got if g]
        if got:
            w = float(np.median([g['warm_median_s'] for g in got]))
            row['xsuite_gpu_f64'] = dict(n_ok=len(got), warm_s=w, warm_range_s=[min(g['warm_median_s'] for g in got), max(g['warm_median_s'] for g in got)], per_particle_turn_us=1e6 * w / (n * 64),
                                         cold_s=float(np.median([g['cold_s'] for g in got])), first5_vs_elegant=diff(got[0]['first5_final_state'], ref))
        rows.append(row)
    old = json.loads((common.EVIDENCE / '2026-10-26_R1/particle_scaling_control.json').read_text())['rows']
    cpu = {(r['code'], r['config'], r['n']): r for r in old if r.get('n_ok')}
    for row in rows:
        for label, key in (('xsuite_cpu_exact_1thread', ('xsuite', 'cpu-f64-yosk8b16-exact')), ('slipstream_cpu_compiled_1thread', ('slipstream', 'cpu-f64-o6q8s1b8-compiled')), ('elegant_1thread', ('elegant', 'cpu-f64-o6s8b16'))):
            r = cpu.get((*key, row['n']))
            if r:
                row[label] = dict(warm_s=r['warm_s'], per_particle_turn_us=r['per_particle_turn_us'], source='particle_scaling_control.json (27 October)')
        if 'xsuite_gpu_f64' in row:
            for lab in ('slipstream_gpu_f32', 'slipstream_gpu_f64'):
                if lab in row:
                    row[lab]['speedup_over_xsuite_gpu'] = row['xsuite_gpu_f64']['warm_s'] / row[lab]['warm_s']
    errs = [x['rows'][-1] for x in xs if x['rows'] and 'error' in x['rows'][-1]]
    out = dict(design=dict(date='2026-10-29', turns=64, ns=list(NS), note='Same host and session for the two Slipstream GPU settings and Xsuite GPU; CPU rows are the stored 27 October one-thread runs. '
                           'Xsuite GPU: line built once per process, then each N with one cold and three warm calls; cupy CUB reduction accelerator disabled (CUPY_ACCELERATORS empty) because its NVRTC compile failed.',
                           xsuite_processes=len(xs), failures=errs),
               xsuite_gpu_build=dict(import_s=[x['import_s'] for x in xs], build_s=[x['build_s'] for x in xs], versions=xs[0]['versions'] if xs else None),
               analysis_sha256=common.sha256(__file__), rows=rows)
    (common.EVIDENCE / '2026-10-26_R1/xsuite_gpu_scaling.json').write_text(json.dumps(out, indent=2) + '\n')
    for r in rows:
        print(r['n'], {k: (round(v['warm_s'], 3) if isinstance(v, dict) and 'warm_s' in v else None) for k, v in r.items() if k != 'n'})


if __name__ == '__main__':
    main()

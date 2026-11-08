"""Particle-count ladder for the ring workload (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P2).

The primary ring case tracks five particles, so it measures dispatch latency.  This control keeps the lattice, energy, settings and
adapters and raises the number of launches to 5, 50, 500, 5000 and 50000 for 64 turns (the first five are the original x = 1..8 mm
launches; the rest are drawn uniformly in x = 0.5..8 mm with a fixed seed, y = 1 mm, zero momenta).  It reports the warm time, the
time per particle-turn, the first-five-particle agreement between codes at the largest common N, and the CPU/GPU crossover.

    python particle_scaling_control.py --make-cases      # write cases/ring_scale_n*.json
    python particle_scaling_control.py                   # analyse results/2026-10-27_scaling
"""
import argparse
import json
from pathlib import Path

import numpy as np

import common

HERE = Path(__file__).resolve().parent
NS = (5, 50, 500, 5000, 50000)
TURNS = 64
CONFIGS = [('slipstream', 'cuda-f64-o6q8s1b8'), ('slipstream', 'cuda-f32-o6q8s1b8'), ('slipstream', 'cpu-f64-o6q8s1b8-compiled'),
           ('xsuite', 'cpu-f64-yosk8b16-exact'), ('elegant', 'cpu-f64-o6s8b16')]
ROOT = common.RESULTS / '2026-10-27_scaling'


def make_cases():
    base = common.load(HERE / 'cases/ring_tunes.json')
    rng = np.random.default_rng(20261027)
    first = base['particles']['x0']
    for n in NS:
        x0 = list(first) + [float(v) for v in rng.uniform(0.5e-3, 8e-3, max(0, n - len(first)))] if n > len(first) else list(first)[:n]
        case = dict(base, id=f'ring_scale_n{n}', title=f'Particle-count ladder: {n} launches, {TURNS} turns, same ring and settings as ring_tunes',
                    turns=TURNS, particles=dict(base['particles'], x0=x0), gate='control (timing versus particle count; no pass gate)')
        case.pop('tune_windows', None)
        (HERE / 'cases' / f'ring_scale_n{n}.json').write_text(json.dumps(case) + '\n')
    print('wrote', [f'ring_scale_n{n}.json' for n in NS])


def analyse():
    rows = []
    states = {}
    for n in NS:
        case = f'ring_scale_n{n}'
        for code, config in CONFIGS:
            rs = [common.load(p) for p in sorted((ROOT / case / code / config).glob('p*/result.json'))]
            ok = [r for r in rs if r['status'] == 'ok']
            row = dict(n=n, code=code, config=config, n_recorded=len(rs), n_ok=len(ok))
            if ok:
                warm = float(np.median([r['timing']['warm_s']['median'] for r in ok]))
                row.update(warm_s=warm, cold_s=float(np.median([r['timing']['cold_s'] for r in ok])), per_particle_turn_us=1e6 * warm / (n * TURNS),
                           cuda_peak_mb=max((r['memory'].get('cuda_peak_allocated') or 0) for r in ok) / 2**20)
                p0 = (ROOT / case / code / config / 'p0' / 'tbt.npz')
                if p0.exists() and n in (5, 50000):
                    states[n, code, config] = np.load(p0)['tbt'][:5, -1, :4]
            rows.append(row)
    # first-five-particle agreement (final state after 64 turns) against elegant at the same N
    for r in rows:
        ref = states.get((r['n'], 'elegant', 'cpu-f64-o6s8b16'))
        got = states.get((r['n'], r['code'], r['config']))
        if ref is not None and got is not None:
            r['first5_state_diff'] = dict(zip(('x_m', 'px', 'y_m', 'py'), np.abs(got - ref).max(axis=0).tolist()))  # per component; positions and momenta are never mixed
    out = dict(design=dict(date='2026-10-27', turns=TURNS, ns=list(NS), procs=2, repeats=1, threads=1,
                           note='Warm times of one adapter call (tracking only, includes process or file overhead for elegant); not a throughput claim for other lattices or settings'),
               analysis_sha256=common.sha256(__file__), rows=rows)
    (common.EVIDENCE / '2026-10-26_R1/particle_scaling_control.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f"{'N':>6s} {'code':11s} {'config':26s} ok  warm s   us/particle-turn  first5 diff (worst component)")
    for r in rows:
        if r['n_ok']:
            print(f"{r['n']:6d} {r['code']:11s} {r['config']:26s} {r['n_ok']}  {r['warm_s']:8.3g}  {r['per_particle_turn_us']:10.3g}  {max((r.get('first5_state_diff') or {'x_m': float('nan')}).values()):.2e}")
        else:
            print(f"{r['n']:6d} {r['code']:11s} {r['config']:26s} {r['n_ok']}  no data")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--make-cases', action='store_true')
    a = ap.parse_args()
    make_cases() if a.make_cases else analyse()

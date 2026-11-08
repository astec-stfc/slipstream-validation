"""Combine the SciBmad scaling ladders into data/2026-10-26_R1/scibmad_scaling.json (7 November 2026).

Inputs: results/2026-11-07_scibmad_scaling/{cpu,cuda}-f{32,64}[-o4k4].json (scibmad_scaling.py; the unsuffixed files are order 6 with 8 steps, ``-o4k4`` is
order 4 with 4 steps per quadrupole and sextupole, the setting that matches Slipstream's and Xsuite's accuracy).  The final states of the first five
particles after 64 turns are compared with the stored elegant run at the same particle count (separate x, px, y, py residuals).
"""
import json
from pathlib import Path

import numpy as np

import common

HERE = Path(__file__).resolve().parent
SRC = common.RESULTS / '2026-11-07_scibmad_scaling'
OLD = common.RESULTS / '2026-10-27_scaling'
REF = common.RESULTS / '2026-10-27_momentum/ring_tunes/elegant/cpu-f64-o6s16b32/p0/tbt.npz'
GPU_TUNES = common.RESULTS / '2026-11-07_scibmad_gpu/ring_tunes/scibmad'   # one process per GPU rung (the CPU rungs are in 2026-11-07_scibmad)
NS = (5, 50, 500, 5000, 50000)
FILES = {'cpu_f64_o6k8': 'cpu-f64.json', 'gpu_f64_o6k8': 'cuda-f64.json', 'gpu_f32_o6k8': 'cuda-f32.json',
         'cpu_f64_o4k4': 'cpu-f64-o4k4.json', 'gpu_f64_o4k4': 'cuda-f64-o4k4.json', 'gpu_f32_o4k4': 'cuda-f32-o4k4.json'}


def elegant_state(n):
    p = OLD / f'ring_scale_n{n}' / 'elegant' / 'cpu-f64-o6s8b16' / 'p0' / 'tbt.npz'
    return np.load(p)['tbt'][:5, -1, :4] if p.exists() else None


def ring_tunes():
    """Tune error against the corrected elegant reference for the GPU rungs (1024 turns, five launches), with the CPU rung of the same setting."""
    case = common.load(HERE / 'cases/ring_tunes.json')
    ref = np.load(REF)['tbt']
    qr = common.tunes(ref, case)
    rows = []
    for d in sorted(GPU_TUNES.glob('cuda-*/p0')) + [common.RESULTS / '2026-11-07_scibmad/ring_tunes/scibmad/cpu-f64-o4k4/p0']:
        res = common.load(d / 'result.json')
        tbt = np.load(d / 'tbt.npz')['tbt']
        q = common.tunes(tbt, case)
        rows.append(dict(config=res['config']['name'], warm_s=res['timing']['warm_s']['median'], cold_s=res['timing']['cold_s'],
                         dq_ref=float(max(np.abs(a - b).max() for a, b in zip(q, qr))),
                         onturn=[float(np.abs(tbt[:, 1, i] - ref[:, 1, i]).max()) for i in range(4)]))
    return rows


def main():
    runs = {k: json.loads((SRC / f).read_text()) for k, f in FILES.items() if (SRC / f).exists()}
    rows = []
    for n in NS:
        ref = elegant_state(n)
        row = dict(n=n)
        for label, r in runs.items():
            x = next((q for q in r['rows'] if q.get('n') == n and 'warm_median_s' in q), None)
            if not x:
                continue
            st = np.asarray(x['first5_final_state'])[:, :4]
            row[label] = dict(warm_s=x['warm_median_s'], warm_range_s=[min(x['warm_s']), max(x['warm_s'])], cold_s=x['cold_s'],
                              per_particle_turn_us=x['per_particle_turn_us'],
                              first5_vs_elegant=None if ref is None else dict(zip(('x_m', 'px', 'y_m', 'py'), np.abs(st - ref).max(axis=0).tolist())))
        rows.append(row)
    out = dict(design=dict(date='2026-11-07', turns=64, ns=list(NS), note='One Julia process per device and precision; line built once, then each N with one cold and three warm calls (device-to-host copy included); '
                           'one CPU thread; GPU runs capped at 4 GiB; order 6 with 8 steps (o6k8) and order 4 with 4 steps (o4k4) per quadrupole and sextupole, fringes off, exact bends.'),
               versions={k: r['versions'] for k, r in runs.items()}, import_s={k: r['import_s'] for k, r in runs.items()},
               analysis_sha256=common.sha256(__file__), ring_tunes=ring_tunes(), rows=rows)
    (common.EVIDENCE / '2026-10-26_R1/scibmad_scaling.json').write_text(json.dumps(out, indent=2) + '\n')
    for r in rows:
        print(r['n'], {k: round(v['warm_s'], 3) for k, v in r.items() if k != 'n'})


if __name__ == '__main__':
    main()

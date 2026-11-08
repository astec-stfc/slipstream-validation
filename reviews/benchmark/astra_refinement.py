"""Summarise independent ASTRA 3-D mesh and time-step sensitivity controls."""
import json
from pathlib import Path

import numpy as np

import common
from analyse_sc import OBS, rel

HERE = Path(__file__).resolve().parent
CASE = HERE / 'cases/sc_drift.json'
OLD = common.RESULTS / '2026-10-26_R1/sc_drift/astra/cpu-f64-3d32h1.4'
NEW = common.RESULTS / '2026-10-27_astra_refinement/sc_drift/astra'


def main():
    case = common.load(CASE)
    refpath = OLD / 'p0/tbt.npz'
    reference = np.load(refpath)['tbt']
    rows = []
    for root, config in [(OLD.parent, OLD.name), (NEW, 'cpu-f64-3d64h1.4'), (NEW, 'cpu-f64-3d32h0.7')]:
        jobs = []
        states = []
        for p in sorted((root / config).glob('p*/result.json')):
            r = common.load(p)
            job = dict(result=r, result_sha256=common.sha256(p))
            if r['status'] == 'ok':
                assert r['case_sha256'] == common.sha256(CASE)
                data = p.parent / 'tbt.npz'
                state = np.load(data)['tbt']
                assert state.shape == reference.shape and np.isfinite(state).all()
                states.append(state)
                job['artifact_sha256'] = common.sha256(data)
                job['versus_baseline'] = {k: rel(state, reference, case['columns'], v) for k, v in OBS.items()}
            jobs.append(job)
        good = [j for j in jobs if 'versus_baseline' in j]
        row = dict(config=config, n_recorded=len(jobs), n_ok=len(good), jobs=jobs)
        if good:
            row.update(warm_s=float(np.median([j['result']['timing']['warm_s']['median'] for j in good])),
                       versus_baseline={k: max(j['versus_baseline'][k] for j in good) for k in OBS})
            native = {
                'cpu64': common.RESULTS / '2026-10-26_R1/sc_drift/slipstream/cpu-f64-3d32t1c1',
                'gpu_fast': common.RESULTS / '2026-10-26_R1_injector_cost/sc_drift/slipstream/cuda-f32-3d32t1c10-r100',
            }
            row['native_vs_this'] = {}
            for name, source in native.items():
                inputs = sorted(source.glob('p*/tbt.npz'))
                assert len(inputs) == 3
                data = [np.load(p)['tbt'] for p in inputs]
                row['native_vs_this'][name] = {k: max(rel(a, b, case['columns'], v) for a in data for b in states)
                                               for k, v in OBS.items()}
                row.setdefault('native_artifact_sha256', {})[name] = [common.sha256(p) for p in inputs]
        rows.append(row)
    out = dict(design=dict(date='2026-10-27', procs=3, repeats=2, threads=1,
                           note='Vary one of 3-D grid and time step independently; no convergence claim from two grid points.'),
               case_sha256=common.sha256(CASE), reference_sha256=common.sha256(refpath),
               analysis_sha256=common.sha256(__file__), rows=rows)
    dest = common.EVIDENCE / '2026-10-26_R1/astra_refinement.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'jobs'} for r in rows], indent=2))


if __name__ == '__main__':
    main()

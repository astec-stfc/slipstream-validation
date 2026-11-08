"""Summarise a push-step refinement at fixed physical refresh and recording intervals."""
import json
from pathlib import Path

import numpy as np

import common
from analyse_sc import OBS, rel

HERE = Path(__file__).resolve().parent
CASE = HERE / 'cases/sc_drift.json'
BASE = common.RESULTS / '2026-10-26_R1_injector_cost/sc_drift/slipstream/cuda-f32-3d32t1c10-r100'
NEW = common.RESULTS / '2026-10-27_injector_step/sc_drift/slipstream/cuda-f32-3d32t0.5c20-r200'
ASTRA = common.RESULTS / '2026-10-26_R1/sc_drift/astra/cpu-f64-3d32h1.4/p0/tbt.npz'
ASTRA64 = common.RESULTS / '2026-10-27_astra_refinement/sc_drift/astra/cpu-f64-3d64h1.4/p0/tbt.npz'


def main():
    case = common.load(CASE)
    base = np.load(BASE / 'p0/tbt.npz')['tbt']
    astra = np.load(ASTRA)['tbt']
    astra64 = np.load(ASTRA64)['tbt']
    rows = []
    for root in (BASE, NEW):
        jobs = []
        for p in sorted(root.glob('p*/result.json')):
            r = common.load(p)
            job = dict(result=r, result_sha256=common.sha256(p))
            if r['status'] == 'ok':
                assert r['case_sha256'] == common.sha256(CASE)
                data = p.parent / 'tbt.npz'
                state = np.load(data)['tbt']
                assert state.shape == base.shape and np.isfinite(state).all()
                job['artifact_sha256'] = common.sha256(data)
                for key, ref in [('versus_base', base), ('versus_astra', astra), ('versus_astra64', astra64)]:
                    job[key] = {k: rel(state, ref, case['columns'], v) for k, v in OBS.items()}
            jobs.append(job)
        good = [j for j in jobs if 'versus_base' in j]
        row = dict(config=root.name, n_recorded=len(jobs), n_ok=len(good), jobs=jobs)
        if good:
            row['warm_s'] = float(np.median([j['result']['timing']['warm_s']['median'] for j in good]))
            for key in ('versus_base', 'versus_astra', 'versus_astra64'):
                row[key] = {k: max(j[key][k] for j in good) for k in OBS}
            a = row['versus_astra']
            row['pass'] = (len(good) == 3 and max(row['versus_base'].values()) <= 1e-3 and
                           a['size'] <= .03 and a['length'] <= .03 and a['emit'] <= .05 and a['spread'] <= .01)
        rows.append(row)
    out = dict(design=dict(date='2026-10-27', procs=3, repeats=2, threads=1,
                           fixed='Cartesian 32^3, 10 ps field refresh, 100 ps recording interval',
                           gate='max size/length/emittance/spread change <=1e-3; original ASTRA gates plus <=1% spread'),
               case_sha256=common.sha256(CASE), base_sha256=common.sha256(BASE / 'p0/tbt.npz'),
               reference_sha256=common.sha256(ASTRA), reference64_sha256=common.sha256(ASTRA64),
               analysis_sha256=common.sha256(__file__), rows=rows)
    dest = common.EVIDENCE / '2026-10-26_R1/injector_step_control.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'jobs'} for r in rows], indent=2))


if __name__ == '__main__':
    main()

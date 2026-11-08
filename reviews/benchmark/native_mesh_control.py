"""Analyse the matched Cartesian-grid drift control, without selecting a reference as truth."""
import json
from pathlib import Path

import numpy as np

import common
from analyse_sc import OBS, rel

HERE = Path(__file__).resolve().parent
CASE = HERE / 'cases/sc_drift.json'
BASE = common.RESULTS / '2026-10-26_R1_injector_cost/sc_drift/slipstream/cuda-f32-3d32t1c10-r100'
NEW = common.RESULTS / '2026-10-27_native_mesh/sc_drift/slipstream'
REFS = {
    'base': BASE / 'p0/tbt.npz',
    'astra32': common.RESULTS / '2026-10-26_R1/sc_drift/astra/cpu-f64-3d32h1.4/p0/tbt.npz',
    'astra64': common.RESULTS / '2026-10-27_astra_refinement/sc_drift/astra/cpu-f64-3d64h1.4/p0/tbt.npz',
}
METRICS = dict(OBS, emit_x=('eps_nx_m',), emit_y=('eps_ny_m',))


def main():
    case = common.load(CASE)
    refs = {name: np.load(path)['tbt'] for name, path in REFS.items()}
    rows = []
    for root in (NEW / 'cuda-f32-3d16t1c10-r100', BASE, NEW / 'cuda-f32-3d64t1c10-r100'):
        jobs = []
        for p in sorted(root.glob('p*/result.json')):
            r = common.load(p)
            job = dict(result=r, result_sha256=common.sha256(p))
            if r['status'] == 'ok':
                assert r['case_sha256'] == common.sha256(CASE)
                data = p.parent / 'tbt.npz'
                state = np.load(data)['tbt']
                assert state.shape == refs['base'].shape and np.isfinite(state).all()
                job['artifact_sha256'] = common.sha256(data)
                for key, ref in refs.items():
                    job['versus_' + key] = {k: rel(state, ref, case['columns'], v) for k, v in METRICS.items()}
            jobs.append(job)
        good = [j for j in jobs if 'versus_base' in j]
        row = dict(config=root.name, n_recorded=len(jobs), n_ok=len(good), jobs=jobs)
        if good:
            row['warm_s'] = float(np.median([j['result']['timing']['warm_s']['median'] for j in good]))
            for key in refs:
                field = 'versus_' + key
                row[field] = {k: max(j[field][k] for j in good) for k in METRICS}
        rows.append(row)
    out = dict(design=dict(procs=3, repeats=2, threads=1, fixed='1 ps push, 10 ps refresh, 100 ps records, '
                           'auto-fitted box with 5% particle-span margin',
                           note='Two ASTRA meshes are peer references, not converged truth. No mesh is selected from observed errors.'),
               case_sha256=common.sha256(CASE), references_sha256={k: common.sha256(v) for k, v in REFS.items()},
               analysis_sha256=common.sha256(__file__), rows=rows)
    dest = common.EVIDENCE / '2026-10-26_R1/native_mesh_control.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'jobs'} for r in rows], indent=2))


if __name__ == '__main__':
    main()

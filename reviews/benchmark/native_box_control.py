"""Summarise Cartesian box-padding controls against saved drift outputs."""
import json
from pathlib import Path

import numpy as np

import common
from analyse_sc import OBS, rel

HERE = Path(__file__).resolve().parent
CASE = HERE / 'cases/sc_drift.json'
ROOT = common.RESULTS / '2026-10-27_native_box'
SAVED = common.RESULTS / '2026-10-27_native_mesh/sc_drift/slipstream/cuda-f32-3d64t1c10-r100/p0/tbt.npz'
ASTRA32 = common.RESULTS / '2026-10-26_R1/sc_drift/astra/cpu-f64-3d32h1.4/p0/tbt.npz'
ASTRA64 = common.RESULTS / '2026-10-27_astra_refinement/sc_drift/astra/cpu-f64-3d64h1.4/p0/tbt.npz'
NAMES = ['cuda-f32-3d64t1c10-r100-p0.05', 'cuda-f32-3d64t1c10-r100-p0.2', 'cuda-f32-3d81t1c10-r100-p0.2']
METRICS = dict(OBS, emit_x=('eps_nx_m',), emit_y=('eps_ny_m',))


def main():
    case = common.load(CASE)
    refs = {k: np.load(p)['tbt'] for k, p in [('saved64', SAVED), ('astra32', ASTRA32), ('astra64', ASTRA64)]}
    root = ROOT / 'sc_drift/slipstream'
    anchor = root / NAMES[0] / 'p0/tbt.npz'
    if anchor.exists():
        refs['new_default'] = np.load(anchor)['tbt']
    rows = []
    for name in NAMES:
        jobs = []
        for path in sorted((root / name).glob('p*/result.json')):
            r = common.load(path)
            job = dict(result=r, result_sha256=common.sha256(path))
            if r['status'] == 'ok':
                assert r['case_sha256'] == common.sha256(CASE)
                data = path.parent / 'tbt.npz'
                state = np.load(data)['tbt']
                assert state.shape == refs['saved64'].shape and np.isfinite(state).all()
                job['artifact_sha256'] = common.sha256(data)
                for key, ref in refs.items():
                    job['versus_' + key] = {k: rel(state, ref, case['columns'], v) for k, v in METRICS.items()}
            jobs.append(job)
        good = [j for j in jobs if 'versus_saved64' in j]
        row = dict(config=name, n_recorded=len(jobs), n_ok=len(good), jobs=jobs)
        if good:
            row['warm_s'] = float(np.median([j['result']['timing']['warm_s']['median'] for j in good]))
            row['cuda_peak_bytes'] = max(j['result']['memory'].get('cuda_peak_allocated', 0) for j in good)
            for key in refs:
                field = 'versus_' + key
                row[field] = {k: max(j[field][k] for j in good) for k in METRICS}
            if name == NAMES[0] and len(good) == 3:
                row['default_parity_pass'] = max(row['versus_saved64'].values()) <= 1e-5
        rows.append(row)
    out = dict(design=dict(procs=3, repeats=2, threads=1,
                           fixed='1 ps push, 10 ps refresh, 100 ps records',
                           default_parity='max relative observable change <=1e-5 against saved 64^3',
                           box='5% and 20% padding at 64^3, plus 20% at 81^3 for approximately matched initial cell size; '
                               'dynamic refitting and cell-size buckets may change actual boxes.',
                           execution='81^3/p0 was a pilot followed by the three-process continuation; all jobs ran sequentially.',
                           references='ASTRA32 and ASTRA64 are peers, not converged truth'),
               case_sha256=common.sha256(CASE), references_sha256={k: common.sha256(p) for k, p in
                   [('saved64', SAVED), ('astra32', ASTRA32), ('astra64', ASTRA64)]},
               source_snapshot_sha256=common.sha256(ROOT / 'source_snapshot.zip'),
               analysis_sha256=common.sha256(__file__), rows=rows)
    dest = common.EVIDENCE / '2026-10-26_R1/native_box_control.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'jobs'} for r in rows], indent=2))


if __name__ == '__main__':
    main()

"""Summarise the double-single ring control against the corrected reference."""
import json
from pathlib import Path

import numpy as np

import common

HERE = Path(__file__).resolve().parent
CASE = HERE / 'cases/ring_tunes.json'
OLD = common.RESULTS / '2026-10-26_R1/ring_tunes/slipstream'
NEW = common.RESULTS / '2026-10-28_precision/ring_tunes/slipstream'
REF = common.RESULTS / '2026-10-27_momentum/ring_tunes/elegant/cpu-f64-o6s16b32/p0/tbt.npz'
CONFIGS = [(OLD, 'cuda-f32-o6q8s1b8'), (OLD, 'cuda-f64-o6q8s1b8'),
           (NEW, 'cuda-f32ds-hybrid-o6q8s1b8'), (NEW, 'cuda-f32ds-full-o6q8s1b8')]


def main():
    case = common.load(CASE)
    ref = np.load(REF)['tbt']
    qref = common.tunes(ref, case)
    rows = []
    for root, config in CONFIGS:
        jobs = []
        for path in sorted((root / config).glob('p*/result.json')):
            r = common.load(path)
            job = dict(result=r, result_sha256=common.sha256(path))
            if r['status'] == 'ok':
                assert r['case_sha256'] == common.sha256(CASE)
                data = path.parent / 'tbt.npz'
                tbt = np.load(data)['tbt']
                assert tbt.shape == ref.shape and np.isfinite(tbt).all()
                q = common.tunes(tbt, case)
                job.update(artifact_sha256=common.sha256(data),
                           dq_ref=float(max(np.abs(a-b).max() for a, b in zip(q, qref))),
                           map_components=np.abs(tbt[:, 1] - ref[:, 1]).max(axis=0).tolist())
            jobs.append(job)
        good = [j for j in jobs if 'dq_ref' in j]
        row = dict(config=config, n_started=len(list((root / config).glob('p*/job.json'))),
                   n_recorded=len(jobs), n_ok=len(good), jobs=jobs)
        if good:
            warm = [j['result']['timing']['warm_s']['median'] for j in good if j['result']['timing'].get('warm_s')]   # none when the job ran without warm repeats
            row.update(warm_s=float(np.median(warm)) if warm else None,
                       cold_s=float(np.median([j['result']['timing']['cold_s'] for j in good])),
                       dq_ref=max(j['dq_ref'] for j in good),
                       map_components=np.max([j['map_components'] for j in good], axis=0).tolist())
            row['both_gates'] = (row['dq_ref'] <= case['tolerance']['tune_abs']
                                 and max(row['map_components']) <= case['tolerance']['onturn_abs']) if len(good) == 3 else None
        rows.append(row)
    out = dict(design=dict(date='2026-11-07', campaign='2026-10-28_precision',
                           note='Campaign name is a label; the jobs ran with slipstream 1.1.0 on 7-8 November 2026 at the paper setting o6q8s1 (the folder keeps its v1.0 name). '
                                'Baseline arrays are the f32 and f64 rungs of 2026-10-26_R1 at the same setting. The full-mode rung ran once, one process and no warm repeats: '
                                'its accuracy and cold time are reported, no warm timing is claimed (the v1.0 attempt was interrupted after more than six minutes of cold setup).',
                           procs=3, repeats=2, threads=1, target='unchanged tune and first-turn component gates'),
               case_sha256=common.sha256(CASE), reference_sha256=common.sha256(REF),
               analysis_sha256=common.sha256(__file__), rows=rows)
    dest = common.EVIDENCE / '2026-10-26_R1/precision_control.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'jobs'} for r in rows], indent=2))


if __name__ == '__main__':
    main()

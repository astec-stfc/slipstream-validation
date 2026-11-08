"""P1 follow-up: reuse native CPU compilation at unchanged integration settings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import common
from analyse_grad import best_fd, errs

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', default='2026-10-26_R1')
    a = ap.parse_args()
    campaign = a.campaign + '_CPU_control'
    root = common.RESULTS / campaign
    root.mkdir(exist_ok=True)
    design = dict(classification='post-baseline performance implementation control',
                  hypothesis='Existing regional CPU compilation reduces eager dispatch while preserving maps/gradients',
                  config='cpu-f64-o6q8s1b8-compiled', procs=3, repeats=2, threads=1,
                  change='compile_cpu=True only; native parser guard is unused by this JSON-to-ElementSpec adapter',
                  gates='Unchanged ring tune gate; full component and AD/self-FD residuals retained')
    (root / 'design.json').write_text(json.dumps(design, indent=2))
    rows = []
    for case_id in ('ring_tunes', 'ring_grad'):
        subprocess.run([sys.executable, '-u', str(HERE / 'run.py'), '--case', case_id, '--campaign', campaign,
                        '--codes', 'slipstream', '--configs', design['config'], '--procs', '3', '--repeats', '2', '--threads', '1'], check=True)
        case = common.load(HERE / 'cases' / (case_id + '.json'))
        reference = common.RESULTS / a.campaign / case_id / 'elegant/cpu-f64-o6s16b32/p0'
        jobs = []
        for p in sorted((root / case_id).glob('*/*/p*/result.json')):
            r = common.load(p)
            job = dict(result=r)
            if r['status'] == 'ok':
                assert r['case_sha256'] == common.sha256(HERE / 'cases' / (case_id + '.json'))
                if case_id == 'ring_tunes':
                    tbt = np.load(p.parent / 'tbt.npz')['tbt']
                    ref = np.load(reference / 'tbt.npz')['tbt']
                    q = common.tunes(tbt, case)
                    qr = common.tunes(ref, case)
                    job['dq_ref'] = float(max(np.max(np.abs(x-y)) for x, y in zip(q, qr)))
                    job['map_components'] = np.abs(tbt[:, 1] - ref[:, 1]).max(axis=0).tolist()
                else:
                    z = np.load(p.parent / 'grad.npz')
                    J, _ = best_fd(z['J_fd'], z['steps'])
                    zr = np.load(reference / 'grad.npz')
                    Jr, _ = best_fd(zr['J_fd'], zr['steps'])
                    job['ad_vs_fd'] = errs(z['J_ad'], J)
                    job['versus_reference'] = errs(z['J_ad'], Jr)
                    old = np.load(common.RESULTS / a.campaign / case_id / 'slipstream/cpu-f64-o6q8s1b8/p0/grad.npz')
                    job['versus_eager_ad'] = errs(z['J_ad'], old['J_ad'])
            jobs.append(job)
        good = [j for j in jobs if j['result']['status'] == 'ok']
        row = dict(case=case_id, jobs=jobs, n_ok=len(good))
        if good and case_id == 'ring_tunes':
            row.update(warm_s=float(np.median([j['result']['timing']['warm_s']['median'] for j in good])),
                       dq_ref=max(j['dq_ref'] for j in good))
        elif good:
            row.update(fd_s=float(np.median([np.mean(list(j['result']['fd_s'].values())) for j in good])),
                       ad_s=float(np.median([j['result']['ad_warm_s']['median'] for j in good])),
                       ad_vs_fd=np.max([j['ad_vs_fd'] for j in good], axis=0).tolist(),
                       versus_eager_ad=np.max([j['versus_eager_ad'] for j in good], axis=0).tolist())
        rows.append(row)
        dest = common.EVIDENCE / a.campaign / 'cpu_control.json'
        dest.write_text(json.dumps(dict(design=design, rows=rows), indent=2) + '\n')
        print(case_id, {k: v for k, v in row.items() if k != 'jobs'}, flush=True)


if __name__ == '__main__':
    main()

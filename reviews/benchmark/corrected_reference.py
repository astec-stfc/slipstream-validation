"""Reanalyse saved observations against the corrected elegant momentum reference."""
import csv
import json
from pathlib import Path

import numpy as np
import common
from analyse_grad import best_fd, errs

HERE = Path(__file__).resolve().parent
OLD = common.RESULTS / '2026-10-26_R1'
NEW = common.RESULTS / '2026-10-27_momentum'
SCIBMAD = common.RESULTS / '2026-11-07_scibmad'
DATA = common.EVIDENCE / '2026-10-26_R1'


def error_tune(state, reference, case):
    a, b = common.tunes(state, case), common.tunes(reference, case)
    return float(max(np.abs(x-y).max() for x, y in zip(a, b)))


def main():
    case = common.load(HERE / 'cases/ring_tunes.json')
    reference_jobs = []
    for name, configs in [('ring_tunes', ('cpu-f64-o6s8b16', 'cpu-f64-o6s16b32')),
                          ('ring_grad', ('cpu-f64-o6s16b32',))]:
        for config in configs:
            jobs = sorted((NEW / name / 'elegant' / config).glob('p*/result.json'))
            assert len(jobs) == 3
            for path in jobs:
                job = common.load(path)
                assert job['status'] == 'ok' and job['case_sha256'] == common.sha256(HERE / 'cases' / (name + '.json'))
                sources = {k.replace('\\', '/'): v for k, v in job['source_sha256'].items()}
                assert sources['workers/w_elegant.py'] == common.archived_source_sha256('workers/w_elegant.py')
                reference_jobs.append(dict(result=job, path=path.relative_to(NEW).as_posix(),
                                           result_sha256=common.sha256(path),
                                           artifacts_sha256={p.name: common.sha256(p) for p in path.parent.glob('*.npz')}))
    ref = np.load(NEW / 'ring_tunes/elegant/cpu-f64-o6s16b32/p0/tbt.npz')['tbt']
    reference = np.load(NEW / 'ring_grad/elegant/cpu-f64-o6s16b32/p0/grad.npz')
    Jref, _ = best_fd(reference['J_fd'], reference['steps'])
    old_rows = list(csv.DictReader((OLD / 'ring_tunes/summary.csv').open(newline='')))
    ring = []
    for row in old_rows:
        if row['code'] == 'elegant':
            continue
        paths = sorted((OLD / 'ring_tunes' / row['code'] / row['config']).glob('p*/tbt.npz'))
        assert len(paths) == 3
        assert all(common.load(p.parent / 'result.json')['case_sha256'] == common.sha256(HERE / 'cases/ring_tunes.json') for p in paths)
        states = [np.load(p)['tbt'] for p in paths]
        assert all(np.isfinite(s).all() and s.shape == ref.shape for s in states)
        row['dq_ref'] = max(error_tune(state, ref, case) for state in states)
        row['pass'] = str(row['dq_ref'] <= case['tolerance']['tune_abs'])
        for i, k in enumerate(('x', 'px', 'y', 'py')):
            row['onturn_' + k] = max(float(np.abs(s[:, 1, i] - ref[:, 1, i]).max()) for s in states)
        ring.append(row)
    for config, rank in [('cpu-f64-o6s8b16', 3), ('cpu-f64-o6s16b32', 4)]:
        paths = sorted((NEW / 'ring_tunes/elegant' / config).glob('p*/result.json'))
        assert len(paths) == 3
        jobs = [common.load(p) for p in paths]
        assert all(r['status'] == 'ok' for r in jobs)
        states = [np.load(p.parent / 'tbt.npz')['tbt'] for p in paths]
        dq = max(error_tune(s, ref, case) for s in states)
        ring.append(dict(code='elegant', config=config, rank=str(rank), warm_s=str(np.median([r['timing']['warm_s']['median'] for r in jobs])),
                         dq_ref=str(dq), **{'pass': str(dq <= case['tolerance']['tune_abs'])},
                         **{'onturn_'+k: max(float(np.abs(s[:, 1, i]-ref[:, 1, i]).max()) for s in states) for i, k in enumerate(('x','px','y','py'))}))
    grad = []
    for row in csv.DictReader((OLD / 'ring_grad/summary.csv').open(newline='')):
        code, config = row['code'], row['config']
        path = OLD / 'ring_grad' / code / config / ('p' + str(sum(1 for r in grad if r['code'] == code))) / 'grad.npz'
        if code == 'elegant':
            path = NEW / 'ring_grad/elegant/cpu-f64-o6s16b32' / path.parent.name / 'grad.npz'
            job = common.load(path.parent / 'result.json')
            assert job['status'] == 'ok'
            row['forward_s'] = job['forward_s']
            row['fd_s'] = float(np.mean(list(job['fd_s'].values())))
        assert common.load(path.parent / 'result.json')['case_sha256'] == common.sha256(HERE / 'cases/ring_grad.json')
        z = np.load(path)
        assert np.isfinite(z['J_fd']).all() and np.isfinite(z['y0']).all()
        J, label = best_fd(z['J_fd'], z['steps'])
        if code == 'elegant':
            row['h_best'] = label
        row['rel_fro'], row['rel_col'] = errs(z['J_ad'] if 'J_ad' in z else J, Jref)
        row['dy0'] = float(np.abs(z['y0'] - reference['y0']).max())
        grad.append(row)
    # SciBmad (added 7 November 2026, replacing Bmad-X in the paper): run later, on the same host, against the same corrected reference.
    for cfg_dir in sorted((SCIBMAD / 'ring_tunes/scibmad').glob('*')):
        paths = sorted(cfg_dir.glob('p*/tbt.npz'))
        assert len(paths) == 3, cfg_dir
        jobs = [common.load(p.parent / 'result.json') for p in paths]
        assert all(j['status'] == 'ok' and j['case_sha256'] == common.sha256(HERE / 'cases/ring_tunes.json') for j in jobs)
        states = [np.load(p)['tbt'] for p in paths]
        assert all(np.isfinite(s).all() and s.shape == ref.shape for s in states)
        dq = max(error_tune(s, ref, case) for s in states)
        ring.append(dict(code='scibmad', config=cfg_dir.name, rank=str(jobs[0]['rank']), warm_s=str(np.median([j['timing']['warm_s']['median'] for j in jobs])),
                         dq_ref=str(dq), **{'pass': str(dq <= case['tolerance']['tune_abs'])},
                         **{'onturn_' + k: max(float(np.abs(s[:, 1, i] - ref[:, 1, i]).max()) for s in states) for i, k in enumerate(('x', 'px', 'y', 'py'))}))
    for cfg_dir in sorted((SCIBMAD / 'ring_grad/scibmad').glob('*')):
        for path in sorted(cfg_dir.glob('p*/grad.npz')):
            job = common.load(path.parent / 'result.json')
            assert job['status'] == 'ok' and job['case_sha256'] == common.sha256(HERE / 'cases/ring_grad.json')
            z = np.load(path)
            assert np.isfinite(z['J_fd']).all() and np.isfinite(z['J_ad']).all() and np.isfinite(z['y0']).all()
            J, label = best_fd(z['J_fd'], z['steps'])
            row = dict(code='scibmad', config=cfg_dir.name, status='ok', h_best=label, forward_s=job['forward_s'], fd_s=float(np.mean(list(job['fd_s'].values()))),
                       ad_cold_s=job['ad_cold_s'], ad_warm_s=job['ad_warm_s']['median'])
            row['ad_vs_fd_fro'], row['ad_vs_fd_col'] = errs(z['J_ad'], J)
            row['rel_fro'], row['rel_col'] = errs(z['J_ad'], Jref)
            row['dy0'] = float(np.abs(z['y0'] - reference['y0']).max())
            grad.append(row)
    model = common.load(DATA / 'xsuite_model_control.json')
    for row in model['rows']:
        for job in row['jobs']:
            r = job['result']
            process = str(r['job']['out_dir']).replace('\\', '/').rsplit('/', 1)[-1]
            p = common.RESULTS / '2026-10-26_R1_Xsuite_exact' / row['case'] / 'xsuite' / row['config'] / process
            if row['case'] == 'ring_tunes':
                state = np.load(p / 'tbt.npz')['tbt']
                job['dq_ref'] = error_tune(state, ref, case)
                job['map_components'] = np.abs(state[:, 1] - ref[:, 1]).max(axis=0).tolist()
            else:
                z = np.load(p / 'grad.npz')
                J, _ = best_fd(z['J_fd'], z['steps'])
                job['rel_fro'], job['rel_col'] = errs(J, Jref)
        if row['case'] == 'ring_tunes':
            row['dq_ref'] = max(j['dq_ref'] for j in row['jobs'])
            row['map_components'] = np.max([j['map_components'] for j in row['jobs']], axis=0).tolist()
            row['pass'] = row['dq_ref'] <= case['tolerance']['tune_abs']
        else:
            row['rel_fro'] = max(j['rel_fro'] for j in row['jobs'])
            row['rel_col'] = max(j['rel_col'] for j in row['jobs'])
    cpu = common.load(DATA / 'cpu_control.json')
    for row in cpu['rows']:
        if row['case'] == 'ring_tunes':
            for job in row['jobs']:
                process = Path(job['result']['job']['out_dir']).name
                state = np.load(common.RESULTS / '2026-10-26_R1_CPU_control/ring_tunes/slipstream/cpu-f64-o6q8s1b8-compiled' / process / 'tbt.npz')['tbt']
                job['dq_ref'] = error_tune(state, ref, case)
            row['dq_ref'] = max(j['dq_ref'] for j in row['jobs'])
    out = dict(reference_campaign='2026-10-27_momentum', parent_campaign='2026-10-26_R1',
               reference_note='elegant p_central matches input p/(mc); original measurements retained',
               analysis_sha256=common.sha256(__file__), reference_jobs=reference_jobs,
               ring_tunes=ring, ring_grad=grad, xsuite=model, cpu=cpu)
    (DATA / 'corrected_reference.json').write_text(json.dumps(out, indent=2) + '\n')
    for r in ring:
        print(r['code'], r['config'], r['dq_ref'], r['onturn_x'], r['onturn_px'])
    print('Xsuite exact:', [(r['config'], r.get('dq_ref'), r.get('rel_fro')) for r in model['rows']])
    print('CPU compiled:', [(r['case'], r.get('dq_ref')) for r in cpu['rows']])


if __name__ == '__main__':
    main()

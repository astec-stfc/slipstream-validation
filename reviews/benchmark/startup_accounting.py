"""Total start-up accounting for the benchmark workers (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, item 28; review point B1).

The shared worker times ``setup()`` as import, model construction as build and the first ``track()`` call as cold; the orchestrator times the whole
worker command.  What a user pays to get *one* answer from a fresh process is therefore

    command-to-output latency  =  wall  -  (sum of the repeated warm calls)  =  import + build + first call + overhead
    overhead                   =  wall  -  (import + build + cold + sum of warm calls)     (interpreter start, tune analysis, result I/O)

``--retro`` recomputes these for every stored job (no new runs).  ``--run`` makes the controlled cache-state measurement for the compiled settings:
one fresh process against an empty compile cache (``TORCHINDUCTOR_CACHE_DIR``/``TRITON_CACHE_DIR`` in a new directory; for Xsuite on WSL also
``CUPY_CACHE_DIR``), then two more against the cache that process populated, each worker doing a single call (``--repeats 0``).

    python startup_accounting.py --retro
    python startup_accounting.py --run --campaign 2026-10-29_startup
    python startup_accounting.py --summarise --campaign 2026-10-29_startup
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

DATA = common.EVIDENCE / '2026-10-26_R1/startup_accounting.json'
COLD = [   # (case, code, config)
    ('ring_tunes', 'slipstream', 'cpu-f64-o6q8s1b8-compiled'),
    ('ring_tunes', 'slipstream', 'cuda-f64-o6q8s1b8'),
    ('ring_tunes', 'slipstream', 'cuda-f32-o6q8s1b8'),
    ('sc_drift', 'slipstream', 'cuda-f32-3d64t1c10-r100-mfix'),
    ('ring_tunes', 'xsuite', 'cuda-f64-yosk8b16-exact'),
]


def account(rec, wall):
    t = rec['timing']
    warm = (t.get('warm_s') or {}).get('samples') or []
    accounted = t['import_s'] + t['build_s'] + (t.get('cold_s') or 0.0) + sum(warm)
    return dict(import_s=t['import_s'], build_s=t['build_s'], cold_s=t.get('cold_s'), warm_median_s=(t.get('warm_s') or {}).get('median'),
                n_warm=len(warm), wall_s=wall, overhead_s=wall - accounted, command_to_output_s=wall - sum(warm))


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def collect(root):
    groups = {}
    for p in sorted(root.glob('*/*/*/*/p*/result.json')):
        rec = json.loads(p.read_text())
        wp = p.parent / 'wall.json'
        if rec.get('status') != 'ok' or not wp.exists() or not rec.get('timing') or rec['timing'].get('cold_s') is None:
            continue
        wall = json.loads(wp.read_text())['wall_s']
        case, code, cfg = p.parents[3].name, p.parents[2].name, p.parents[1].name
        groups.setdefault((p.parents[4].name, case, code, cfg), []).append(account(rec, wall))
    rows = []
    for (camp, case, code, cfg), rs in groups.items():
        row = dict(campaign=camp, case=case, code=code, config=cfg, n_proc=len(rs))
        for k in ('import_s', 'build_s', 'cold_s', 'warm_median_s', 'overhead_s', 'command_to_output_s', 'wall_s'):
            row[k] = med([r[k] for r in rs])
        row['command_to_output_range_s'] = [min(r['command_to_output_s'] for r in rs), max(r['command_to_output_s'] for r in rs)]
        row['per_process_command_to_output_s'] = [round(r['command_to_output_s'], 2) for r in rs]
        rows.append(row)
    return rows


def run(campaign, only=None):
    py = sys.executable
    for case, code, cfg in [c for c in COLD if only is None or f'{c[1]}/{c[0]}' == only]:
        tmp = Path(tempfile.mkdtemp(prefix='cold_cache_'))
        env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(tmp / 'inductor'), TRITON_CACHE_DIR=str(tmp / 'triton'))
        if code == 'xsuite':
            env.update(CUPY_CACHE_DIR=f'/tmp/{tmp.name}/cupy', WSLENV='CUPY_CACHE_DIR')
        for state, repeats in (('cold', 0), ('warm1', 0)) + (() if code == 'xsuite' else (('warm2', 0),)):
            cmd = [py, str(HERE / 'run.py'), '--case', case, '--campaign', f'{campaign}_{code}_{cfg}_{state}', '--codes', code, '--configs', cfg,
                   '--procs', '1', '--repeats', str(repeats), '--threads', '1']
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            print(code, cfg, state, 'rc', r.returncode, r.stdout.strip().splitlines()[-1:] if r.stdout else '', flush=True)
        shutil.rmtree(tmp, ignore_errors=True)


def summarise(campaign):
    rows = []
    # each state is its own campaign directory, layout <campaign>_<code>_<config>_<state>/<case>/<code>/<config>/p0
    for c, d in sorted((c, p) for c in campaign.split(',') for p in (common.RESULTS).glob(f'{c}_*')):   # comma-separated campaign prefixes
        if not d.is_dir():
            continue
        tag = d.name[len(c) + 1:]
        state = tag.rsplit('_', 1)[1]
        for p in sorted(d.glob('*/*/*/p*/result.json')):
            rec = json.loads(p.read_text())
            wp = p.parent / 'wall.json'
            if rec.get('status') != 'ok' or not wp.exists():
                rows.append(dict(setting=tag, status=rec.get('status')))
                continue
            a = account(rec, json.loads(wp.read_text())['wall_s'])
            rows.append(dict(case=p.parents[3].name, code=p.parents[2].name, config=p.parents[1].name, state=state, **a))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retro', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--summarise', action='store_true')
    ap.add_argument('--campaign', default='2026-10-29_startup', help='--summarise accepts several, comma-separated')
    ap.add_argument('--only', help='code/case for --run, e.g. slipstream/ring_tunes: rerun just these settings')
    a = ap.parse_args()
    out = json.loads(DATA.read_text()) if DATA.exists() else {}
    if a.retro:
        rows = collect(common.RESULTS)
        out['retro'] = dict(note='Every stored job with a result and a wall time; medians over processes. command_to_output = wall - sum(warm calls) = import + build + first call + overhead',
                            analysis_sha256=__import__('common').sha256(__file__), rows=rows)
        print(len(rows), 'groups')
    if a.run:
        run(a.campaign, a.only)
    if a.summarise:
        out['controlled'] = dict(note='cold = empty compile cache in a new directory, warm1/warm2 = the cache that process populated; one call per process', rows=summarise(a.campaign))
    DATA.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    main()

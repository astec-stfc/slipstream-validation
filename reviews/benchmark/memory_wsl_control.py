"""Process-tree memory of the WSL-hosted codes (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P2; complements memory_tree_control.py).

Xsuite, Bmad-X and ImpactX run inside WSL, where the Windows process tree is blind.  Each accepted CPU setting is rerun through the same
worker command as run.py, with the worker started by launch/wsl_tree_rss.py, which samples /proc every 25 ms and sums the resident set of
the whole tree (the worker and any child process).  The baseline is the same interpreter importing the worker module and running its
``setup()`` only.  The harness's own ``rss_mb`` of the same run is shown next to it.

    python memory_wsl_control.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import run  # noqa: E402

JOBS = [('xsuite', 'ring_tunes', 'cpu-f64-yosk8b16-exact'), ('bmadx', 'ring_tunes', 'cpu-f64-q1s64'), ('impactx', 'sc_drift', 'cpu-f64-mg32x40s2')]
SAMPLER = HERE / 'launch/wsl_tree_rss.py'


def with_sampler(cmd, out, extra):
    i = cmd.index('--')
    return cmd[:i + 1] + [run.plat('xsuite', SAMPLER), '--out', out, '--'] + extra


def main():
    root = common.RESULTS / '2026-10-28_memory_wsl'
    rows = []
    for code, case_id, config in JOBS:
        out = root / case_id / code / config
        out.mkdir(parents=True, exist_ok=True)
        job = dict(case_path=run.plat(code, HERE / 'cases' / f'{case_id}.json'), config=config, repeats=1, threads=1, out_dir=run.plat(code, out))
        (out / 'job.json').write_text(json.dumps(job, indent=1))
        cmd, env = run.command(code, ['--job', run.plat(code, out / 'job.json')], 1, case_id)
        worker_args = cmd[cmd.index('--') + 1:]
        module = Path(run.worker_path(code, case_id)).stem
        setup = f"import sys, time; sys.path[:0] = [{run.plat(code, HERE / 'workers')!r}, {run.plat(code, HERE)!r}]; import {module} as w; w.setup(); time.sleep(0.5)"
        for name, extra in (('baseline', ['-c', setup]), ('run', worker_args)):
            res = out / f'tree_{name}.json'
            c, _ = run.command(code, [], 1, case_id)
            subprocess.run(with_sampler(c, run.plat(code, res), extra), env=env, capture_output=True, text=True, timeout=run.REG[code].get('timeout', 3600))
        base, tree = common.load(out / 'tree_baseline.json'), common.load(out / 'tree_run.json')
        r = common.load(out / 'result.json') if (out / 'result.json').exists() else {}
        row = dict(code=code, case=case_id, config=config, status=r.get('status', 'missing'), wall_s=tree['wall_s'], max_processes=tree['max_processes'],
                   baseline_tree_mb=base['tree_peak_bytes'] / 2**20, tree_peak_mb=tree['tree_peak_bytes'] / 2**20, own_peak_mb=tree['own_peak_bytes'] / 2**20,
                   harness_rss_mb=(r.get('memory', {}).get('rss_peak_bytes') or 0) / 2**20, children_extra_mb=(tree['tree_peak_bytes'] - tree['own_peak_bytes']) / 2**20,
                   incremental_over_baseline_mb=(tree['tree_peak_bytes'] - base['tree_peak_bytes']) / 2**20)
        rows.append(row)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = dict(design=dict(date='2026-10-28', interval_s=0.025, note='One process per configuration, one repeat; /proc resident set summed over the tree; sampling can miss sub-25 ms spikes; CPU settings only'),
               analysis_sha256=common.sha256(__file__), rows=rows)
    (common.EVIDENCE / '2026-10-26_R1/memory_wsl_control.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()

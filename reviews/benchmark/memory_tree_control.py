"""Process-tree and full-device memory accounting (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P2).

The harness's ``rss_mb`` is the Python worker's own peak working set, so it omits the child executables (elegant, ASTRA, GPT, Julia) and
the CUDA context, and it includes a large import baseline.  This control re-runs one accepted configuration per code through the same
worker command as run.py and samples, every 25 ms,
  * the summed working set (RSS) of the whole process tree rooted at the launcher's child,
  * the worker's own RSS (harness-style),
  * the full-device GPU memory in use (nvidia-smi), against its value just before launch (other desktop applications use the GPU).
Baseline = the same interpreter running only the worker module's ``setup()`` (the heavy library imports), no tracking.  Windows-hosted
codes only: processes inside WSL are not visible from the Windows process tree.

    python memory_tree_control.py
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import run  # noqa: E402

JOBS = [  # (code, case, config)
    ('slipstream', 'ring_tunes', 'cpu-f64-o6q8s1b8-compiled'), ('slipstream', 'ring_tunes', 'cuda-f64-o6q8s1b8'),
    ('slipstream', 'sc_drift', 'cuda-f32-3d32t1c10-r100'), ('elegant', 'ring_tunes', 'cpu-f64-o6s8b16'),
    ('cheetah', 'sc_drift', 'cpu-f64-g32s2'), ('ocelot', 'sc_drift', 'cpu-f64-g32s2'), ('astra', 'sc_drift', 'cpu-f64-3d32h1.4'),
    ('gpt', 'sc_drift', 'cpu-f64-mesh-acc6'), ('jutrack', 'ring_tunes', 'cpu-f64-n16b32')]
INTERVAL = 0.025
GPU_INTERVAL = 0.25


def gpu_used_mib():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return float(r.stdout.split()[0])


def tree_rss(proc):
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0, 0, 0
    total, n = 0, 0
    for p in procs:
        try:
            total += p.memory_info().rss
            n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        own = proc.memory_info().rss
    except psutil.NoSuchProcess:
        own = 0
    return total, own, n


def sample(cmd, env, timeout, watch_gpu):
    """Run ``cmd`` and return (peak tree RSS, peak own RSS, max process count, GPU before, GPU peak, wall)."""
    gpu0 = gpu_used_mib() if watch_gpu else None
    proc = psutil.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = dict(tree=0, own=0, n=0, gpu=gpu0 or 0.0)
    stop = threading.Event()

    def gpu_loop():
        while not stop.is_set():
            peak['gpu'] = max(peak['gpu'], gpu_used_mib())
            stop.wait(GPU_INTERVAL)

    if watch_gpu:
        threading.Thread(target=gpu_loop, daemon=True).start()
    t0 = time.perf_counter()
    while proc.poll() is None and time.perf_counter() - t0 < timeout:
        total, own, n = tree_rss(proc)
        peak['tree'], peak['own'], peak['n'] = max(peak['tree'], total), max(peak['own'], own), max(peak['n'], n)
        time.sleep(INTERVAL)
    stop.set()
    if proc.poll() is None:
        proc.kill()
    return peak['tree'], peak['own'], peak['n'], gpu0, peak['gpu'], time.perf_counter() - t0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='code/case, e.g. slipstream/ring_tunes: rerun just these jobs and merge them into the saved table')
    a = ap.parse_args()
    root = common.RESULTS / ('2026-10-27_memory_tree' if not a.only else '2026-11-07_memory_tree')
    rows = []
    for code, case_id, config in [j for j in JOBS if a.only is None or f'{j[0]}/{j[1]}' == a.only]:
        case_path = HERE / 'cases' / f'{case_id}.json'
        out = root / case_id / code / config
        out.mkdir(parents=True, exist_ok=True)
        job = dict(case_path=run.plat(code, case_path), config=config, repeats=1, threads=1, out_dir=run.plat(code, out))
        (out / 'job.json').write_text(json.dumps(job, indent=1))
        cmd, env = run.command(code, ['--job', run.plat(code, out / 'job.json')], 1, case_id)
        module = Path(run.worker_path(code, case_id)).stem
        base_cmd = [run.REG[code]['python'], '-c', f"import sys, time; sys.path[:0] = [{str(HERE / 'workers')!r}, {str(HERE)!r}]; import {module} as w; w.setup(); time.sleep(0.5)"]
        gpu = config.startswith('cuda')
        base = sample(base_cmd, env, 600, False)
        tree, own, n, gpu0, gpu_peak, wall = sample(cmd, env, run.REG[code].get('timeout', 3600), gpu)
        res = common.load(out / 'result.json') if (out / 'result.json').exists() else {}
        mem = res.get('memory', {})
        row = dict(code=code, case=case_id, config=config, status=res.get('status', 'missing'), wall_s=wall,
                   baseline_tree_mb=base[0] / 2**20, tree_peak_mb=tree / 2**20, own_peak_mb=own / 2**20, max_processes=n,
                   harness_rss_mb=(mem.get('rss_peak_bytes') or 0) / 2**20, children_extra_mb=(tree - own) / 2**20,
                   incremental_over_baseline_mb=(tree - base[0]) / 2**20)
        if gpu:
            row.update(gpu_device_increase_mb=gpu_peak - gpu0, gpu_before_mb=gpu0, torch_allocated_mb=(mem.get('cuda_peak_allocated') or 0) / 2**20,
                       torch_reserved_mb=(mem.get('cuda_peak_reserved') or 0) / 2**20)
        rows.append(row)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = dict(design=dict(date='2026-10-27', interval_s=INTERVAL, gpu_interval_s=GPU_INTERVAL, note='One process per configuration, one repeat; sampling can miss sub-25 ms spikes; WSL-hosted codes are not covered'),
               analysis_sha256=common.sha256(__file__), rows=rows)
    dest = common.EVIDENCE / '2026-10-26_R1/memory_tree_control.json'
    if a.only and dest.exists():   # merge: the rerun rows replace the saved rows of the same code/case, every other row is kept as measured
        saved = json.loads(dest.read_text())
        out['design'] = dict(saved['design'], reruns=saved['design'].get('reruns', []) + [dict(date='2026-11-07', only=a.only, campaign='2026-11-07_memory_tree', note='slipstream 1.1.0, o6q8s1')])
        out['rows'] = [r for r in saved['rows'] if f"{r['code']}/{r['case']}" != a.only] + rows
    dest.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()

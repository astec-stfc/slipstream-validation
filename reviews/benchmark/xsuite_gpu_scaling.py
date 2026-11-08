"""Xsuite on the GPU across the particle-count ladder (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, item 29; review point B9).

Item 16 compared Slipstream's GPU settings with Xsuite on one CPU thread.  This control puts Xsuite on the same GPU: the line is built once
(the kernel build dominates, see startup_accounting.py), then the ``ring_scale_n*`` launches (5 ... 50 000 particles, 64 turns) are tracked
through the adapter's own ``track`` (which also copies the turn-by-turn monitor to the host, as the other adapters' calls do), one cold call and
``--repeats`` warm calls per particle count, synchronised.  Runs inside the WSL conda environment of the Xsuite worker:

    wsl -d Ubuntu-22.04 bash launch/conda_run.sh xsuite-gpu 1 -- xsuite_gpu_scaling.py --out results/2026-10-29_xsuite_gpu/p0.json
"""
import argparse
import json
import os
import pathlib
import sys
import time

# cupy's CUB reduction kernels (used by xpart's allclose consistency checks for >= ~500 particles) fail to compile under this NVRTC
# (NVRTC_ERROR_COMPILATION, 29 October); turning the accelerator off changes only those reductions, not the tracking kernels.
os.environ['CUPY_ACCELERATORS'] = ''

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'workers'))
import common  # noqa: E402

NS = (5, 50, 500, 5000, 50000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default='cuda-f64-yosk8b16-exact')
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    import w_xsuite as w

    cfg = next(c for c in w.CONFIGS if c['name'] == a.config)
    t0 = time.perf_counter()
    w.setup()
    t_import = time.perf_counter() - t0
    case0 = common.load(HERE / 'cases/ring_scale_n5.json')
    lat = common.load(HERE / 'cases' / case0['lattice'])
    t0 = time.perf_counter()
    model, _ = w.build(lat, case0, cfg)
    w.sync(cfg['device'])
    t_build = time.perf_counter() - t0
    res = dict(config=cfg, versions=w.versions(), import_s=t_import, build_s=t_build, rows=[], host=common.provenance())
    for n in NS:
        case = common.load(HERE / f'cases/ring_scale_n{n}.json')
        init = common.launches(case)
        turns = int(case['turns'])

        def timed():
            w.sync(cfg['device'])
            t = time.perf_counter()
            r = w.track(model, init, turns, cfg)
            w.sync(cfg['device'])
            return r, time.perf_counter() - t

        try:
            first, t_cold = timed()
            warm = []
            for _ in range(a.repeats):
                last, dt = timed()
                warm.append(dt)
        except Exception as e:  # noqa: BLE001  keep what was measured; the failure is part of the record
            res['rows'].append(dict(n=n, error=repr(e)[:400]))
            print(f'n={n} failed: {e!r}'[:300], flush=True)
            break
        row = dict(n=n, turns=turns, cold_s=t_cold, warm_s=warm, warm_median_s=float(np.median(warm)),
                   per_particle_turn_us=1e6 * float(np.median(warm)) / (n * turns), finite=bool(np.isfinite(last).all()),
                   first5_final_state=last[:5, -1, :4].tolist(), repeat_maxdiff=float(np.nanmax(np.abs(first - last))))
        res['rows'].append(row)
        print(f"n={n:6d} cold {t_cold:.3f} warm {row['warm_median_s']:.3f} s  {row['per_particle_turn_us']:.3g} us/particle-turn", flush=True)
    res['cupy_accelerators'] = os.environ['CUPY_ACCELERATORS']
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()

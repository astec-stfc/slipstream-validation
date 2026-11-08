"""Where does an Xsuite GPU call spend its time?  Splits the adapter's ``track`` into its three parts (7 November 2026).

The paper reports Xsuite on the GPU in float64 at 1.6 to 1.8 times Slipstream's time for every particle count, with cupy's CUB reduction accelerator disabled
(its NVRTC compile fails for >= ~500 particles; see xsuite_gpu_scaling.py).  The adapter's call is  build_particles  ->  line.track(turn_by_turn_monitor=True)
->  host copy of the monitor.  xpart's constructor runs cupy allclose/reduction checks, the only place the accelerator matters.  This script times each part
separately, plus a track without the monitor, so the CUB question and the monitor cost are measured rather than argued.

    CUPY_ACCELERATORS=          python xsuite_gpu_profile.py --out results/2026-11-07_xsuite_gpu_profile/cub_off.json      (the paper's setting)
    (unset: cupy default)       python xsuite_gpu_profile.py --out results/2026-11-07_xsuite_gpu_profile/cub_default.json --ns 5 50   (CUB compiles below ~500)

Run inside the WSL conda env of the Xsuite worker, e.g. via launch/conda_run.sh xsuite-gpu 1 [CUPY_ACCELERATORS=] -- xsuite_gpu_profile.py ...
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'workers'))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default='cuda-f64-yosk8b16-exact')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--ns', type=int, nargs='+', default=[5, 50, 500, 5000, 50000])
    a = ap.parse_args()
    import w_xsuite as w

    cfg = next(c for c in w.CONFIGS if c['name'] == a.config)
    w.setup()
    case0 = common.load(HERE / 'cases/ring_scale_n5.json')
    lat = common.load(HERE / 'cases' / case0['lattice'])
    t0 = time.perf_counter()
    model, inv = w.build(lat, case0, cfg)
    w.sync(cfg['device'])
    res = dict(config=cfg, versions=w.versions(), cupy_accelerators=os.environ.get('CUPY_ACCELERATORS', '<unset>'), build_s=time.perf_counter() - t0,
               n_elements=inv['n'], rows=[], host=common.provenance())
    line = model['line']
    to_np = lambda x: x.get() if hasattr(x, 'get') else np.asarray(x)  # noqa: E731
    for n in a.ns:
        case = common.load(HERE / f'cases/ring_scale_n{n}.json')
        init = common.launches(case)
        turns = int(case['turns'])
        rows = []
        try:
            for rep in range(a.repeats + 1):  # repeat 0 is cold
                w.sync(cfg['device'])
                t = time.perf_counter()
                p = line.build_particles(x=init[:, 0], px=init[:, 1], y=init[:, 2], py=init[:, 3], delta=init[:, 5], zeta=init[:, 4])
                w.sync(cfg['device']); t_bp = time.perf_counter() - t
                t = time.perf_counter()
                line.track(p, num_turns=turns, turn_by_turn_monitor=True)
                w.sync(cfg['device']); t_tr = time.perf_counter() - t
                t = time.perf_counter()
                m = line.record_last_track
                np.stack([to_np(m.x), to_np(m.px), to_np(m.y), to_np(m.py)], axis=-1).astype(float)
                t_cp = time.perf_counter() - t
                p2 = line.build_particles(x=init[:, 0], px=init[:, 1], y=init[:, 2], py=init[:, 3], delta=init[:, 5], zeta=init[:, 4])
                w.sync(cfg['device'])
                t = time.perf_counter()
                line.track(p2, num_turns=turns)
                w.sync(cfg['device']); t_nm = time.perf_counter() - t
                rows.append(dict(rep=rep, build_particles_s=t_bp, track_monitor_s=t_tr, host_copy_s=t_cp, track_no_monitor_s=t_nm))
        except Exception as e:  # noqa: BLE001
            res['rows'].append(dict(n=n, error=repr(e)[:300], partial=rows))
            print(f'n={n} failed: {e!r}'[:260], flush=True)
            break
        warm = rows[1:]
        med = {k: float(np.median([r[k] for r in warm])) for k in warm[0] if k != 'rep'}
        res['rows'].append(dict(n=n, turns=turns, cold=rows[0], warm=warm, warm_median=med, call_total_s=med['build_particles_s'] + med['track_monitor_s'] + med['host_copy_s']))
        print(f"n={n:6d} build_particles {med['build_particles_s']:.4f}  track(monitor) {med['track_monitor_s']:.4f}  host copy {med['host_copy_s']:.4f}  track(no monitor) {med['track_no_monitor_s']:.4f} s", flush=True)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()

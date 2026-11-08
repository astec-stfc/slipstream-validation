"""SciBmad particle-count scaling on one thread of the CPU and on the GPU (7 November 2026; counterpart of xsuite_gpu_scaling.py).

One Julia process per device and precision, so the kernel JIT (about two minutes on the GPU) is paid once; the line is built once and the
``ring_scale_n*`` launches (5 ... 50 000 particles, 64 turns) are tracked through the adapter's own ``track`` (device-to-host copy included),
one cold and ``--repeats`` warm calls per particle count.

    python scibmad_scaling.py --device cuda --dtype float64 --out results/2026-11-07_scibmad_scaling/cuda-f64.json
    python scibmad_scaling.py --device cpu  --dtype float64 --out results/2026-11-07_scibmad_scaling/cpu-f64.json
    python scibmad_scaling.py --device cpu --dtype float64 --order 4 --steps 4 --out results/2026-11-07_scibmad_scaling/cpu-f64-o4k4.json   (accuracy-matched rung)
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'workers'))
import common  # noqa: E402

NS = (5, 50, 500, 5000, 50000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    ap.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    ap.add_argument('--out', required=True)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--order', type=int, default=6)
    ap.add_argument('--steps', type=int, default=8, help='steps per quadrupole and sextupole')
    a = ap.parse_args()
    import w_scibmad as w

    w._wants_cuda = lambda: a.device == 'cuda'   # setup() normally reads the job file
    cfg = dict(name=f"{a.device}-f{a.dtype[-2:]}-o{a.order}k{a.steps}", device=a.device, dtype=a.dtype, method='sym', order=a.order, n_steps=a.steps)
    t0 = time.perf_counter()
    w.setup()
    t_import = time.perf_counter() - t0
    case0 = common.load(HERE / 'cases/ring_scale_n5.json')
    lat = common.load(HERE / 'cases' / case0['lattice'])
    t0 = time.perf_counter()
    model, _ = w.build(lat, case0, cfg)
    res = dict(config=cfg, versions=w.versions(), import_s=t_import, build_s=time.perf_counter() - t0, rows=[], host=common.provenance())
    for n in NS:
        case = common.load(HERE / f'cases/ring_scale_n{n}.json')
        init = common.launches(case)
        turns = int(case['turns'])
        try:
            times = []
            for _ in range(a.repeats + 1):
                t = time.perf_counter()
                out = w.track(model, init, turns, cfg)
                times.append(time.perf_counter() - t)
        except Exception as e:  # noqa: BLE001  keep what was measured
            res['rows'].append(dict(n=n, error=repr(e)[:400]))
            print(f'n={n} failed: {e!r}'[:300], flush=True)
            break
        warm = times[1:]
        row = dict(n=n, turns=turns, cold_s=times[0], warm_s=warm, warm_median_s=float(np.median(warm)),
                   per_particle_turn_us=1e6 * float(np.median(warm)) / (n * turns), finite=bool(np.isfinite(out).all()),
                   first5_final_state=out[:5, -1, :].tolist())
        res['rows'].append(row)
        print(f"n={n:6d} cold {times[0]:.2f} warm {row['warm_median_s']:.3f} s  {row['per_particle_turn_us']:.3g} us/particle-turn", flush=True)
    out_path = pathlib.Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()

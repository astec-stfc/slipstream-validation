"""CLARA field-map injector: bridge from the 23 October convergence study to the present workspace source (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, item 26).

The convergence study (reports/CLARA_Injector_Convergence) stored its package-default runs with ``sc: {}`` and was verified at commit c4c74a2.
This control re-runs the cathode-to-3.37 m injector at 20, 100 and 250 pC with the checkout in this workspace, two ways:

    default    ``SpaceChargeSettings()`` as the source now resolves them (what a user gets)
    explicit   every setting the study documents spelled out (emission_steps 100, dt_safety 3.5e5, dt_floor 15 fs, dt_ceiling 2 ps,
               subcycle_steps 20, no smoothing, 6 particles/cell, 12 cells per sigma_z, adaptive step), so the run does not move when a default does

and compares both with the study's stored default records (eps, sizes, longitudinal observables, steps, loop time) and with ASTRA (shipped; the
finest-step h = 0.175 ps run).  Three timed repeats per run, one process, GPU float32, 262 144 particles, nothing else on the GPU.
Run records go to results/<campaign>/ (the study's own caches are never written).

    SSV_WORKBENCH=<neg_coating_rwi checkout> SSV_SLIPSTREAM=<slipstream checkout>  (Windows also needs CXX=clang-cl)
    python clara_release_bridge.py --campaign 2026-10-29_clara_bridge
"""
import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

REPO = common.WORKBENCH  # SSV_WORKBENCH: the neg_coating_rwi checkout
sys.path.insert(0, str(REPO))

CHARGES = ('20pC', '100pC', '250pC')
EXPLICIT = dict(smooth_r=0, smooth_z=0, emission_steps=100, dt_floor_s=1.5e-14, dt_safety=3.5e5, dt_ceiling_s=2e-12,
                subcycle_steps=20, min_particles_per_cell=6.0, cells_per_sigma_zs=12.0, dt_adaptive=True)
OBS = ('eps_x', 'eps_y', 'sigma_x', 'sigma_y', 'sigma_z', 'dE', 'eps_z', 'Ekin')
STORED = REPO / 'slipstream_dev/Stories/CLARA_InjectorConvergence/output/data/runs'


def git(*a):
    return subprocess.run(['git', '-C', str(REPO / 'slipstream'), *a], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', default='2026-10-29_clara_bridge')
    ap.add_argument('--charges', nargs='*', default=list(CHARGES))
    a = ap.parse_args()
    root = common.RESULTS / a.campaign
    from slipstream_dev.Stories.CLARA_InjectorConvergence import sweep
    from slipstream.tracking.native.injector import SpaceChargeSettings
    sweep.DATA = root / 'runs'
    sweep.warm_up()
    now = SpaceChargeSettings()
    same = {k: bool(getattr(now, k) == v) for k, v in EXPLICIT.items()}
    rows = []
    for charge in a.charges:
        old = json.loads((STORED / f'{charge}_default.json').read_text())
        for label, sc in (('default', {}), ('explicit', EXPLICIT)):
            r = sweep.run(charge, f'bridge_{label}', family='bridge', sc=sc, repeats=3)
            row = dict(charge=charge, setting=label, loops_s=r['loops_s'], wall_s=r['wall_s'], n_steps=r['n_steps'], n_alive=r['n_alive'],
                       stored_loops_s=old['loops_s'], stored_n_steps=old['n_steps'])
            for k in OBS:
                row[k] = r[k]
                row[f'{k}_astra'] = r[f'{k}_astra']
                row[f'{k}_stored'] = old[k]
                row[f'{k}_vs_stored'] = r[k] / old[k] - 1.0
                row[f'{k}_vs_astra'] = r[k] / r[f'{k}_astra'] - 1.0
            fine = sweep.astra_reference(charge, 'h0175')
            z = r['z_stop_m']
            for k, (_, ak) in sweep.SCORED.items():
                row[f'{k}_astra_h0175'] = sweep.at_stop(fine, ak, z)
                row[f'{k}_vs_astra_h0175'] = r[k] / row[f'{k}_astra_h0175'] - 1.0
            rows.append(row)
            print(charge, label, f"eps_x {r['eps_x']:.4f} (stored {old['eps_x']:.4f}, {row['eps_x_vs_stored']:+.1e}) steps {r['n_steps']} (stored {old['n_steps']}) loops {[round(x, 1) for x in r['loops_s']]}", flush=True)
    out = dict(design=dict(date='2026-10-29', particles=262144, dtype='float32', device='cuda', repeats=3, explicit_settings=EXPLICIT,
                           explicit_equals_present_default=same, note='native run records only; the study caches are untouched'),
               source=dict(slipstream_head=git('rev-parse', '--short', 'HEAD'), slipstream_dirty=bool(git('status', '--short')),
                           study_verified_commit='c4c74a2'),
               analysis_sha256=common.sha256(__file__), rows=rows)
    (common.EVIDENCE / '2026-10-26_R1/clara_release_bridge.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()

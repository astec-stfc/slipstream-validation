"""P1 controls: isolate diagnostic stride, then physical field-refresh interval."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import common
from analyse_sc import OBS, rel

HERE = Path(__file__).resolve().parent
CONFIGS = [f'cuda-f32-3d32t1c{c}-r{r}' for c, r in ((1, 50), (1, 100), (5, 100), (10, 100))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', default='2026-10-26_R1')
    a = ap.parse_args()
    name = a.campaign + '_injector_cost'
    root = common.RESULTS / name
    root.mkdir(exist_ok=True)
    design = dict(classification='post-baseline workload-specific performance control', configs=CONFIGS,
                  fixed='CUDA32 Cartesian 32^3, 1 ps push, common 20000 particles and 20 output stations',
                  procs=3, repeats=2, threads=1,
                  diagnostics_gate='Worst size/length/emittance/spread change <=1e-4 relative to record_every=5',
                  refresh_gate='Worst observable change <=1e-3 relative to c1-r100',
                  physics_gate='Original 3% size/length and 5% emittance; additional <=1% energy-spread error vs ASTRA32',
                  scope='Free-space 20 pC drift only; no gun/RF preset or converged-reference claim')
    (root / 'design.json').write_text(json.dumps(design, indent=2) + '\n')
    subprocess.run([sys.executable, '-u', str(HERE / 'run.py'), '--case', 'sc_drift', '--campaign', name,
                    '--codes', 'slipstream', '--configs', ','.join(CONFIGS), '--procs', '3', '--repeats', '2', '--threads', '1'], check=True)
    case = common.load(HERE / 'cases/sc_drift.json')
    primary = common.RESULTS / a.campaign / 'sc_drift'
    ref = np.load(primary / 'astra/cpu-f64-3d32h1.4/p0/tbt.npz')['tbt']
    old = np.load(common.RESULTS / (a.campaign + '_GPU_control') / 'sc_drift/slipstream/cuda-f32-3d32t1c1/p0/tbt.npz')['tbt']
    cadence_path = root / 'sc_drift/slipstream/cuda-f32-3d32t1c1-r100/p0/tbt.npz'
    cadence = np.load(cadence_path)['tbt'] if cadence_path.exists() else None
    rows = []
    for config in CONFIGS:
        jobs = []
        for p in sorted((root / 'sc_drift/slipstream' / config).glob('p*/result.json')):
            r = common.load(p)
            job = dict(result=r)
            if r['status'] == 'ok':
                assert r['case_sha256'] == common.sha256(HERE / 'cases/sc_drift.json')
                z = np.load(p.parent / 'tbt.npz')['tbt']
                job['versus_astra'] = {k: rel(z, ref, case['columns'], v) for k, v in OBS.items()}
                job['versus_record5'] = {k: rel(z, old, case['columns'], v) for k, v in OBS.items()}
                if cadence is not None:
                    job['versus_c1r100'] = {k: rel(z, cadence, case['columns'], v) for k, v in OBS.items()}
            jobs.append(job)
        good = [j for j in jobs if j['result']['status'] == 'ok']
        row = dict(config=config, n_ok=len(good), jobs=jobs)
        if good:
            row['warm_s'] = float(np.median([j['result']['timing']['warm_s']['median'] for j in good]))
            for field in ('versus_astra', 'versus_record5', 'versus_c1r100'):
                row[field] = {k: max(j[field][k] for j in good if field in j) for k in OBS} if all(field in j for j in good) else None
            errors = row['versus_astra']
            row['physics_pass'] = errors['size'] <= .03 and errors['length'] <= .03 and errors['emit'] <= .05 and errors['spread'] <= .01
            diagnostic_only = '-3d32t1c1-' in config
            parity = row['versus_record5'] if diagnostic_only else row['versus_c1r100']
            row['parity_pass'] = parity is not None and max(parity.values()) <= (1e-4 if diagnostic_only else 1e-3)
            row['pass'] = len(good) == 3 and row['physics_pass'] and row['parity_pass']
        rows.append(row)
    passing = [r for r in rows if r.get('pass')]
    selected = min(passing, key=lambda r: r['warm_s'])['config'] if passing else None
    destination = common.EVIDENCE / a.campaign / 'injector_cost_control.json'
    destination.write_text(json.dumps(dict(design=design, rows=rows, cheapest_passing=selected), indent=2) + '\n')
    print(destination, 'selected', selected, flush=True)


if __name__ == '__main__':
    main()

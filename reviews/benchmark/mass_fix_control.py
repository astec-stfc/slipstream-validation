"""Effect of writing the Slipstream adapter's deck momenta with the case mass (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P1).

The shared ASTRA deck writer converted the frozen beta*gamma to eV/c with ASTRA's 511003.0 eV electron mass.  ASTRA divides by that same mass,
so ASTRA receives the intended bunch, but Slipstream's deck reader divides by the CODATA mass and so received a bunch whose momenta were
7.9e-6 too high, a mean-energy offset of 43.3 eV at 5 MeV.  The ``-mfix`` settings write the deck with the case mass.  This script compares them
with the unchanged settings at 20 pC and at charge x 1e-6, and with ASTRA, for size, length, emittance, spread and mean-energy change, and
adds the particle-based ASTRA energy gain (astra_particle_energy_control.py) and the Gaussian field-energy estimate.

    python mass_fix_control.py
"""
import json
from pathlib import Path

import numpy as np

import common
from analyse_sc import OBS, rel
from charge_off_control import field_energy_gain_eV, mass_offset_eV

HERE = Path(__file__).resolve().parent
R = common.RESULTS
CASE20, CASE0 = HERE / 'cases/sc_drift.json', HERE / 'cases/sc_drift_q0.json'
PAIRS = {  # mfix setting -> (unchanged q20 result, unchanged q0 result)
    'cuda-f32-3d32t1c10-r100': (R / '2026-10-26_R1_injector_cost/sc_drift/slipstream/cuda-f32-3d32t1c10-r100', R / '2026-10-27_charge_off/sc_drift_q0/slipstream/cuda-f32-3d32t1c10-r100'),
    'cpu-f64-3d32t1c1': (R / '2026-10-26_R1/sc_drift/slipstream/cpu-f64-3d32t1c1', R / '2026-10-27_charge_off/sc_drift_q0/slipstream/cpu-f64-3d32t1c1')}
ASTRA20 = R / '2026-10-26_R1/sc_drift/astra/cpu-f64-3d32h1.4/p0/tbt.npz'
ASTRA0 = R / '2026-10-27_charge_off/sc_drift_q0/astra/cpu-f64-3d32h1.4/p0/tbt.npz'


def states(root):
    out = []
    for p in sorted(root.glob('p*/result.json')):
        r = common.load(p)
        if r['status'] == 'ok':
            out.append(np.load(p.parent / 'tbt.npz')['tbt'])
    return out


def main():
    case20, case0 = common.load(CASE20), common.load(CASE0)
    cols = case20['columns']
    ie = cols.index('ekin_eV')
    ref20, ref0 = np.load(ASTRA20)['tbt'], np.load(ASTRA0)['tbt']
    astra_energy = common.load(common.EVIDENCE / '2026-10-26_R1/astra_particle_energy.json')['rows']
    rows = []
    for name, (old20, old0) in PAIRS.items():
        row = dict(config=name, offset_prediction_eV=mass_offset_eV(case0['kinetic_eV'], case0['mass_eV']))
        for label, case, old, ref, root in (('q20', case20, old20, ref20, R / '2026-10-27_mass_fix/sc_drift/slipstream'),
                                            ('q0', case0, old0, ref0, R / '2026-10-27_mass_fix/sc_drift_q0/slipstream')):
            new = states(root / (name + '-mfix'))
            base = states(old)
            if not new:
                row[label] = dict(n_ok=0)
                continue
            row[label] = dict(
                n_ok=len(new),
                change_vs_unchanged={k: max(rel(a, b, cols, v) for a in new for b in base) for k, v in OBS.items()},
                gap_vs_astra={k: max(rel(a, ref, cols, v) for a in new) for k, v in OBS.items()},
                gap_unchanged_vs_astra={k: max(rel(a, ref, cols, v) for a in base) for k, v in OBS.items()},
                dEkin_new_eV=float(np.mean([a[-1, ie] - a[0, ie] for a in new])), dEkin_unchanged_eV=float(np.mean([a[-1, ie] - a[0, ie] for a in base])),
                field_energy_gain_eV=field_energy_gain_eV(new[0], cols, case['charge_C'], float(new[0][0, ie]), case['mass_eV']))
        rows.append(row)
    out = dict(design=dict(date='2026-10-27', procs=2, repeats=1, threads=1,
                           note='Adapter-level fix: deck momenta = case mass * beta*gamma for the Slipstream adapter only; ASTRA and other adapters unchanged'),
               astra_particle_energy=astra_energy, case_sha256=dict(q20=common.sha256(CASE20), q0=common.sha256(CASE0)),
               analysis_sha256=common.sha256(__file__), rows=rows)
    (common.EVIDENCE / '2026-10-26_R1/mass_fix_control.json').write_text(json.dumps(out, indent=2) + '\n')
    pc = lambda v: f'{100 * v:.3g}%'  # noqa: E731
    print('ASTRA particle-based gain [eV]:', {k: round(v['gain_eV'], 2) for k, v in astra_energy.items()}, '(Zemit table:', {k: v['zemit_table_gain_eV'] for k, v in astra_energy.items()}, ')')
    for r in rows:
        for label in ('q20', 'q0'):
            q = r.get(label, {})
            if not q.get('n_ok'):
                print(r['config'], label, 'no data')
                continue
            c, g = q['change_vs_unchanged'], q['gap_vs_astra']
            print(f"{r['config']:26s} {label:3s} change vs unchanged size {pc(c['size']):>9s} emit {pc(c['emit']):>9s} spread {pc(c['spread']):>9s} | gap vs ASTRA size {pc(g['size']):>8s} "
                  f"emit {pc(g['emit']):>8s} spread {pc(g['spread']):>8s} | dE {q['dEkin_unchanged_eV']:7.2f} -> {q['dEkin_new_eV']:7.2f} eV (field-energy est {q['field_energy_gain_eV']:.1f})")


if __name__ == '__main__':
    main()

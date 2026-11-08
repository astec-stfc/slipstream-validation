"""Particle-based ASTRA mean-energy gain for the sc_drift bunch (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P1).

ASTRA's ``Zemit`` mean-energy column is printed in 100 eV steps near 5 MeV, so it cannot resolve the ~30 eV space-charge energy gain of
the 20 pC drift.  This control runs the same ASTRA 3-D deck (32^3, 1.4 ps) at the case charge and at charge x 1e-6 and computes the mean
kinetic energy from the full-precision phase-space file written at ZSTOP (``Zphase``), against the input particles, with ASTRA's electron
mass.  Not a timing run; one process each.

    python astra_particle_energy_control.py
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'workers'))
import common  # noqa: E402
from astra_deck import ASTRA_MC2_EV, write_deck  # noqa: E402

EXE = pathlib.Path(os.environ.get('BENCH_ASTRA_EXE', 'Astra'))


def ekin_eV(table):
    """Mean kinetic energy of an ASTRA particle table (row 0 is the absolute reference particle)."""
    ref, p = table[0], table[1:]
    pz = p[:, 5] + ref[5]
    ptot = np.sqrt(p[:, 3] ** 2 + p[:, 4] ** 2 + pz ** 2)
    return float((np.sqrt(ptot ** 2 + ASTRA_MC2_EV ** 2) - ASTRA_MC2_EV).mean())


def run(case, n3d=32):
    d = pathlib.Path(tempfile.mkdtemp(prefix='bench_astra_energy_'))
    deck = write_deck(d, common.launches(case), case, 0, 0, 1.4e-3, True, n3d)
    r = subprocess.run([str(EXE), deck.name], cwd=d, capture_output=True, text=True, timeout=3600, input='\n')
    final = sorted(d.glob('sc_drift.[0-9][0-9][0-9][0-9].001'))
    if not final:
        raise RuntimeError((r.stdout + r.stderr)[-1500:])
    e0, e1 = ekin_eV(np.loadtxt(d / 'beam.ini')), ekin_eV(np.loadtxt(final[-1]))
    z = np.loadtxt(d / 'sc_drift.Zemit.001')
    return dict(phase_space_file=final[-1].name, n_particles=int(np.loadtxt(final[-1]).shape[0] - 1), ekin_initial_eV=e0, ekin_final_eV=e1,
                gain_eV=e1 - e0, zemit_table_gain_eV=float((z[-1, 2] - z[0, 2]) * 1e6))


def main():
    rows = {}
    meshes = [int(m) for m in sys.argv[1].split(',')] if len(sys.argv) > 1 else []
    jobs = [('sc_drift', 'sc_drift', 32), ('sc_drift_q0', 'sc_drift_q0', 32)] + [(f'sc_drift_3d{n}', 'sc_drift', n) for n in meshes]
    for key, name, n3d in jobs:
        case = common.load(HERE / 'cases' / f'{name}.json')
        rows[key] = dict(charge_C=case['charge_C'], n3d=n3d, **run(case, n3d))
        print(key, rows[key], flush=True)
    out = dict(design=dict(date='2026-10-27', deck='ASTRA 3-D, 1.4 ps; meshes in each row (n3d)', mass_eV=ASTRA_MC2_EV,
                           note='Mean kinetic energy from full-precision particle files at ZSTOP against the input file; fixed z, not fixed t'),
               analysis_sha256=common.sha256(__file__), rows=rows)
    (common.EVIDENCE / '2026-10-26_R1/astra_particle_energy.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()

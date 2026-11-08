"""Declare and evaluate longitudinal gates for ``sc_drift`` (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P1).

These gates are RETROSPECTIVE: they were declared after the stored runs and the mesh-convergence work were known, so their thresholds are not
prospective design gates (an unchanged case checksum preserves the original design, not the prospectivity of these thresholds).

The frozen case file gates transverse size, bunch length and emittance only (3% / 3% / 5%).  Energy spread and mean-energy gain were reported but
never gated.  The gates below are declared here, not in cases/sc_drift.json, because editing the case would change its checksum and invalidate the
provenance of every stored run; they are evaluated against the finest effective ASTRA mesh (128^3), which the mesh-convergence work (items 14 and
the Richardson control) shows is the better of ASTRA's references.

  G_spread  worst station-wise relative difference of the rms energy spread from ASTRA 128^3  <= 5%   (the emittance gate).  The measured 128^3 -> limit
            error of ASTRA is 0.1% and native 64^3 differs from ASTRA 128^3 by 0.6%, so 5% is a loose gate on purpose: the charge-off control showed the
            solver-to-solver differences are real space-charge-model differences, which is what the gate is meant to expose.
  G_gain    |mean-energy gain - 30.98 eV| <= 5 eV.  30.98 eV is the gain from ASTRA 128^3's full-precision particle file; 5 eV is ASTRA's own spread over its
            32^3/64^3/128^3 results (35.45, 27.85, 30.98 eV: largest deviation 4.47 eV), i.e. the resolution of the reference, so no tighter gate is meaningful.
            The gain of a code is read from its own station table (last station minus first); ASTRA's 100 eV-quantised table cannot serve, so ASTRA is
            judged from its particle files and its cylindrical setting (table only) is not assessable.

The verdicts are insensitive to the exact thresholds: the output lists the interval of thresholds that gives the same pass/fail list as the declared gate.

    python longitudinal_gates.py
"""
import csv
import json
from pathlib import Path

import numpy as np

import common
from analyse_sc import rel
from convergence_extension import ASTRA, CASE, NATIVE, NEWC, states

HERE = Path(__file__).resolve().parent
PRIMARY = common.RESULTS / '2026-10-26_R1/sc_drift'
SPREAD_GATE, GAIN_GATE_EV = 0.05, 5.0


def astra_gains():
    """Particle-file gain by effective mesh: 32 and 64 from astra_particle_energy_control.py, 128 from the harness's own energy_gain.json files."""
    ae = common.load(common.EVIDENCE / '2026-10-26_R1/astra_particle_energy.json')['rows']
    g = {v['n3d']: v['gain_eV'] for k, v in ae.items() if k != 'sc_drift_q0' and v['n3d'] != 96}
    g[128] = float(np.mean([common.load(p)['gain_eV'] for p in sorted(ASTRA[128].glob('p*/energy_gain.json'))]))
    return g


def tables(root):
    return states(root)[0]


def main():
    case = common.load(CASE)
    cols = case['columns']
    ie = cols.index('ekin_eV')
    ref = tables(ASTRA[128])[0]
    ag = astra_gains()
    ref_gain = ag[128]
    rows = []

    def add(code, config, family, t, gain_source='table', gain=None):
        sp = rel(t, ref, cols, ('sigma_E_eV',))
        g = float(t[-1, ie] - t[0, ie]) if gain is None else gain
        rows.append(dict(code=code, config=config, family=family, spread_vs_astra128=sp, gain_eV=g, gain_dev_eV=g - ref_gain, gain_source=gain_source,
                         pass_spread=bool(sp <= SPREAD_GATE), pass_gain=bool(abs(g - ref_gain) <= GAIN_GATE_EV)))

    for d in sorted(PRIMARY.glob('*/*')):
        if not d.is_dir():
            continue
        ts = tables(d)
        if not ts:
            continue
        code, config = d.parent.name, d.name
        if code == 'astra':
            n3d = {'cpu-f64-3d32h1.4': 32}.get(config)
            if n3d:
                add(code, config, 'primary', ts[0], 'particle file', ag[n3d])
            else:  # cylindrical ASTRA: only the 100 eV-quantised table exists, which cannot resolve a 30 eV gain
                add(code, config, 'primary', ts[0], 'table (100 eV quantised): gain gate not assessable', None)
                rows[-1].update(pass_gain=None)
        else:
            add(code, config, 'primary', ts[0])
    for n, d in ASTRA.items():
        if n in (32, 96) or not d.exists():  # 32^3 is the primary row above; 96^3 duplicates 128^3
            continue
        ts = tables(d)
        if ts:
            add('astra', d.name, 'ladder', ts[0], 'particle file', ag[n])
    for n, d in NATIVE.items():
        if n == 16 or not d.exists():
            continue
        ts = tables(d)
        if ts:
            add('slipstream', d.name, 'ladder (mass-fixed)', ts[0])
    out = dict(design=dict(spread_gate=SPREAD_GATE, gain_gate_eV=GAIN_GATE_EV, gain_reference_eV=ref_gain, reference='ASTRA cpu-f64-3d128h1.4 (station table for spread, particle file for gain)',
                           note='Declared in this script, not in the frozen case file; the original case gates (size 3%, length 3%, emittance 5%) are unchanged.'),
               case_sha256=common.sha256(CASE), analysis_sha256=common.sha256(__file__), rows=rows)
    sp = [r['spread_vs_astra128'] for r in rows if r['pass_spread']], [r['spread_vs_astra128'] for r in rows if not r['pass_spread']]
    gd = [abs(r['gain_dev_eV']) for r in rows if r['pass_gain']], [abs(r['gain_dev_eV']) for r in rows if r['pass_gain'] is False]
    out['design']['insensitive_spread_gate_range'] = [max(sp[0]), min(sp[1])]
    out['design']['insensitive_gain_gate_range_eV'] = [max(gd[0]), min(gd[1])]
    (common.EVIDENCE / '2026-10-26_R1/longitudinal_gates.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f"unchanged verdicts for spread gates in {[round(100 * v, 2) for v in out['design']['insensitive_spread_gate_range']]} % and gain gates in {[round(v, 2) for v in out['design']['insensitive_gain_gate_range_eV']]} eV")
    print(f'reference gain {ref_gain:.2f} eV; gates: spread <= {100 * SPREAD_GATE:g}%, |gain - ref| <= {GAIN_GATE_EV:g} eV')
    print(f"{'code':11s} {'config':34s} {'family':20s}  spread vs A128   gain [eV]  dev [eV]   spread  gain")
    for r in rows:
        print(f"{r['code']:11s} {r['config']:34s} {r['family']:20s}  {100 * r['spread_vs_astra128']:9.2f}%   {r['gain_eV']:9.2f}  {r['gain_dev_eV']:+9.2f}   {'pass' if r['pass_spread'] else 'FAIL':6s}  {'n/a' if r['pass_gain'] is None else 'pass' if r['pass_gain'] else 'FAIL'}")


if __name__ == '__main__':
    main()

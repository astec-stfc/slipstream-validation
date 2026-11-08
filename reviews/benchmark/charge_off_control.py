"""Charge-off control for the sc_drift comparison (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P1).

The same frozen bunch, stations and adapters with the charge scaled by 1e-6 (cases/sc_drift_q0.json): space charge is negligible, so any
remaining difference between codes comes from diagnostics, coordinate frames, time-versus-position sampling or kinematics, not from a
space-charge solver.  Each code's q0 result is compared with ASTRA's q0 result; the 20 pC gap of the same setting is shown beside it, and
so is how much each code's own observables move between q0 and 20 pC (the size of the space-charge effect).

    python charge_off_control.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

import common
from analyse_sc import OBS, rel

HERE = Path(__file__).resolve().parent
CASE0, CASE20 = HERE / 'cases/sc_drift_q0.json', HERE / 'cases/sc_drift.json'
ROOT0 = common.RESULTS / '2026-10-27_charge_off/sc_drift_q0'
R1 = common.RESULTS / '2026-10-26_R1/sc_drift'
COST = common.RESULTS / '2026-10-26_R1_injector_cost/sc_drift'
SETTINGS = [('astra', 'cpu-f64-3d32h1.4'), ('slipstream', 'cuda-f32-3d32t1c10-r100'), ('slipstream', 'cpu-f64-3d32t1c1'),
            ('slipstream', 'cuda-f32-r32z128t2c5'), ('cheetah', 'cpu-f64-g32s2'), ('ocelot', 'cpu-f64-g32s2'), ('gpt', 'cpu-f64-mesh-acc6'),
            ('impactx', 'cpu-f64-mg32x40s2')]


K_E, E_CHARGE = 8.9875517923e9, 1.602176634e-19


def gaussian_field_energy_J(charge_C, sx, sy, sz):
    """Electrostatic energy of a Gaussian charge ellipsoid: Q^2 k_e / (4 sqrt(pi)) * int_0^inf ds / sqrt(prod(sigma_i^2 + s))."""
    f = lambda s: 1.0 / np.sqrt((sx**2 + s) * (sy**2 + s) * (sz**2 + s))  # noqa: E731
    return charge_C**2 * K_E / (4 * np.sqrt(np.pi)) * quad(f, 0, np.inf, epsabs=0, epsrel=1e-10, limit=500)[0]


def field_energy_gain_eV(table, cols, charge_C, ekin0_eV, mass_eV):
    """Mean kinetic-energy gain per particle from the release of rest-frame field energy, assuming the Gaussian shape is kept:
    gamma_b * (W0 - Wf) / N in the lab frame (rest-frame longitudinal size = gamma * sigma_z).  A rough independent expectation, not a reference."""
    gam = 1.0 + ekin0_eV / mass_eV
    i = [cols.index(k) for k in ('sigma_x_m', 'sigma_y_m', 'sigma_z_m')]
    w0, wf = (gaussian_field_energy_J(charge_C, table[k, i[0]], table[k, i[1]], gam * table[k, i[2]]) for k in (0, -1))
    return gam * (w0 - wf) / E_CHARGE / (abs(charge_C) / E_CHARGE)


def mass_offset_eV(ekin_eV, mass_eV, astra_mass_eV=511003.0):
    """Mean-energy offset from reading deck momenta (eV/c, written with ASTRA's electron mass) with the CODATA mass: beta * p c * (m_ASTRA/m - 1)."""
    e = ekin_eV + mass_eV
    pc = np.sqrt(e**2 - mass_eV**2)
    return float(pc / e * pc * (astra_mass_eV / mass_eV - 1.0))


def q20_path(code, config):
    for root in (R1, COST):
        p = root / code / config / 'p0/tbt.npz'
        if p.exists():
            return p
    raise FileNotFoundError(f'{code}/{config}')


def main():
    case0, case20 = common.load(CASE0), common.load(CASE20)
    assert case0['columns'] == case20['columns'] and abs(case0['charge_C'] / case20['charge_C'] - 1e-6) < 1e-12
    cols = case0['columns']
    ref0 = np.load(ROOT0 / 'astra/cpu-f64-3d32h1.4/p0/tbt.npz')['tbt']
    ref20 = np.load(q20_path('astra', 'cpu-f64-3d32h1.4'))['tbt']
    ie = cols.index('ekin_eV')
    rows = []
    offset = mass_offset_eV(case0['kinetic_eV'], case0['mass_eV'])
    for code, config in SETTINGS:
        jobs, states = [], []
        for p in sorted((ROOT0 / code / config).glob('p*/result.json')):
            r = common.load(p)
            job = dict(result_sha256=common.sha256(p), status=r['status'], warm_s=r.get('timing', {}).get('warm_s', {}).get('median'))
            if r['status'] == 'ok':
                assert r['case_sha256'] == common.sha256(CASE0)
                st = np.load(p.parent / 'tbt.npz')['tbt']
                assert st.shape == ref0.shape and np.isfinite(st).all()
                states.append(st)
                job['artifact_sha256'] = common.sha256(p.parent / 'tbt.npz')
            jobs.append(job)
        if not states:
            rows.append(dict(code=code, config=config, n_recorded=len(jobs), n_ok=0, jobs=jobs))
            continue
        s20 = np.load(q20_path(code, config))['tbt']
        row = dict(code=code, config=config, n_recorded=len(jobs), n_ok=len(states), jobs=jobs,
                   q0_vs_astra_q0={k: max(rel(s, ref0, cols, v) for s in states) for k, v in OBS.items()},
                   q20_vs_astra_q20={k: rel(s20, ref20, cols, v) for k, v in OBS.items()},
                   own_q0_vs_q20={k: max(rel(s, s20, cols, v) for s in states) for k, v in OBS.items()},
                   spread_q0_eV=float(states[0][-1, cols.index('sigma_E_eV')]), spread_q20_eV=float(s20[-1, cols.index('sigma_E_eV')]),
                   dEkin_q0_eV=float(states[0][-1, ie] - states[0][0, ie]), dEkin_q20_eV=float(s20[-1, ie] - s20[0, ie]),
                   field_energy_gain_eV=field_energy_gain_eV(s20, cols, case20['charge_C'], float(s20[0, ie]), case20['mass_eV']),
                   run_to_run_q0={k: max(rel(a, b, cols, v) for a in states for b in states) for k, v in OBS.items()})
        rows.append(row)
    out = dict(mass_convention_offset_eV=offset, design=dict(date='2026-10-27', procs=2, repeats=1, threads=1, charge_scale=1e-6,
                           note='Control: negligible space charge. Differences from ASTRA at q0 are not space-charge-solver differences. No pass gate.'),
               case_sha256=common.sha256(CASE0), analysis_sha256=common.sha256(__file__), rows=rows)
    (common.EVIDENCE / '2026-10-26_R1/charge_off_control.json').write_text(json.dumps(out, indent=2) + '\n')
    pc = lambda v: f'{100 * v:.3g}%'  # noqa: E731
    print(f'predicted deck-momentum mass-convention offset: {offset:.2f} eV')
    print(f"{'code':11s} {'config':26s} n  gap@q0 size/len/emit/spread                       gap@20pC spread   own q0->20pC spread   dE q0/q20 [eV]")
    for r in rows:
        if not r['n_ok']:
            print(r['code'], r['config'], 'no ok jobs')
            continue
        a, b, c = r['q0_vs_astra_q0'], r['q20_vs_astra_q20'], r['own_q0_vs_q20']
        print(f"{r['code']:11s} {r['config']:26s} {r['n_ok']}  {pc(a['size']):>8s} {pc(a['length']):>8s} {pc(a['emit']):>8s} {pc(a['spread']):>8s}"
              f"       {pc(b['spread']):>8s}            {pc(c['spread']):>8s}          {r['dEkin_q0_eV']:8.2f} {r['dEkin_q20_eV']:8.2f}   field-energy estimate {r['field_energy_gain_eV']:.1f} eV")


if __name__ == '__main__':
    main()

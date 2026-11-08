"""Unit-invariant, component-wise acceptance for the ring_grad Jacobians (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P2).

The inherited metrics (``rel_fro``, ``rel_col`` in analyse_grad.py) use raw SI-valued rows: the same physical difference gives a
different number if x is quoted in mm or px in mrad, and one large entry can hide a wrong small one.  Here every entry is made
dimensionless, S_ij = theta_j * dy_i/dtheta_j / sigma_i, with theta_j the nominal family strength and sigma_i the rms of coordinate i
over the first 40 turns of the corrected-elegant trajectory of the same launch.  S_ij is the fractional change of output i (relative to
its own amplitude) per unit fractional change of knob j; it does not change with units.

Limits declared in this file before the analysis was run (physical budget, not fitted to any code):
    e_abs = max_ij |dS_ij| / max_ij |S_ref|              <= 1e-3   (worst entry against the largest sensitivity)
    e_rel = max over significant entries |dS_ij|/|S_ref|  <= 1e-2   (significant: |S_ref,ij| >= 1e-3 * max|S_ref|, the zero-derivative floor)
Entries below the floor are bounded only by e_abs.
Because the endpoint state itself differs between codes with different element models, the Jacobian gap is also reported next to the
trajectory gap, traj_gap = max_i |y_i - y_ref,i| / sigma_i; e_abs / traj_gap of order one means the derivative is consistent with the
code's own (different) trajectory rather than separately wrong.  The reference is the corrected-elegant finest central-difference Jacobian.

    python gradient_acceptance.py                       # writes ../data/2026-10-26_R1/gradient_acceptance.json
    python gradient_acceptance.py --case ring_grad_b    # second launch and knob set (coverage extension), gradient_acceptance_b.json

For ring_grad_b the limits are unchanged and the amplitude scales sigma_i come from the exact native trajectory of the same launch (the scale
needs no more than 1e-3 relative accuracy, and the corrected elegant tracking of this launch was not recorded).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import common
from analyse_grad import best_fd, errs

HERE = Path(__file__).resolve().parent
OLD = common.RESULTS / '2026-10-26_R1'
NEW = common.RESULTS / '2026-10-27_momentum'
GRAD_B = common.RESULTS / '2026-10-28_grad_b'
SCIBMAD = common.RESULTS / '2026-11-07_scibmad'   # SciBmad, run 7 November 2026 (replaces Bmad-X in the paper)
LIMITS = dict(e_abs=1e-3, e_rel=1e-2, floor=1e-3)
UNIT_VARIANTS = {'SI (m, rad)': (1., 1., 1., 1.), 'mm, mrad': (1e3, 1e3, 1e3, 1e3), 'x,y in mm; px,py in rad': (1e3, 1., 1e3, 1.),
                 'x,y in um; px,py in mrad': (1e6, 1e3, 1e6, 1e3)}


def ratio(values):
    """max/min of the same metric over the unit conventions (1 when the metric is exactly zero, i.e. the reference compared with itself)."""
    return max(values) / min(values) if min(values) > 0 else 1.0


def scaled(J, k0, sigma):
    return J * k0[None, :] / sigma[:, None]


def metrics(S, Sref, limits=LIMITS):
    d = np.abs(S - Sref)
    smax = np.abs(Sref).max()
    sig = np.abs(Sref) >= limits['floor'] * smax
    e_abs = float(d.max() / smax)
    e_rel = float((d[sig] / np.abs(Sref[sig])).max())
    return dict(e_abs=e_abs, e_rel=e_rel, n_below_floor=int((~sig).sum()),
                passed=bool(e_abs <= limits['e_abs'] and e_rel <= limits['e_rel']))


def raw_metrics_in_units(J, Jref, units):
    u = np.asarray(units)[:, None]
    return errs(J * u, Jref * u)


def native_sigma(case):
    """rms of (x, px, y, py) over the first ``turns`` turns of the exact native CPU float64 trajectory of the case's launch."""
    import torch
    from slipstream.tracking.native.lattice_import import ElementSpec
    from slipstream.tracking.native.ring import RingMap

    lat = common.load(HERE / 'cases' / case['lattice'])
    specs = [ElementSpec(e['name'], e['kind'], e['length'], angle=e.get('angle', 0.0), k1=e.get('k1', 0.0), k2=e.get('k2', 0.0)) for e in lat['elements']]
    ring = RingMap(specs, energy_eV=case['energy_eV'], mc2_eV=case['mass_eV'], device='cpu', dtype=torch.float64, n_kicks=8, n_kicks_bend=16, integration_order=6)
    s = torch.as_tensor(common.launches(case)[:1], dtype=torch.float64)
    rec = []
    with torch.no_grad():
        for _ in range(case['turns']):
            rec.append(s[0, :4].clone())
            s = ring.turn(s)
    return torch.stack(rec).pow(2).mean(0).sqrt().numpy()


def load_jobs(roots=None):
    """(code, config, path-of-grad.npz) for every ring_grad process: primary campaign plus the two controls that re-ran a code."""
    roots = roots or [OLD / 'ring_grad', common.RESULTS / '2026-10-26_R1_Xsuite_exact/ring_grad', common.RESULTS / '2026-10-26_R1_CPU_control/ring_grad',
                      SCIBMAD / 'ring_grad']
    out = []
    for root in roots:
        for p in sorted(root.glob('*/*/p*/grad.npz')):
            res = common.load(p.parent / 'result.json')
            if res['status'] == 'ok':
                out.append((p.parts[-4], p.parts[-3], p))
    return out


def main(case_id='ring_grad'):
    second = case_id != 'ring_grad'
    case = common.load(HERE / f'cases/{case_id}.json')
    ref_z = np.load((GRAD_B / f'{case_id}' if second else NEW / 'ring_grad') / 'elegant/cpu-f64-o6s16b32/p0/grad.npz')
    Jref, label = best_fd(ref_z['J_fd'], ref_z['steps'])
    k0 = ref_z['k0']
    if second:
        sigma = native_sigma(case)
    else:
        traj = np.load(NEW / 'ring_tunes/elegant/cpu-f64-o6s16b32/p0/tbt.npz')['tbt'][2, :case['turns']]   # launch x0 = 4 mm
        sigma = np.sqrt((traj[:, :4] ** 2).mean(axis=0))
    Sref = scaled(Jref, k0, sigma)
    rows = []
    for code, config, path in load_jobs([GRAD_B / case_id, SCIBMAD / case_id] if second else None):
        z = np.load(path)
        if not np.allclose(z['k0'], k0):
            raise ValueError(f'{path}: nominal strengths differ from the reference')
        modes = {'fd': best_fd(z['J_fd'], z['steps'])[0]}
        if 'J_ad' in z.files:
            modes['ad'] = z['J_ad']
        for mode, J in modes.items():
            m = metrics(scaled(J, k0, sigma), Sref)
            fro, col = errs(J, Jref)
            raw = {name: raw_metrics_in_units(J, Jref, u) for name, u in UNIT_VARIANTS.items()}
            m.update(code=code, config=config, mode=mode, proc=path.parent.name, raw_fro=fro, raw_col=col,
                     raw_fro_unit_spread=ratio([v[0] for v in raw.values()]),
                     raw_col_unit_spread=ratio([v[1] for v in raw.values()]),
                     traj_gap=float((np.abs(z['y0'] - ref_z['y0']) / sigma).max()))
            # invariance: rescaling the outputs by any unit factor must not change the scaled metric
            u = np.asarray(UNIT_VARIANTS['x,y in um; px,py in mrad'])
            m['e_abs_unit_change'] = metrics(scaled(J * u[:, None], k0, sigma * u), Sref * 1.0)['e_abs']
            rows.append(m)
    # worst of the three processes per (code, config, mode)
    summary = {}
    for r in rows:
        key = (r['code'], r['config'], r['mode'])
        s = summary.setdefault(key, dict(code=r['code'], config=r['config'], mode=r['mode'], n=0, e_abs=0., e_rel=0., passed=True, raw_fro=0.,
                                         raw_col=0., raw_fro_unit_spread=1., raw_col_unit_spread=1., traj_gap=0., n_below_floor=r['n_below_floor'], max_invariance_gap=0.))
        s['n'] += 1
        s['e_abs'], s['e_rel'] = max(s['e_abs'], r['e_abs']), max(s['e_rel'], r['e_rel'])
        s['passed'] &= r['passed']
        for k in ('raw_fro', 'raw_col', 'raw_fro_unit_spread', 'raw_col_unit_spread', 'traj_gap'):
            s[k] = max(s[k], r[k])
        s['e_over_traj'] = max(s.get('e_over_traj', 0.), r['e_abs'] / r['traj_gap'] if r['traj_gap'] > 0 else 0.)
        s['max_invariance_gap'] = max(s['max_invariance_gap'], abs(r['e_abs_unit_change'] - r['e_abs']))
    out = dict(design=dict(limits=LIMITS, unit_variants=list(UNIT_VARIANTS), reference=f'corrected elegant cpu-f64-o6s16b32 central difference {label}',
                           scales=dict(theta0=k0.tolist(), sigma=sigma.tolist(), sigma_note=('rms over the first 40 turns of the exact native CPU float64 trajectory of the launch' if second else 'rms over the first 40 turns of the corrected elegant trajectory, launch x0=4 mm'),
                           launch=case['particles'], knobs=[f"{k['family']}.{k['kind']}" for k in case['knobs']]),
                           note='Limits declared before the analysis; raw ranges show how the inherited norms move with the unit choice.'),
               case_sha256=common.sha256(HERE / f'cases/{case_id}.json'), analysis_sha256=common.sha256(__file__),
               Sref=Sref.tolist(), rows=list(summary.values()))
    dest = common.EVIDENCE / ('2026-10-26_R1/gradient_acceptance_b.json' if second else '2026-10-26_R1/gradient_acceptance.json')
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(f"sigma = {np.array2string(sigma, precision=3)}; max|S_ref| = {np.abs(Sref).max():.4g}")
    print(f"{'code':11s} {'config':26s} {'mode':4s} {'n':>1s} {'e_abs':>9s} {'e_rel':>9s} {'gate':>5s} {'traj_gap':>9s} {'raw_fro':>9s} x{'unit':>5s} {'raw_col':>9s} x{'unit':>5s} {'e/traj':>7s}")
    for s in summary.values():
        print(f"{s['code']:11s} {s['config']:26s} {s['mode']:4s} {s['n']:1d} {s['e_abs']:9.2e} {s['e_rel']:9.2e} {'PASS' if s['passed'] else 'FAIL':>5s}  "
              f"{s['traj_gap']:9.2e} {s['raw_fro']:9.2e} x{s['raw_fro_unit_spread']:5.2f} {s['raw_col']:9.2e} x{s['raw_col_unit_spread']:5.2f} {s['e_over_traj']:7.2f}  inv-gap {s['max_invariance_gap']:.1e}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='ring_grad')
    main(ap.parse_args().case)

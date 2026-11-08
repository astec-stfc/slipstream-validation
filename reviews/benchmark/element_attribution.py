"""Attribute the ring-model differences to single elements against an independent exact reference (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P1).

probe_elements.py saved each code's one-element maps for 135 states (x, px, y, py, delta spanning +-8 mm, +-1 mm, +-1%, momenta up to 0.01).  It
compared them with elegant, whose reference momentum was later found to be mismatched.  Here the quadrupole and sextupole maps are compared with an
independent DOP853 integration of the exact fixed-energy Hamiltonian

    H = -sqrt((1+delta)^2 - px^2 - py^2) + k1/2 (x^2 - y^2) + k2/6 (x^3 - 3 x y^2),

the drift with its closed form, and each code is judged by its own deviation from that reference, not from another code.  (The bend was treated
the same way in bend_attribution.py.)  The settings are the finest of each ladder, as probed.

    python element_attribution.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import common
from bend_attribution import sector

HERE = Path(__file__).resolve().parent
ROOT = common.RESULTS / '2026-10-26_R1/element_probes'
CODES = ('slipstream', 'elegant', 'xsuite', 'cheetah', 'jutrack', 'bmadx', 'scibmad')
ROOTS = {'scibmad': common.RESULTS / '2026-11-07_scibmad/element_probes'}   # SciBmad was probed on 7 November 2026 (replaces Bmad-X in the paper)


def exact_map(state, element):
    x, px, y, py, _, delta = state
    k1, k2, length = element.get('k1', 0.0), element.get('k2', 0.0), element['length']

    def rhs(_, u):
        x, px, y, py = u
        pz = np.sqrt((1 + delta) ** 2 - px ** 2 - py ** 2)
        return [px / pz, -k1 * x - k2 / 2 * (x ** 2 - y ** 2), py / pz, k1 * y + k2 * x * y]

    sol = solve_ivp(rhs, (0.0, length), [x, px, y, py], method='DOP853', rtol=1e-13, atol=1e-16)
    return sol.y[:, -1]


def main():
    ref_run = common.load(ROOT / 'elegant/result.json')
    initial = np.load(ROOT / 'elegant/maps.npz')['initial']
    rows, exact = [], {}
    for kind in ('drift', 'quad', 'sext'):
        element = ref_run['elements'][kind]
        exact[kind] = np.array([exact_map(s, element) for s in initial])
    bend = ref_run['elements']['bend']
    exact['bend'] = sector(initial, bend['length'], bend['angle'], 3e-14)
    on = initial[:, 5] == 0
    for code in CODES:
        rec = common.load(ROOTS.get(code, ROOT) / code / 'result.json')
        if rec['status'] != 'ok':
            rows.append(dict(code=code, status=rec['status']))
            continue
        z = np.load(ROOTS.get(code, ROOT) / code / 'maps.npz')
        for kind in ('drift', 'quad', 'sext', 'bend'):
            diff = np.abs(z[kind] - exact[kind])
            names = ('x_m', 'px', 'y_m', 'py')
            # Size of the element's own effect for context: the change of the state across the element
            effect = np.maximum(np.abs(exact[kind] - initial[:, :4]).max(axis=0), 1e-300)
            rows.append(dict(code=code, element=kind, config=rec['config']['name'], element_def={k: ref_run['elements'][kind].get(k) for k in ('length', 'k1', 'k2', 'angle')},
                             max_error=dict(zip(names, diff.max(axis=0).tolist())), on_momentum_max_error=dict(zip(names, diff[on].max(axis=0).tolist())),
                             relative_to_effect=dict(zip(names, (diff.max(axis=0) / effect).tolist())), effect=dict(zip(names, effect.tolist()))))
    out = dict(design=dict(reference='DOP853, rtol 1e-13 (3e-14 for the bend), exact fixed-energy Hamiltonian (drift, quadrupole, sextupole, sector bend)', states=int(len(initial)),
                           note='Per-element deviation of each code from an independent reference. The elegant probe used the original (mismatched) p_central; its corrected bend is in bend_attribution.json (3e-12)'),
               probe_sha256={c: common.sha256(ROOTS.get(c, ROOT) / c / 'maps.npz') for c in CODES if (ROOTS.get(c, ROOT) / c / 'maps.npz').exists()}, analysis_sha256=common.sha256(__file__), rows=rows)
    (common.EVIDENCE / '2026-10-26_R1/element_attribution.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f"{'code':11s} {'element':6s} {'config':20s}  max error x [m]  px          y [m]       py          (worst / element effect)")
    for r in rows:
        if 'max_error' not in r:
            print(r)
            continue
        e = r['max_error']
        print(f"{r['code']:11s} {r['element']:6s} {r['config']:20s}  {e['x_m']:.2e}  {e['px']:.2e}  {e['y_m']:.2e}  {e['py']:.2e}   {max(r['relative_to_effect'].values()):.2e}")


if __name__ == '__main__':
    main()

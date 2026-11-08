"""Compare a sector bend against independent ODE and explicit elegant settings."""
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'workers'))
import common
import w_elegant


def sector(initial, length, angle, rtol):
    """Integrate the exact fixed-energy sector Hamiltonian in path coordinate s."""
    h = angle / length
    n = len(initial)
    delta = initial[:, 5]

    def rhs(_, flat):
        state = flat.reshape(n, 4)
        x, px, _, py = state.T
        pz = np.sqrt((1 + delta)**2 - px**2 - py**2)
        return np.stack(((1+h*x)*px/pz, h*(pz-1-h*x), (1+h*x)*py/pz, np.zeros(n)), axis=1).ravel()

    result = solve_ivp(rhs, (0, length), initial[:, :4].ravel(), method='DOP853', rtol=rtol, atol=rtol * .01)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:, -1].reshape(n, 4)


def main():
    root = common.RESULTS / '2026-10-26_R1/element_probes'
    old = np.load(root / 'elegant/maps.npz')
    initial = old['initial']
    element = json.loads((root / 'elegant/result.json').read_text())['elements']['bend']
    case = common.load(HERE / 'cases/ring_tunes.json')
    case.update(turns=2, particles={key: initial[:, i].tolist() for key, i in
                (('x0', 0), ('px0', 1), ('y0', 2), ('py0', 3), ('delta', 5))})
    reference = sector(initial, element['length'], element['angle'], 3e-14)
    coarse = sector(initial, element['length'], element['angle'], 1e-11)
    native = np.load(root / 'slipstream/maps.npz')['bend']
    rows = [dict(name='ode_refinement', error=np.abs(coarse-reference).max(axis=0).tolist()),
            dict(name='slipstream_exact', error=np.abs(native-reference).max(axis=0).tolist()),
            dict(name='elegant_original', error=np.abs(old['bend']-reference).max(axis=0).tolist())]
    for slices in (4, 16, 32, 64):
        for edge in (0, 1):
            cfg = dict(order=6, n_slices=16, n_slices_bend=slices)
            model, _ = w_elegant.build(dict(elements=[element]), case, cfg)
            run_file = model['dir'] / 'run.ele'
            text = run_file.read_text().replace(f'p_central={model["p0c"] / model["m"]!r}',
                                                f'p_central_mev={model["p0c"] / 1e6!r}')
            run_file.write_text(text)
            lte = model['dir'] / 'ring.lte'
            text = lte.read_text().replace('E1=0,E2=0,', f'E1=0,E2=0,EDGE1_EFFECTS={edge},EDGE2_EFFECTS={edge},EXPAND_HAMILTONIAN=0,')
            lte.write_text(text)
            result = w_elegant.track(model, initial, 2, cfg)[:, 1]
            rows.append(dict(name='elegant_explicit', slices=slices, edge=edge,
                             error=np.abs(result-reference).max(axis=0).tolist(),
                             versus_original=np.abs(result-old['bend']).max(axis=0).tolist()))
    cfg = dict(order=6, n_slices=16, n_slices_bend=32)
    model, _ = w_elegant.build(dict(elements=[element]), case, cfg)
    result = w_elegant.track(model, initial, 2, cfg)[:, 1]
    rows.append(dict(name='elegant_p_central_matched', error=np.abs(result-reference).max(axis=0).tolist(),
                     versus_original=np.abs(result-old['bend']).max(axis=0).tolist()))
    dest = common.EVIDENCE / '2026-10-26_R1/bend_attribution.json'
    dest.write_text(json.dumps(dict(element=element, n=len(initial), rows=rows,
        method='DOP853 independent integration of exact fixed-energy sector Hamiltonian; error components x,px,y,py',
        elegant=w_elegant.versions(), source_sha256=common.sha256(__file__)), indent=2) + '\n')
    print(dest.read_text())


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()

"""Richardson extrapolation of the mesh ladders to a mesh-free limit per solver (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P1: reference).

ASTRA's effective 3-D meshes are 32^3, 64^3 and 128^3 and the finest still changes the emittance by about 1% from the previous one, so a
native-versus-ASTRA gap below that is inside the reference's own convergence error.  This control fits f(N) = f_inf + C N^-p to each solver's own
ladder, station by station, and compares the two extrapolated limits.  If two independently written solvers extrapolate to the same
limit, that limit is a better reference than either finite mesh.  Nothing here assumes the asymptotic range: the order p is solved from the data,
a fit is accepted only where the ladder is monotone, and the spread between alternative fits is reported as the extrapolation uncertainty.

  * ASTRA: three points (32, 64, 128), solved exactly.
  * Native Cartesian (mass-fixed, 1 ps push, 10 ps refresh): triples (32,64,128), (64,96,128), (32,64,96) solved exactly and a least-squares
    fit over (32, 64, 96, 128); the 16^3 mesh is outside the asymptotic range and excluded.
  * The same three-point fits are repeated for the 200 000-particle bunch of injector_scaling_control.py (ASTRA and native 32, 64, 128), where the particle
    noise is a factor of three smaller (key ``at_200k``).

    python richardson_reference.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares

import common
from convergence_extension import ASTRA, CASE, NATIVE, states

HERE = Path(__file__).resolve().parent
ROOT200K = common.RESULTS / '2026-10-28_inj_scaling/sc_drift_n200000'
OBS = {'sigma_x_m': 'size x', 'sigma_y_m': 'size y', 'sigma_z_m': 'length', 'eps_nx_m': 'emittance x', 'eps_ny_m': 'emittance y', 'sigma_E_eV': 'energy spread'}


def solve3(n, f):
    """Exact f(N) = f_inf + C N^-p through three points; returns (f_inf, p) or None when the ladder is not monotone-convergent."""
    (n1, n2, n3), (f1, f2, f3) = n, f
    d12, d23 = f2 - f1, f3 - f2
    if d12 == 0 or d23 == 0 or d12 * d23 <= 0 or abs(d23) >= abs(d12):
        return None
    target = d12 / d23
    g = lambda p: (n1 ** -p - n2 ** -p) / (n2 ** -p - n3 ** -p) - target  # noqa: E731
    try:
        p = brentq(g, 1e-3, 12.0)
    except ValueError:
        return None
    c = d23 / (n3 ** -p - n2 ** -p)
    return float(f3 - c * n3 ** -p), float(p)


def lsq(n, f):
    n, f = np.asarray(n, float), np.asarray(f, float)
    scale = np.ptp(f) or 1.0
    best = None
    for p0 in (0.5, 1, 2, 3):
        r = least_squares(lambda u: (u[0] + u[1] * n ** -u[2] - f) / scale, [f[-1], (f[0] - f[-1]) * n[0] ** p0, p0], bounds=([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 12]))
        if best is None or r.cost < best.cost:
            best = r
    return float(best.x[0]), float(best.x[2]), float(best.cost)


def runs(root, case_file):
    """Station tables of every ok process in a config directory (case checksum verified)."""
    tabs = []
    for p in sorted(root.glob('p*/result.json')):
        r = common.load(p)
        if r['status'] == 'ok':
            assert r['case_sha256'] == common.sha256(case_file), p
            tabs.append(np.load(p.parent / 'tbt.npz')['tbt'])
    return tabs


def ladder(paths, meshes):
    tabs = {n: states(paths[n])[0] for n in meshes}
    return {n: t[0] for n, t in tabs.items() if t}


def extrapolate(tabs, triples, fit_all=None):
    """Per observable at the last station and the worst relative profile deviation; returns a dict per observable."""
    cols = common.load(CASE)['columns']
    res = {}
    for name in OBS:
        i = cols.index(name)
        prof = {n: t[1:, i] for n, t in tabs.items()}  # drop station 0 (input)
        lim = []
        for tri in triples:
            sol = [solve3(tri, [prof[n][s] for n in tri]) for s in range(len(prof[tri[0]]))]
            lim.append(sol)
        ok = np.array([[x is not None for x in sol] for sol in lim])  # (triples, stations)
        mask = ok.all(0)  # stations where every triple is a monotone, convergent ladder
        row = dict(final_station=dict(finite={str(n): float(prof[n][-1]) for n in tabs},
                                      triples={str(t): (sol[-1][0], sol[-1][1]) if sol[-1] else None for t, sol in zip(triples, lim)}),
                   valid_stations=int(mask.sum()), n_stations=int(mask.size), monotone_all_stations=bool(mask.all()))
        if mask.any():
            L = np.array([[s[0] if s else np.nan for s in sol] for sol in lim])  # (triples, stations)
            row['limit_profile'] = np.where(mask, L.mean(0), np.nan).tolist()
            row['limit_spread_rel_max'] = float(np.nanmax(np.where(mask, np.abs(L - L.mean(0)) / L.mean(0), np.nan)))
        if fit_all:
            fa = lsq(fit_all, [prof[n][-1] for n in fit_all])
            row['final_station']['least_squares'] = dict(limit=fa[0], p=fa[1], cost=fa[2])
        res[name] = row
    return res


def compare(ast, nat, native_triples, fit_all, cols):
    """Extrapolate both ladders and compare the limits (maxima over the stations where both are convergent)."""
    out = dict(astra=extrapolate(ast, [(32, 64, 128)]), native=extrapolate(nat, native_triples, fit_all=fit_all), agreement={})
    for name in OBS:
        a, nn = out['astra'][name], out['native'][name]
        if a['valid_stations'] and nn['valid_stations']:
            la, ln = np.array(a['limit_profile']), np.array(nn['limit_profile'])
            m = ~np.isnan(la) & ~np.isnan(ln)
            fa128 = ast[128][1:, cols.index(name)]
            fn128 = nat[128][1:, cols.index(name)]
            r = lambda x, y: float(np.max(np.abs(x[m] - y[m]) / np.abs(y[m])))  # noqa: E731
            out['agreement'][name] = dict(stations_used=int(m.sum()), limit_gap_rel_max=r(ln, la), limit_gap_rel_final=float((ln[-1] - la[-1]) / la[-1]) if m[-1] else None,
                                          native128_vs_astra128_rel_max=r(fn128, fa128), native128_vs_astra_limit_rel_max=r(fn128, la),
                                          astra128_vs_astra_limit_rel_max=r(fa128, la), native128_vs_native_limit_rel_max=r(fn128, ln))
        else:
            out['agreement'][name] = dict(note='no station has a monotone convergent ladder in both solvers')
    return out


def show(tag, res):
    print(tag)
    print('observable      stations  ASTRA128->lim  native128->lim   limit gap native-ASTRA (max / final)   native128 vs ASTRA128   orders p: ASTRA | native triples | LSQ')
    for name, label in OBS.items():
        a, nn, ag = res['astra'][name], res['native'][name], res['agreement'][name]
        if 'limit_gap_rel_max' not in ag:
            print(f'{label:14s}  {ag["note"]}')
            continue
        ap = a['final_station']['triples']["(32, 64, 128)"]
        npv = [(v[1] if v else float('nan')) for v in nn['final_station']['triples'].values()]
        fin = f"{100 * ag['limit_gap_rel_final']:+.2f}%" if ag['limit_gap_rel_final'] is not None else 'n/a'
        print(f"{label:14s}  {ag['stations_used']:2d}/{a['n_stations']}   {100 * ag['astra128_vs_astra_limit_rel_max']:6.2f}%   {100 * ag['native128_vs_native_limit_rel_max']:6.2f}%      "
              f"{100 * ag['limit_gap_rel_max']:6.2f}% / {fin:>7s}                 {100 * ag['native128_vs_astra128_rel_max']:.2f}%      "
              f"{(ap[1] if ap else float('nan')):.2f} | {' '.join(f'{x:.2f}' for x in npv)} | {nn['final_station']['least_squares']['p']:.2f}")


def main():
    cols = common.load(CASE)['columns']
    out = dict(case_sha256=common.sha256(CASE), analysis_sha256=common.sha256(__file__),
               design=dict(note='Fits are per solver and per station; ASTRA 96^3 equals 128^3 and is not used; native 16^3 is excluded. Mean-energy gain is not extrapolated: '
                                'the ASTRA particle-file gains (35.45, 27.85, 30.98 eV at 32, 64, 128) are not monotone and the native station-table gain is quantised by float32 (0.5 eV). '
                                'Top-level astra/native/agreement are the original 20 000-particle bunch; at_200k repeats the three-point fits on 200 000 particles.'))
    r20 = compare(ladder(ASTRA, (32, 64, 128)), ladder(NATIVE, (32, 64, 96, 128)), [(32, 64, 128), (64, 96, 128), (32, 64, 96)], (32, 64, 96, 128), cols)
    out.update(r20)
    case200 = HERE / 'cases/sc_drift_n200000.json'
    a200 = {n: runs(ROOT200K / 'astra' / f'cpu-f64-3d{n}h1.4', case200)[0] for n in (32, 64, 128)}
    n200 = {n: runs(ROOT200K / 'slipstream' / f'cuda-f32-3d{n}t1c10-r100-mfix', case200)[0] for n in (32, 64, 128)}
    r200 = compare(a200, n200, [(32, 64, 128)], (32, 64, 128), cols)
    r200['case_sha256'] = common.sha256(case200)
    out['at_200k'] = r200
    (common.EVIDENCE / '2026-10-26_R1/richardson_reference.json').write_text(json.dumps(out, indent=2) + '\n')
    show('20 000 macroparticles', r20)
    show('200 000 macroparticles', r200)


if __name__ == '__main__':
    main()

"""Particle-count ladder for the space-charge drift (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P2; also tests whether the mesh-convergence order is limited by shot noise).

The primary drift case has 20 000 macroparticles, so mesh convergence at 128^3 (2 million cells) is probed with about 0.01 particles per cell.  This control
keeps the physical bunch (Gaussian, 5 MeV, 20 pC, same sigmas) and raises the number of macroparticles to 2 000, 200 000 and 1 000 000 (fresh frozen
sets; 20 000 is the original case).  For each N it reports warm time, the native-versus-ASTRA gap at matched mesh, and the native self-convergence between meshes.

    python injector_scaling_control.py --make-cases   # writes cases/sc_drift_n*.json and .npz (and the seed-control variants sc_drift_n<N>s1/s2)
    python injector_scaling_control.py                # analyses results/2026-10-28_inj_scaling plus the stored 20 000-particle runs
"""
import argparse
import json
from pathlib import Path

import numpy as np

import common
from analyse_sc import rel
from convergence_extension import ASTRA as ASTRA20K, METRICS, NATIVE as NATIVE20K

HERE = Path(__file__).resolve().parent
NS = (2000, 200000, 1000000)
ROOT = common.RESULTS / '2026-10-28_inj_scaling'
NATIVE_CFG = {n: f'cuda-f32-3d{n}t1c10-r100-mfix' for n in (32, 64, 128)}
ASTRA_CFG = {n: f'cpu-f64-3d{n}h1.4' for n in (32, 64, 128)}


SEED_NS = (2000, 20000, 200000)
SEED_COPIES = (1, 2)


def make_cases():
    """Same recipe as make_cases.py (centred zero-mean bunch, sigma_x 0.5 mm, sigma_z 1 mm, eps_n 0.3 um, delta 1e-3), different N and seed."""
    base = common.load(HERE / 'cases/sc_drift.json')
    mc2, ek = 0.51099895e6, 5.0e6
    bg0 = np.sqrt((1 + ek / mc2) ** 2 - 1)
    sx, sz, epsn, sdelta = 0.5e-3, 1.0e-3, 0.3e-6, 1e-3
    for n in NS:
        rng = np.random.default_rng(20261028 + n)
        x, y, z = rng.normal(0, sx, n), rng.normal(0, sx, n), rng.normal(0, sz, n)
        bgx, bgy = rng.normal(0, epsn / sx, n), rng.normal(0, epsn / sx, n)
        bgz = bg0 * (1 + rng.normal(0, sdelta, n))
        for a in (x, y, z, bgx, bgy, bgz):
            a -= a.mean() if a is not bgz else a.mean() - bg0
        np.savez_compressed(HERE / 'cases' / f'sc_drift_n{n}.npz', x=x, y=y, z=z, bgx=bgx, bgy=bgy, bgz=bgz)
        case = dict(base, id=f'sc_drift_n{n}', title=f'{base["title"]} ({n} macroparticles)', particles=dict(base['particles'], file=f'sc_drift_n{n}.npz'),
                    gate='control (particle-count ladder; no pass gate)')
        (HERE / 'cases' / f'sc_drift_n{n}.json').write_text(json.dumps(case, indent=1) + '\n')
    print('wrote', [f'sc_drift_n{n}.json' for n in NS])
    # seed control: independent bunches of the same size and distribution (native-only runs show how much of the order / gap trends is sampling)
    for n in SEED_NS:
        for k in SEED_COPIES:
            if (HERE / 'cases' / f'sc_drift_n{n}s{k}.json').exists():
                continue
            rng = np.random.default_rng(20261028 + n + 1000 * k)
            x, y, z = rng.normal(0, sx, n), rng.normal(0, sx, n), rng.normal(0, sz, n)
            bgx, bgy = rng.normal(0, epsn / sx, n), rng.normal(0, epsn / sx, n)
            bgz = bg0 * (1 + rng.normal(0, sdelta, n))
            for a in (x, y, z, bgx, bgy, bgz):
                a -= a.mean() if a is not bgz else a.mean() - bg0
            np.savez_compressed(HERE / 'cases' / f'sc_drift_n{n}s{k}.npz', x=x, y=y, z=z, bgx=bgx, bgy=bgy, bgz=bgz)
            case = dict(base, id=f'sc_drift_n{n}s{k}', title=f'{base["title"]} ({n} macroparticles, seed copy {k})', particles=dict(base['particles'], file=f'sc_drift_n{n}s{k}.npz'),
                        gate='control (seed control; no pass gate)')
            (HERE / 'cases' / f'sc_drift_n{n}s{k}.json').write_text(json.dumps(case, indent=1) + '\n')
    print('wrote seed copies', [f'sc_drift_n{n}s{k}' for n in SEED_NS for k in SEED_COPIES])


def runs(root, case_file):
    """Station tables and warm times of every ok process under ``root`` (a config directory)."""
    tabs, warm, cold = [], [], []
    sha = common.sha256(case_file)
    for p in sorted(root.glob('p*/result.json')):
        r = common.load(p)
        if r['status'] != 'ok':
            continue
        assert r['case_sha256'] == sha, p
        tabs.append(np.load(p.parent / 'tbt.npz')['tbt'])
        cold.append(r['timing']['cold_s'])
        if r['timing']['warm_s']:
            warm.append(r['timing']['warm_s']['median'])
    return tabs, warm, cold


def gap(a, b, cols):
    return {k: rel(a, b, cols, v) for k, v in METRICS.items()}


def seed_control(cols, rows):
    """Native 32/64/128 on independent bunches of the same size: observed order and the 128-from-64 change, with the main bunch included."""
    res = {}
    for n in SEED_NS:
        per_bunch = {}
        main_row = next((r for r in rows if r['n'] == n), None)
        if main_row and 'native' in main_row.get('observed_order', {}):
            per_bunch['main'] = dict(order_emit_x=main_row['observed_order']['native']['emit_x'], order_spread=main_row['observed_order']['native']['spread'],
                                     change_128_from_64_emit_x=main_row['native_successive']['128_vs_64']['emit_x'])
        for k in SEED_COPIES:
            case_file = HERE / 'cases' / f'sc_drift_n{n}s{k}.json'
            if not case_file.exists():
                continue
            tab = {}
            for m, cfg in NATIVE_CFG.items():
                t = runs(ROOT / f'sc_drift_n{n}s{k}/slipstream' / cfg, case_file)[0]
                if t:
                    tab[m] = t[0]
            if len(tab) == 3:
                s1, s2 = gap(tab[64], tab[32], cols), gap(tab[128], tab[64], cols)
                per_bunch[f's{k}'] = dict(order_emit_x=float(np.log2(s1['emit_x'] / s2['emit_x'])), order_spread=float(np.log2(s1['spread'] / s2['spread'])),
                                          change_128_from_64_emit_x=s2['emit_x'])
        if per_bunch:
            res[str(n)] = per_bunch
    return res


def main():
    cols = common.load(HERE / 'cases/sc_drift.json')['columns']
    out = dict(design=dict(date='2026-10-28', ns=[2000, 20000, *NS[1:]], note='Each N is a separate frozen bunch with the same distribution; gaps are between codes on the same particles. '
                                                                               '20 000 is the original case (stored runs).'),
               analysis_sha256=common.sha256(__file__), rows=[])
    for n in (NS[0], 20000, *NS[1:]):
        if n == 20000:
            case_file = HERE / 'cases/sc_drift.json'
            paths = {('slipstream', m): NATIVE20K[m] for m in (32, 64, 128)} | {('astra', m): ASTRA20K[m] for m in (32, 64, 128)}
        else:
            case_file = HERE / f'cases/sc_drift_n{n}.json'
            paths = {('slipstream', m): ROOT / f'sc_drift_n{n}/slipstream' / NATIVE_CFG[m] for m in NATIVE_CFG} | \
                    {('astra', m): ROOT / f'sc_drift_n{n}/astra' / ASTRA_CFG[m] for m in ASTRA_CFG}
        cpu = {2000: ROOT / 'sc_drift_n2000/slipstream/cpu-f64-3d32t1c1-mfix', 20000: common.RESULTS / '2026-10-27_mass_fix/sc_drift/slipstream/cpu-f64-3d32t1c1-mfix'}.get(n)
        if cpu:
            paths[('slipstream_cpu', 32)] = cpu
        data = {k: runs(p, case_file) for k, p in paths.items() if p.exists()}
        tab = {k: v[0][0] for k, v in data.items() if v[0]}
        z = np.load(HERE / 'cases' / common.load(case_file)['particles']['file'])
        s0 = common.lab_stats(*[z[k] for k in ('x', 'y', 'z', 'bgx', 'bgy', 'bgz')], mass_eV=common.load(case_file)['mass_eV'])
        row = dict(n=n, input_stats={k: float(s0[k]) for k in ('sigma_x_m', 'sigma_z_m', 'eps_nx_m', 'sigma_E_eV')},
                   warm_s={f'{c}{m}': float(np.median(v[1])) for (c, m), v in data.items() if v[1]},
                   cold_s={f'{c}{m}': float(np.median(v[2])) for (c, m), v in data.items() if v[2]},  # first call; the only timing of single-call (repeats 0) runs
                   n_ok={f'{c}{m}': len(v[0]) for (c, m), v in data.items()},
                   final={f'{c}{m}': {k: float(t[-1, cols.index(k)]) for k in ('sigma_x_m', 'sigma_z_m', 'eps_nx_m', 'eps_ny_m', 'sigma_E_eV')} for (c, m), t in tab.items()},
                   native_vs_astra_matched={str(m): gap(tab[('slipstream', m)], tab[('astra', m)], cols) for m in (32, 64, 128) if ('slipstream', m) in tab and ('astra', m) in tab},
                   native_successive={f'{b}_vs_{a}': gap(tab[('slipstream', b)], tab[('slipstream', a)], cols) for a, b in ((32, 64), (64, 128)) if ('slipstream', a) in tab and ('slipstream', b) in tab},
                   astra_successive={f'{b}_vs_{a}': gap(tab[('astra', b)], tab[('astra', a)], cols) for a, b in ((32, 64), (64, 128)) if ('astra', a) in tab and ('astra', b) in tab})
        if ('astra', 64) in tab:
            row['native_vs_astra64'] = {str(m): gap(tab[('slipstream', m)], tab[('astra', 64)], cols) for m in (32, 64, 128) if ('slipstream', m) in tab}
        succ = {'native': row['native_successive'], 'astra': row['astra_successive']}
        row['observed_order'] = {code: {k: float(np.log2(v['64_vs_32'][k] / v['128_vs_64'][k])) for k in ('size', 'emit_x', 'emit_y', 'spread')}
                                 for code, v in succ.items() if '64_vs_32' in v and '128_vs_64' in v}  # ratio of successive changes over one mesh doubling
        out['rows'].append(row)
    out['seed_control'] = seed_control(cols, out['rows'])
    (common.EVIDENCE / '2026-10-26_R1/injector_scaling_control.json').write_text(json.dumps(out, indent=2) + '\n')
    pc = lambda v: f'{100 * v:5.2f}%'  # noqa: E731
    for r in out['rows']:
        print(f"N = {r['n']:>8d}  warm [s]: " + '  '.join(f'{k} {v:.3g}' for k, v in r['warm_s'].items()) + '   | first call [s]: ' + '  '.join(f'{k} {v:.3g}' for k, v in r['cold_s'].items()))
        for name in ('native_vs_astra_matched', 'native_successive', 'astra_successive'):
            for k, v in r[name].items():
                print(f"   {name:25s} {k:>8s}  size {pc(v['size'])}  emit_x {pc(v['emit_x'])}  emit_y {pc(v['emit_y'])}  spread {pc(v['spread'])}")
        print('   observed order (emit_x, spread):', {c: (round(v['emit_x'], 2), round(v['spread'], 2)) for c, v in r['observed_order'].items()})
    print('seed control (native order of emit_x / spread, change 128 from 64):')
    for n, v in out['seed_control'].items():
        print(f'   N={n}:', {k: (round(x['order_emit_x'], 2), round(x['order_spread'], 2), f"{100 * x['change_128_from_64_emit_x']:.2f}%") for k, x in v.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--make-cases', action='store_true')
    a = ap.parse_args()
    make_cases() if a.make_cases else main()

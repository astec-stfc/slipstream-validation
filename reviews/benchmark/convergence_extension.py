"""Extend the 3-D mesh convergence evidence beyond 64^3 and refine the native 64^3 step and field refresh (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P1).

Adds ASTRA 3-D 96^3 and 128^3 (1.4 ps) and native Cartesian 96^3 and 128^3 (1 ps, refresh 10 ps, record 100 ps, 5% padding) to the existing
32^3 and 64^3 pairs (the native 32^3 and 64^3 members are re-run with the mass-fixed deck, which changes observables by <= 9e-6; the 16^3 run is the
earlier one), plus two 64^3 refinements of the fast preset (push step halved; physical field-refresh and record intervals halved).
ASTRA rounds Nxf to a power of two (see astra_mesh_cap_probe.py), so its effective meshes are 32^3, 64^3 and 128^3; the 96^3 run duplicates 128^3.
Reports, without selecting any run as truth:
  * how much each solver changes between successive meshes (self-convergence),
  * the native-versus-ASTRA gap at matched mesh and against the finest ASTRA run,
  * the change of the native 64^3 result under step and refresh refinement.

    python convergence_extension.py
"""
import json
from pathlib import Path

import numpy as np

import common
from analyse_sc import OBS, rel

HERE = Path(__file__).resolve().parent
CASE = HERE / 'cases/sc_drift.json'
R = common.RESULTS
METRICS = dict(OBS, emit_x=('eps_nx_m',), emit_y=('eps_ny_m',))
NEWC = R / '2026-10-27_convergence/sc_drift'

ASTRA = {32: R / '2026-10-26_R1/sc_drift/astra/cpu-f64-3d32h1.4', 64: R / '2026-10-27_astra_refinement/sc_drift/astra/cpu-f64-3d64h1.4',
         96: NEWC / 'astra/cpu-f64-3d96h1.4', 128: NEWC / 'astra/cpu-f64-3d128h1.4'}
NATIVE = {16: R / '2026-10-27_native_mesh/sc_drift/slipstream/cuda-f32-3d16t1c10-r100',
          32: R / '2026-10-27_mass_fix/sc_drift/slipstream/cuda-f32-3d32t1c10-r100-mfix',
          64: NEWC / 'slipstream/cuda-f32-3d64t1c10-r100-mfix',
          96: NEWC / 'slipstream/cuda-f32-3d96t1c10-r100-mfix', 128: NEWC / 'slipstream/cuda-f32-3d128t1c10-r100-mfix'}
NATIVE64 = {'step/2': NEWC / 'slipstream/cuda-f32-3d64t0.5c20-r200-mfix', 'refresh/2': NEWC / 'slipstream/cuda-f32-3d64t1c5-r50-mfix'}


def states(root):
    out, jobs = [], []
    for p in sorted(root.glob('p*/result.json')):
        r = common.load(p)
        job = dict(result_sha256=common.sha256(p), status=r['status'])
        if r['status'] == 'ok':
            assert r['case_sha256'] == common.sha256(CASE)
            st = np.load(p.parent / 'tbt.npz')['tbt']
            assert np.isfinite(st).all()
            out.append(st)
            job.update(artifact_sha256=common.sha256(p.parent / 'tbt.npz'), warm_s=(r['timing']['warm_s'] or {}).get('median'))
        jobs.append(job)
    return out, jobs


def gap(a, b, cols):
    return {k: max(rel(x, y, cols, v) for x in a for y in b) for k, v in METRICS.items()}


def main():
    cols = common.load(CASE)['columns']
    ast = {n: states(p) for n, p in ASTRA.items() if p.exists()}
    nat = {n: states(p) for n, p in NATIVE.items() if p.exists()}
    n64 = {k: states(p) for k, p in NATIVE64.items() if p.exists()}
    out = dict(design=dict(date='2026-10-27', note='All gaps are peer comparisons; no mesh is taken as converged truth. Maximum over stations and over process pairs.'),
               case_sha256=common.sha256(CASE), analysis_sha256=common.sha256(__file__))
    out['runs'] = {f'astra{n}': dict(n_ok=len(s), warm_s=float(np.median([j['warm_s'] for j in jobs if j.get('warm_s') is not None] or [np.nan])) if s else None, jobs=jobs)
                   for n, (s, jobs) in ast.items()}
    out['runs'].update({f'native{n}': dict(n_ok=len(s), warm_s=float(np.median([j['warm_s'] for j in jobs if j.get('warm_s') is not None] or [np.nan])) if s else None, jobs=jobs)
                        for n, (s, jobs) in nat.items()})
    out['runs'].update({f'native64_{k}': dict(n_ok=len(s), warm_s=float(np.median([j['warm_s'] for j in jobs if j.get('warm_s') is not None] or [np.nan])) if s else None, jobs=jobs)
                        for k, (s, jobs) in n64.items()})
    # ASTRA rounds Nxf to a power of two (astra_mesh_cap_probe.py): 48..88 -> 64, 96 and 128 -> 128.  The 96^3 run is checked to equal the 128^3 run
    # and is dropped from the effective ladder 32, 64, 128.
    out['astra96_equals_astra128'] = bool(ast.get(96) and ast.get(128) and ast[96][0] and ast[128][0] and gap(ast[96][0], ast[128][0], cols) == {k: 0.0 for k in METRICS})
    if out['astra96_equals_astra128']:
        del ast[96]
    effective = {16: 16, 32: 32, 64: 64, 96: 128, 128: 128}
    meshes = sorted(ast)
    out['astra_successive'] = {f'{b}_vs_{a}': gap(ast[b][0], ast[a][0], cols) for a, b in zip(meshes, meshes[1:]) if ast[a][0] and ast[b][0]}
    nm = sorted(nat)
    out['native_successive'] = {f'{b}_vs_{a}': gap(nat[b][0], nat[a][0], cols) for a, b in zip(nm, nm[1:]) if nat[a][0] and nat[b][0]}
    out['native_vs_astra_matched'] = {str(n): gap(nat[n][0], ast[effective[n]][0], cols) for n in nat if effective[n] in ast and nat[n][0] and ast[effective[n]][0]}
    out['astra_effective_mesh_of_native'] = {str(n): effective[n] for n in nat}
    finest = max((n for n in ast if ast[n][0]), default=None)
    if finest:
        out['astra_finest'] = finest
        out['native_vs_astra_finest'] = {str(n): gap(nat[n][0], ast[finest][0], cols) for n in nat if nat[n][0]}
    if 64 in nat and nat[64][0]:
        out['native64_refinement_vs_64'] = {k: gap(s, nat[64][0], cols) for k, (s, _) in n64.items() if s}
        if finest:
            out['native64_refinement_vs_astra_finest'] = {k: gap(s, ast[finest][0], cols) for k, (s, _) in n64.items() if s}
    ie = cols.index('ekin_eV')
    ae = common.load(common.EVIDENCE / '2026-10-26_R1/astra_particle_energy.json')['rows']
    out['mean_energy_gain_eV'] = dict(
        native={str(n): float(np.mean([a[-1, ie] - a[0, ie] for a in nat[n][0]])) for n in nat if n != 16 and nat[n][0]},
        astra_particles={**{str(v['n3d']): v['gain_eV'] for k, v in ae.items() if k != 'sc_drift_q0' and v['n3d'] != 96},
                         **{str(n): float(np.mean([common.load(p)['gain_eV'] for p in sorted(ASTRA[n].glob('p*/energy_gain.json'))]))
                            for n in ASTRA if n != 96 and sorted(ASTRA[n].glob('p*/energy_gain.json'))}},
        note='native: station table of mass-fixed runs; ASTRA: full-precision phase-space file at ZSTOP')
    (common.EVIDENCE / '2026-10-26_R1/convergence_extension.json').write_text(json.dumps(out, indent=2) + '\n')
    pc = lambda v: f'{100 * v:.3g}%'  # noqa: E731
    for name in ('astra_successive', 'native_successive', 'native_vs_astra_matched', 'native_vs_astra_finest', 'native64_refinement_vs_64', 'native64_refinement_vs_astra_finest'):
        if name in out:
            print(name)
            for k, v in out[name].items():
                print(f'  {k:>10s}  size {pc(v["size"]):>8s}  emit_x {pc(v["emit_x"]):>8s}  emit_y {pc(v["emit_y"]):>8s}  spread {pc(v["spread"]):>8s}')
    print({k: (v['n_ok'], v['warm_s']) for k, v in out['runs'].items()})
    print('mean energy gain [eV]', {k: {m: round(x, 2) for m, x in v.items()} for k, v in out['mean_energy_gain_eV'].items() if k != 'note'})


if __name__ == '__main__':
    main()

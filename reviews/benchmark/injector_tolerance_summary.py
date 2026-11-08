"""Summarise the injector tolerance budget (Jacobian against Monte Carlo) into data/2026-10-26_R1/injector_tolerance.json.

Inputs (written by slipstream_dev/benchmark/injector_tolerance_mc.py, CLARA 5 pC deck, CPU float64, 1024 particles, one fixed step):
  results/tolerance/2026-11-03_jacobian.json   autograd Jacobian, and each knob alone at +-10 sigma (one-at-a-time)
  results/tolerance/2026-11-03_mc_part{0,1,2}.json   Monte Carlo samples at 1, 5 and 20 times the tolerances
Per output and scale: the Jacobian prediction ``sigma = scale sqrt(sum (g_i sigma_i)^2)``, the one-at-a-time linear prediction, the second-order prediction (the
diagonal curvature from the +-10 sigma runs), the Monte Carlo mean shift and standard deviation with a bootstrap 95% interval, and each knob's share of the variance.
"""
import json
from pathlib import Path

import numpy as np

import common

HERE = Path(__file__).resolve().parent
RES = common.RESULTS / 'tolerance'
OUTPUTS = ('emit_x_mm_mrad', 'sigma_x_mm', 'sigma_z_mm', 'Ekin_MeV', 'dE_rms_keV')
KNOBS = ['cavities[0].phase_deg', 'cavities[0].e_max_V_per_m', 'cavities[1].phase_deg', 'cavities[1].e_max_V_per_m', 'solenoids[0].b_max_T', 'charge_scale']


def main():
    jac = json.loads((RES / '2026-11-03_jacobian.json').read_text())
    parts = [json.loads(p.read_text()) for p in sorted(RES.glob('2026-11-03_mc_part*.json'))]
    samples = [s for p in parts for s in p['samples']]
    sig = jac['sigma_abs']
    f0, grad, oat, T = jac['f0'], jac['grad'], jac['oat'], jac['oat_scale']
    rng = np.random.default_rng(0)
    rows = []
    for out in OUTPUTS:
        g = np.array([grad[out][k] * sig[k] for k in KNOBS])                      # response to a one-sigma error of each knob
        g_oat = np.array([(oat[k]['plus'][out] - oat[k]['minus'][out]) / (2 * T) for k in KNOBS])
        h = np.array([(oat[k]['plus'][out] + oat[k]['minus'][out] - 2 * f0[out]) / T**2 for k in KNOBS])   # curvature, per sigma squared
        for scale in sorted({s['scale'] for s in samples}):
            y = np.array([s['out'][out] for s in samples if s['scale'] == scale])
            boots = [np.std(rng.choice(y, len(y)), ddof=1) for _ in range(2000)]
            lin = scale * np.sqrt(np.sum(g**2))
            rows.append(dict(output=out, scale=scale, n=len(y), value=f0[out], mc_mean_shift=float(y.mean() - f0[out]), mc_std=float(np.std(y, ddof=1)),
                             mc_std_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], mc_mean_shift_se=float(np.std(y, ddof=1) / np.sqrt(len(y))),
                             jacobian_std=float(lin), oat_linear_std=float(scale * np.sqrt(np.sum(g_oat**2))),
                             second_order_std=float(np.sqrt(lin**2 + 0.5 * np.sum(h**2) * scale**4)), second_order_mean_shift=float(0.5 * np.sum(h) * scale**2),
                             variance_share=[float(v) for v in g**2 / np.sum(g**2)]))
    dt = [np.diff([0.0] + [s['t_s'] for s in p['samples']]) for p in parts if p['samples']]
    out = dict(design=dict(date='2026-11-03', deck='CLARA 5 pC injector400.in', particles=jac['particles'], n_steps=jac['n_steps'], dt_s=jac['dt_s'], knobs=KNOBS,
                           sigma_abs=sig, sigma_unit=jac['sigma_rel_or_deg'], scales=sorted({s['scale'] for s in samples}), oat_scale=T, outputs=list(OUTPUTS),
                           n_samples=len(samples), autograd_s=jac['autograd_s'], forward_s_median=float(np.median(np.concatenate(dt))),
                           note='Own-code check of the linearised budget: Monte Carlo of the same differentiable function, same particles; the jitter of the knobs is Gaussian and independent.'),
               source_sha256={p.name: common.sha256(p) for p in sorted(RES.glob('2026-11-03_*.json'))}, analysis_sha256=common.sha256(__file__), rows=rows)
    (common.EVIDENCE / '2026-10-26_R1/injector_tolerance.json').write_text(json.dumps(out, indent=2) + '\n')
    for r in rows:
        print(f"{r['output']:16s} x{r['scale']:<4g} n={r['n']:3d} MC {r['mc_std']:.3e} [{r['mc_std_ci'][0]:.2e},{r['mc_std_ci'][1]:.2e}]  Jac {r['jacobian_std']:.3e}  ratio {r['mc_std'] / r['jacobian_std']:.2f}  "
              f"2nd {r['second_order_std'] / r['jacobian_std']:.2f}  shift MC {r['mc_mean_shift']:+.2e} 2nd {r['second_order_mean_shift']:+.2e}")


if __name__ == '__main__':
    main()

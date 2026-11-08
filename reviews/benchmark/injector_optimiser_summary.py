"""Summarise the five-knob injector tuning comparison into data/2026-10-26_R1/injector_optimiser.json.

Inputs (written by slipstream_dev/benchmark/injector_optimiser_compare.py, CLARA 5 pC deck, CPU float64, 1024 particles, 1 ps step, three methods run at once on
one machine with equal wall-time budgets): results/optimiser/2026-11-03_start{1,2}_{lbfgs_ad,neldermead,bayes}.json. The Nelder-Mead and Bayesian traces of
start 1 were rebuilt from their stdout logs (their own JSON write failed; x is not recorded for them).
Per start and method: the trace of the best objective against wall time and evaluation number, the cost per evaluation, and the time and number of evaluations to reach
within 10 and 5 per cent of the best objective found by any method from that start.
"""
import json
from pathlib import Path

import numpy as np

import common

HERE = Path(__file__).resolve().parent
RES = common.RESULTS / 'optimiser'
METHODS = ('lbfgs_ad', 'neldermead', 'bayes')


def first_reaching(trace, level):
    for r in trace:
        if r['J_best'] <= level:
            return {'t_s': r['t_s'], 'n_eval': r['n_eval']}
    return None


def main():
    runs = {}
    for f in sorted(RES.glob('2026-11-03_start*_*.json')):
        d = json.loads(f.read_text())
        start = f.name.split('_')[1]
        runs.setdefault(start, {})[d['method']] = d
    out = dict(design=dict(date='2026-11-03', deck='CLARA 5 pC injector400.in', objective='emittance_n,x [mm mrad] + 0.05 (E_kin - 30 MeV)^2 at 3.37 m', budget_s=None,
                           note='Three processes at once on a 16-core CPU, 5 threads each, below-normal priority: the wall times include the contention of the other two and the backward pass '
                                'or surrogate fit of the method. Single starts, one run per method.'),
               source_sha256={}, analysis_sha256=common.sha256(__file__), starts={})
    for start, ms in runs.items():
        best = min(r['J_best'] for d in ms.values() for r in d['trace'])
        entry = dict(start_x=next(iter(ms.values()))['start'], best_J=best, methods={})
        for m in METHODS:
            if m not in ms:
                continue
            d = ms[m]
            tr = d['trace']
            out['design']['budget_s'] = d['budget_s']
            gaps = np.diff([0.0] + [r['t_s'] for r in tr])
            entry['methods'][m] = dict(stopped=d['stopped'], n_eval=len(tr), wall_s=d['wall_s'], final_best_J=tr[-1]['J_best'], median_s_per_eval=float(np.median(gaps)),
                                       to_10pct=first_reaching(tr, 1.10 * best), to_5pct=first_reaching(tr, 1.05 * best), source=d.get('source', 'own JSON'),
                                       trace=[{k: r[k] for k in ('t_s', 'n_eval', 'J', 'J_best', 'eps', 'Ekin')} for r in tr])
        out['starts'][start] = entry
        for f in sorted(RES.glob(f'2026-11-03_{start}_*.json')):
            out['source_sha256'][f.name] = common.sha256(f)
    (common.EVIDENCE / '2026-10-26_R1/injector_optimiser.json').write_text(json.dumps(out, indent=1) + '\n')
    for start, e in out['starts'].items():
        print(start, 'best J', round(e['best_J'], 4))
        for m, v in e['methods'].items():
            print(f"  {m:11s} {v['n_eval']:4d} evals  {v['median_s_per_eval']:6.1f} s/eval  final {v['final_best_J']:.4f}  10%: {v['to_10pct']}  5%: {v['to_5pct']}")


if __name__ == '__main__':
    main()

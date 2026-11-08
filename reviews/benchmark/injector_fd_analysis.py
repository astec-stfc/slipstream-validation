"""Summarise the injector derivative check into data/2026-10-26_R1/injector_derivative_control.json (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, item 31).

Input: results/2026-10-29_injector_fd.json, written by ``slipstream_dev/benchmark/injector_autograd_clara.py --modes fd`` (CLARA 5 pC deck, cathode to
1 m at one fixed step, 1024 particles, CPU float64, ``Injector.track_differentiable`` with bunch-fitted box, checkpointed autograd).  For every knob
(cavity phase, cavity field, gun solenoid) and objective (horizontal emittance, rms size, rms length) the autograd derivative is compared with central
differences at a relative step ladder; the best-agreeing step and the agreement at every step are kept.
"""
import json
from pathlib import Path

import numpy as np

import common

HERE = Path(__file__).resolve().parent
SRC = common.RESULTS / '2026-10-29_injector_fd.json'


def main():
    raw = json.loads(SRC.read_text())
    fd = raw['fd']
    rows = fd['rows']
    summary = []
    for knob in dict.fromkeys(r['knob'] for r in rows):
        for obj in dict.fromkeys(r['objective'] for r in rows):
            rs = [r for r in rows if r['knob'] == knob and r['objective'] == obj]
            best = min(rs, key=lambda r: r['rel'])
            ordered = sorted(rs, key=lambda r: -r['h'])
            rel_steps = sorted(raw['args']['fd_h'], reverse=True)   # the i-th largest absolute step is the i-th largest relative step (h = h_rel * scale)
            assert len(ordered) == len(rel_steps)
            summary.append(dict(knob=knob, objective=obj, autograd=rs[0]['autograd'], best_step=best['h'], best_rel_step=rel_steps[ordered.index(best)], best_rel=best['rel'],
                                by_step=[dict(h=r['h'], h_rel=hr, fd=r['fd'], rel=r['rel']) for r, hr in zip(ordered, rel_steps)]))
    out = dict(design=dict(date='2026-10-29', deck='CLARA 5 pC injector400.in (slipstream_dev/examples/clara_linac/ASTRA/5pC)', device=raw['args']['fd_device'], dtype=raw['args']['fd_dtype'],
                           particles=fd['particles'], n_steps=fd['n_steps'], z_stop_m=raw['args']['z_stop'], steps=raw['args']['fd_h'], env=raw['env'],
                           note='Own-code check: autograd (checkpointed track_differentiable) against central differences of the same function, CPU float64; cathode image off, bunch-fitted fixed-shape box.'),
               analysis_sha256=common.sha256(__file__), source_sha256=common.sha256(SRC), rows=summary)
    (common.EVIDENCE / '2026-10-26_R1/injector_derivative_control.json').write_text(json.dumps(out, indent=2) + '\n')
    for r in summary:
        print(f"{r['knob']:26s} {r['objective']:16s} AD {r['autograd']: .4e} best h {r['best_step']:.1e} rel {r['best_rel']:.2e}")


if __name__ == '__main__':
    main()

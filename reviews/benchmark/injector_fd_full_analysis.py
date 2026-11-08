"""Summarise the full-length, all-knob injector derivative check into data/2026-10-26_R1/injector_derivative_full.json.

Inputs: results/2026-11-03_injector_fd_all_knobs.json (relative steps 1e-4, 1e-6, 1e-7) and, when present, results/2026-11-03_injector_fd_all_knobs_b.json (1e-2, 1e-3,
1e-5, 1e-8, 1e-9), both written by
``slipstream_dev/benchmark/injector_autograd_clara.py --modes fd --knobs all --z-stop 3.37147 --dt-s 5e-13 --fd-particles 1024 --fd-h <steps> --no-compile``
(CLARA 5 pC deck, cathode to the end of the line at one fixed step, 1024 particles, CPU float64, ``Injector.track_differentiable`` with a bunch-fitted box and
checkpointed autograd). For every knob (phase and field of both cavities, the gun solenoid field, the bunch charge) and objective (horizontal emittance, rms size,
rms length, longitudinal emittance) the autograd derivative is compared with central differences over the whole step ladder; the best-agreeing step and the
agreement at every step are kept. This extends injector_fd_analysis.py, which covers three gun knobs over the first metre.
"""
import json
from pathlib import Path

import common

HERE = Path(__file__).resolve().parent
SRCS = [common.RESULTS / '2026-11-03_injector_fd_all_knobs.json', common.RESULTS / '2026-11-03_injector_fd_all_knobs_b.json']


def main():
    raws = [json.loads(s.read_text()) for s in SRCS if s.exists()]
    srcs = [s for s in SRCS if s.exists()]
    base = raws[0]
    pairs = {}
    for raw in raws:
        rel_steps = sorted(raw['args']['fd_h'], reverse=True)   # the i-th largest absolute step is the i-th largest relative step (h = h_rel * scale)
        rows = raw['fd']['rows']
        for knob in dict.fromkeys(r['knob'] for r in rows):
            for obj in dict.fromkeys(r['objective'] for r in rows):
                rs = sorted((r for r in rows if r['knob'] == knob and r['objective'] == obj), key=lambda r: -r['h'])
                assert len(rs) == len(rel_steps)
                p = pairs.setdefault((knob, obj), dict(knob=knob, objective=obj, autograd=rs[0]['autograd'], by_step=[]))
                assert abs(p['autograd'] - rs[0]['autograd']) <= 1e-9 * abs(p['autograd']), (knob, obj)   # one deterministic autograd value
                p['by_step'] += [dict(h=r['h'], h_rel=hr, fd=r['fd'], rel=r['rel']) for r, hr in zip(rs, rel_steps)]
    summary = []
    for p in pairs.values():
        p['by_step'].sort(key=lambda s: -s['h_rel'])
        best = min(p['by_step'], key=lambda s: s['rel'])
        summary.append(dict(knob=p['knob'], objective=p['objective'], autograd=p['autograd'], best_step=best['h'], best_rel_step=best['h_rel'], best_rel=best['rel'],
                            best_abs=abs(p['autograd'] - best['fd']), by_step=p['by_step']))
    steps = sorted({s['h_rel'] for p in summary for s in p['by_step']}, reverse=True)
    out = dict(design=dict(date='2026-11-03', deck='CLARA 5 pC injector400.in (slipstream_dev/examples/clara_linac/ASTRA/5pC)', device=base['args']['fd_device'],
                           dtype=base['args']['fd_dtype'], particles=base['fd']['particles'], n_steps=base['fd']['n_steps'], z_stop_m=base['args']['z_stop'],
                           dt_s=base['args']['dt_s'], steps=steps, knobs=base['args']['knobs'], env=base['env'],
                           note='Own-code check: autograd (checkpointed track_differentiable) against central differences of the same function, CPU float64; cathode image off, '
                                'bunch-fitted field box. The two linac solenoids of the 5 pC deck are off, so the gun solenoid is the only solenoid knob.'),
               analysis_sha256=common.sha256(__file__), source_sha256={s.name: common.sha256(s) for s in srcs}, rows=summary)
    (common.EVIDENCE / '2026-10-26_R1/injector_derivative_full.json').write_text(json.dumps(out, indent=2) + '\n')
    rel = sorted(r['best_rel'] for r in summary)
    print(f"{len(summary)} pairs, {len(steps)} steps; best-step agreement min {rel[0]:.1e}, median {rel[len(rel) // 2]:.1e}, max {rel[-1]:.1e}; "
          f"{sum(v < 1e-5 for v in rel)} below 1e-5")


if __name__ == '__main__':
    main()

# benchmark review

The four frozen cases behind the paper's cross-code benchmark, run in the validation image. The harness was moved here from
`slipstream_dev/reports/Slipstream/benchmarks/`; the workers, `common.py` and the analysis scripts are unchanged except that they
read `common.RESULTS` instead of a folder next to them, and carry no machine path.

| Case | Question | Image codes (CPU) | Archived codes | Gates |
| --- | --- | --- | --- | --- |
| `ring_tunes` | amplitude-dependent tunes of a 48 m ring over 1024 turns | Slipstream, elegant, Xsuite, Cheetah | JuTrack, Bmad-X | G1 conventions, G2 nonlinear ring, G3 precision |
| `ring_grad` | five-knob Jacobian by automatic differentiation against finite differences | Slipstream, elegant, Xsuite, Cheetah | JuTrack, Bmad-X | G11 |
| `sc_drift` | free-drift space charge on one frozen bunch | Slipstream, ASTRA, Cheetah, Ocelot | GPT, ImpactX | G4 |
| `linear_fodo` | proton FODO, analytic control | Slipstream | HELIX | G1 |

`cases.toml` lists the configurations a campaign runs, `workers.toml` the worker of each code, `bench.toml` (repository root) how a
code is reached. With `gpus = true` in `bench.local.toml` the `gpu` configurations of `cases.toml` are added.

## Run

```bash
python -m slipstream_validation.runner checkout ../slipstream HEAD --name head       # git archive of the commit under test
python -m slipstream_validation.runner plan benchmark --case ring_tunes                # prints the jobs, starts nothing
python -m slipstream_validation.runner run benchmark --execute --tag pc2 --slipstream checkouts/slipstream-head --case ring_tunes
python -m slipstream_validation.check compare pc2 --case ring_tunes
```

`--procs 3` repeats each configuration in three independent processes, as the paper's timings did; accuracy needs one.

## Gates

`check compare` is the gate of a rerun: the tune error against the case's reference must not move by more than
max(25 % of the archived value, 1e-9) from `reviews/claims/evidence/2026-10-26_R1/corrected_reference.json`. This is a ratchet from one
host pair, to be re-measured on a second host. The physics gate of the case itself (tune error up to 1e-5, first-turn residual up to
1e-8) is in `cases/ring_tunes.json` and in `PLAN.md` of the report folder (G2).

Only `ring_tunes` has archived evidence wired into `check.py`. The other three cases were run in the image on 6 November 2026 (16 jobs, all ok) and their analysis scripts work; they need their archive loaders written. Configuration names in `cases.toml` must be real: list a worker's with `python workers/w_NAME.py --list` in the image.

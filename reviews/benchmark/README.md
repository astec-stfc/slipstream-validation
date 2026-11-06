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

All four cases are wired. `python -m slipstream_validation.check compare TAG` analyses every case in the campaign and compares it with the archive; `CASES` in `check.py` says which columns are gated, the floor of each case and its absolute limits:

| Case | Gated columns (ratchet) | Floor | Also |
| --- | --- | --- | --- |
| `ring_tunes` | `dq_ref` | 1e-9 | |
| `ring_grad` | `ad_vs_fd_fro`, `ad_vs_fd_col`, `rel_fro`, `rel_col`, `dy0` | 1e-9 | `ad_vs_fd_fro` must stay below `jac_rel` = 1e-5 (G11), whatever the archive says |
| `sc_drift` | `size`, `length`, `emit`, `spread` against the reference, `dEkin_change_eV` | 1e-9 | `pass` flag equal |
| `linear_fodo` | `position_abs_m`, `momentum_abs` | 1e-15 | `pass` flag equal; `cpu-f64-o4k4b8` fails the 1e-11 gate by design, in the archive too |

Rows: `ok`, `new` (a rung with no archived row on purpose, `NOREF_OK`), `NOREF` (no archived row and not allowed: a hole), `MOVED`, `LIMIT`, `FAILED`, and `XFAIL` (moved, with a reason in `KNOWN`). Exit status is 1 for `MOVED`, `LIMIT` and `FAILED`. `tests/test_framework.py` checks that every planned job has an archived row and that the gate can fail. A Slipstream CPU float64 row without a CPU archive row is compared with the archived GPU row of the same model (`ALIASES`; printed as `[vs ...]`). Rung-to-rung columns (`*_vs_prev`) are not gated because they depend on which rungs a campaign ran.

**First campaign, 6 November 2026 (`pc1`, 25 jobs):** every row agrees with the archive except four `ring_grad` metrics, all XFAIL for one reason. The elegant Jacobian reference of the archive is not reproduced: a rerun of the same case gives an endpoint state 5.2e-7 and a Jacobian 2.3e-4 away, in the image (elegant 2026.1.0) and in native Windows (2026.3.0, which agree to 6e-14), while the Slipstream and Xsuite arrays reproduce the archive (1e-15 and exactly). So Slipstream's distance to the reference fell from 2.2e-4 to 4e-7 only because the reference moved. Cause not isolated; the hash of the archived `w_elegant.py` differs from the file in the old report folder. The gated self-consistency (AD against the code's own central difference) did not move: Slipstream 3.88e-9 in both. The ASTRA of the image (Linux serial, Version 4.0, DESY 2022) gives turn-by-turn moments identical to the archived Windows `Astra.exe` (same banner): maximum difference 0.0 on the two rungs checked (`3d32h1.4`, `n64l128h0.7`).

Configuration names in `cases.toml` must be real: list a worker's with `python workers/w_NAME.py --list` in the image.

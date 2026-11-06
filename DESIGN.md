# slipstream-validation: design

Status: 6 November 2026. Scaffold. Tier 0 (the `claims` review) runs; the image, the benchmark port and the injector review are planned in section 9.

## 1. What it is for

Slipstream's own tests check the package without any other code. The evidence behind the paper needs more: ASTRA, elegant, Xsuite,
Ocelot, Cheetah and GPT to compare with, frozen benchmark cases, a CLARA injector study, and saved results that every quoted number is
recomputed from. All of that lives today in `slipstream_dev/reports/Slipstream/` next to the manuscript, with machine paths in
`benchmarks/registry.toml` and 1.7 GB of untracked raw results. It runs on one workstation.

This repository is for the checks, not the paper. A second person on a second PC must be able to **pull, run and report** the same
data. Every check is a **gate**: a quantity, a reference and a tolerance. A gate that fails means a bug, or a documented exception
written next to the gate. It is not a ranking of codes and not a speed contest; Slipstream is compared with ASTRA, GPT, elegant and
the others as peers, and none of them is treated as truth.

## 2. Principles

These are those of [simba-integration-tests](https://github.com/astec-stfc/simba-integration-tests), with three additions that
come from what Slipstream does.

1. **Gates, not tables.** A tolerance comes from a measured spread and says where from. A known difference is `xfail` with the reason, so a fix is noticed.
2. **Frozen inputs, recorded provenance.** Cases, bunches and decks are hashed. Every result records the Slipstream commit (and dirty flag), the version of each code, the image id, the host (CPU, GPU, threads, OS) and the exact command.
3. **A review is a folder** with the same skeleton, so the next one does not touch shared code.
4. **Tiers.** What needs no simulation code runs in CI on every push; what needs codes runs in the image.
5. **Accuracy reproduces, timing is reported.** *(new)* Accuracy observables (tunes, emittances, beam sizes, derivatives) must reproduce on another PC within a tolerance taken from the spread between hosts. Wall time and memory do not reproduce across hosts, so a second host adds a row beside the archived one and the gates on timing are ratios between codes on the same host.
6. **Archived evidence for codes that cannot run.** *(new)* GPT (licence), ImpactX (conda build), JuTrack (Julia), HELIX and Bmad-X are not in the image. Their results are saved once, hashed, with the code version, and used as the reference. A machine that has the code sets `platform` in `bench.local.toml` and reruns it.
7. **No machine path in a committed file.** *(new)* `benchmarks/registry.toml` held `C:/Users/...` paths; they move to an uncommitted `bench.local.toml`.

## 3. Can the simcodes image give the same data on another PC?

Yes for accuracy, and for most of the codes. Not for everything, and not for timing. Measured on this PC on 6 November 2026
(`ghcr.io/astec-stfc/simcodes-docker:master`, 16.7 GB, `docker run`, nothing built):

| Piece | In the image | Same as the paper's run? | Consequence |
| --- | --- | --- | --- |
| Slipstream | mounted from a `git archive` export (the repository is private, so the image cannot clone it) | torch 2.13.0+cu130, Python 3.14.6, CPU or `--gpus all` | Accuracy gates run. GPU needs the NVIDIA container toolkit (Docker Desktop on WSL 2 passes the card through; `nvidia-smi` sees the RTX 3050 here) |
| Xsuite | xtrack 0.111.0 | Same version as the benchmark's WSL environment | CPU rows run. The GPU rows need cupy, which the image lacks (tier 2) |
| Ocelot | ocelot-desy 25.6.0 | Not checked against the paper's build | The result records the version; a mismatch is a finding, not a failure |
| Cheetah | yes | Version not yet compared | Same |
| elegant | **2026.1.0** | **The paper cites 2026.3.0** | Rebuild the image with the newer tag, or narrow the claim to the version run. Decide before the first campaign |
| ASTRA | Linux serial and parallel from DESY | The paper's runs used the Windows 4.0 (2022) build | Compare the two builds on `sc_drift` first; the parallel build also gives the 16-rank timings the CLARA study used |
| GPT | no (licence) | n/a | Archived reference, or native Windows with a licence |
| ImpactX, JuTrack, HELIX, Bmad-X | no | n/a | Archived reference; Bmad-X could join the image later (it installs with pip) |

What this means for the paper's reproducibility item (R5 in `PLAN.md`): a second PC with Docker reproduces the accuracy gates and
the figures that depend on them. Performance claims stay tied to the measured workstation unless a second GPU host reruns the
timing cases; the report puts the hosts side by side and gives ratios, and the paper's wording can be narrowed to what was
measured. Floating-point results differ between hosts at about 1e-12 in float64 on a CPU (BLAS threading) and more in float32 on a
GPU, so tolerances for reruns are set from the spread between two hosts, not from one run.

## 4. Layout

```text
slipstream-validation/
├── README.md  DESIGN.md  AGENTS.md           what it is, why, rules for coding agents (AGENTS.md to be written)
├── pyproject.toml                            package slipstream_validation; pytest markers
├── bench.toml                                how each code is reached; copy to bench.local.toml (not committed)
├── docker/Dockerfile                         simcodes image + Slipstream
├── src/slipstream_validation/                (planned) runner.py, check.py, results.py, report.py
├── reviews/
│   ├── claims/                               runs now: verify_claims.py, test_claims.py, evidence/
│   ├── benchmark/                            (planned) cases/, workers/, cases.toml, gates/, evidence/
│   └── injector/                             (planned) CLARA 20/100/250 pC, ASTRA and GPT archived
├── tests/                                    (planned) tier-0 tests of the framework itself
└── .github/workflows/tier0.yml               (planned) tier 0 on push and pull request
```

Not committed: `results/`, `runs/`, `work/`, `checkouts/`, `bench.local.toml`. Committed evidence is compact (JSON, CSV, a few text
files; 4 MB today); raw arrays and run folders stay local, and the data-availability statement of the paper (R4) points to an
archive deposit of them, not to this repository.

## 5. Tiers

| Tier | Needs | Runs | What it checks |
| --- | --- | --- | --- |
| 0 | Python, numpy, pytest | every push, in CI, 0.4 s | the 401 quoted numbers against archived evidence; hashes of cases and evidence; that cases, workers and the code table agree |
| 1 | the image, Slipstream on CPU | by hand or scheduled, minutes to an hour | ring tunes, five-knob Jacobian, free-drift space charge and linear FODO in Slipstream, elegant, ASTRA, Ocelot, Xsuite (CPU) and Cheetah; reruns agree with the archived campaign within the cross-host tolerance |
| 2 | a GPU, a licence or an external build | by hand on the host that has it | GPU rows and Xsuite on GPU, GPT, ImpactX, JuTrack, HELIX, Bmad-X, the long CLARA runs |

## 6. Reviews and gates

The acceptance gates G0 to G13 of `PLAN.md` already are gates in the sense above (test, pass condition, state). They become the
review READMEs' gate tables, one row each, with the tolerance and the tier.

| Review | Gates it carries | Reference | Source in the report folder |
| --- | --- | --- | --- |
| claims | G0 (every quoted number follows from archived data) | the archived evidence | `scripts/verify_v8_numbers.py`, `data/2026-10-26_R1/` |
| benchmark | G1 maps, G2 nonlinear ring, G3 precision, G4 field solver, G11 derivatives, G12 performance | corrected elegant reference; Richardson reference for space charge; analytic FODO | `benchmarks/` (`cases/`, `workers/`, `study.py`, the control scripts) |
| injector | G5 injector transport, G6 space-charge scheduling, G9 wake kicks | refined ASTRA (three runs), GPT archived | `Stories/CLARA_InjectorConvergence/`, `benchmarks/clara_*.py` |
| csr, rings (later) | G7, G8, G10 | elegant, Pelegant, IW2D archived | `data/2026-10-26_R1/csr_validation.json`, booster stories |

## 7. How a second PC pulls, runs and reports

1. `git clone` this repository and `docker pull ghcr.io/astec-stfc/simcodes-docker:master` (several GB, once).
2. `docker build -t slipstream-validation:local -f docker/Dockerfile docker/` (seconds; adds pytest and soliday.sdds).
3. `python -m slipstream_validation.runner run benchmark --execute --tag pc2`, with `--slipstream checkouts/slipstream-NAME` (a `git archive` export of the commit or branch to test; it needs read access to the Slipstream repository) and `gpus = true` in `bench.local.toml` on a GPU host.
4. `python -m slipstream_validation.check compare archived pc2` lists what moved beyond the cross-host tolerance; `check report pc2` writes the markdown tables with the host, versions and image id on top.
5. Send back `results/pc2/` (compact JSON) or commit it under `reviews/benchmark/evidence/`.

Steps 3 and 4 exist for all four cases (section 10, phase 3): `python -m slipstream_validation.check compare TAG`.

## 8. What moves from the report folder

Copied, not moved: the report folder is untouched until the new repository has reproduced its numbers. Then the report keeps the
manuscript, drafts, figures and builders, and points here for the evidence.

| Goes to this repository | Stays in the report folder |
| --- | --- |
| `data/2026-10-26_R1/` and the four CLARA files the claims need (done, `reviews/claims/evidence/`) | `overleaf_git/`, `drafts/`, `final/`, `overleaf/`, `figures/`, `references/` |
| `benchmarks/` without `results/` (1.7 GB stays local) and with `registry.toml` rewritten as `bench.toml` | `build/`, `reference_reports/`, `sources/`, `archive/` |
| `scripts/verify_*`, `validate_*`, `sc_validation.py` | `scripts/make_*` (figure makers), bibliography builders |
| `review/vN/claims.csv` as the archived reference of the claims | `review/` evidence of the draft reviews |
| the CLARA convergence runners from `Stories/CLARA_InjectorConvergence/` (outputs, 12 GB, stay local) | the story write-ups |

## 9. Changes to the Slipstream package tests

The package keeps portable tests only: no other code, no workbench, no data it does not carry. Anything that needs ASTRA, elegant,
GPT, LAURA lattices or the 12 GB of story output belongs here.

PLAN.md (R5) records 375 passed, 9 skipped and 8 errors at `f18488d`. On 6 November 2026 the suite at `50a12dc` (seven commits later) on
the Windows workstation gave **473 passed, 7 skipped, 0 errors** in 11 minutes, so those errors are not present at that commit
there. The seven skips are the ones that should skip: no ELENA table, no lattice file, no elegant development helpers.

The same suite in the clean Linux image (CPU only, 3 minutes, working tree mounted) gave 385 passed, 71 skipped, **2 failed and 19 errors**.
All 21 are in five files that read `examples/` (`test_fit_structure`, `test_laser_image_fit`, `test_photocathode_example`,
`test_mls_example`, `test_diamond_example`), with `FileNotFoundError` on `examples/photocathode/fit_structure.py` and similar. The cause is
this workstation, not the package: `slipstream/examples` is a Windows junction to the sibling `slipstream_examples` folder, and the
junction target is not inside the Docker mount, so every example file looks missing. Run from a clean `git archive HEAD` export instead,
the same five files give **18 passed, 7 skipped, 0 failed, 0 errors**; the skips are CUDA, `laura` not installed and the laser camera
frame not in the checkout, each with a reason.

Consequences:

1. The runner mounts a `git archive` export under `checkouts/` (as simba-integration-tests' `pr.py` does), never a working tree. That is also what a second PC has, and it ignores junctions, untracked files and local edits. A dirty working tree is refused or recorded.
2. The eight errors of R5 cannot be tied to a defect in the package at `50a12dc`. Whether the old `f18488d` errors were this same junction is not checked; a `git archive f18488d` run in the image would settle it (about 4 minutes).
3. The package needs no change for this. What it could add is a marker for tests that need `examples/`, `laura`, CUDA or Warp, so a reader sees the 71 skips by kind (`pytest -m "not cuda"`); that is optional and is the Slipstream maintainers' decision.

## 10. Order of work

| Phase | Work | Effort |
| --- | --- | --- |
| 1 | Scaffold, `claims` review, tier-0 CI, this document | done |
| 2 | Port `benchmarks/run.py` to the `docker` platform: `runner.py` (checkout, plan, run), `check.py` (compare), `bench.toml`, `reviews/benchmark/` with cases, workers and analysis. Workers are unchanged (they exchange the benchmark standard frame). `check report` (markdown tables) still to do | done, 6 Nov |
| 3 | First image campaign on this PC. `ring_tunes` done (below). Still to run and wire: `sc_drift` (Slipstream, ASTRA, Cheetah, Ocelot), `ring_grad`, `linear_fodo`, and the ASTRA Linux against Windows build question | `sc_drift` about half a day |
| 4 | Second PC: the same campaign; cross-host tolerances; replace the unreproducible performance wording | 0.5 day plus a machine |
| 5 | `injector` review with ASTRA in the image and GPT archived | 1 to 2 days |

### First image campaign: `ring_tunes`, 6 November 2026

Tag `pc1`: this PC, Docker in WSL, the validation image, Slipstream `50a12dc` from a `git archive` export, CPU only, one process per
configuration, nine jobs. All nine agree with the archived `2026-10-26_R1` tune errors (`check compare pc1`):

| Code | Configuration | Archived dq | Image dq |
| --- | --- | --- | --- |
| elegant | o6s8b16 | 3.817e-9 | 3.817e-9 |
| Slipstream | o6k8b16 (CPU f64; archived row is GPU f64) | 1.630e-9 | 1.625e-9 |
| Slipstream | o6k16b32 (CPU f64; archived row is GPU f64) | 3.59e-11 | 4.2e-11 |
| Xsuite | yosk8b16 | 2.899e-4 | 2.899e-4 |
| Xsuite | yosk8b16-exact | 6.863e-6 | 6.863e-6 |
| Xsuite | yosk16b32-exact | 6.025e-7 | 6.025e-7 |
| Cheetah | dri16 | 1.040e-4 | 1.040e-4 |
| Cheetah | lin1 | 3.359e-3 | 3.359e-3 |

What it shows:

1. **The elegant version does not matter for this case.** The image's 2026.1.0 gives the same tune errors as the paper's 2026.3.0 to four figures. The version difference stays worth one sentence in the paper, but needs no rebuild for `ring_tunes`.
2. **Different library versions, same numbers.** Xsuite is 0.111.0 here and 0.111.6 in the archive, torch 2.13 against 2.14, Cheetah 0.8.4 in both. The runs agree to the digits shown, except Slipstream's 1e-11 level, which is float64 noise on a different device.
3. **The first campaign was not a timing result.** Wall times differ greatly and mean nothing across devices (Slipstream CPU eager, one thread, 124 s against 2.9 s on the GPU). They are printed side by side and never gated.
4. **The other three cases also run in the image.** `sc_drift` (Slipstream 3 configurations, ASTRA 3, Cheetah 2, Ocelot 2), `ring_grad` (Slipstream, elegant, Xsuite, Cheetah) and `linear_fodo` (Slipstream, 2 configurations): 16 more jobs, 25 in total, all ok. All four cases now have an archived loader in `check.py` (columns, floors and limits in `CASES`; table in `reviews/benchmark/README.md`), and `check compare pc1` runs them all. `sc_drift` and `linear_fodo` agree with the archive on every row, with Slipstream's float64 CPU rows compared with the archived float32 GPU rows (the size, length, emittance and spread differences from the reference move by at most 2.3e-6 absolute; the energy change moves 0.7 % and 1.2 %). The ASTRA of the image is the Linux serial build, Version 4.0 DESY 2022, like the paper's Windows `Astra.exe`; its turn-by-turn moments are identical to the archive (maximum difference 0.0, two rungs checked), which closes the cross-build ASTRA question for this case. An extra rung, ASTRA `3d64h1.4`, has no archived row (status `new`) and differs from the `3d32h1.4` reference by 0.94 % in size.
5. **A wrong guess caught by a check:** `cases.toml` first carried configuration names taken from the study's globs, and five were not real. Listing each worker's configurations in the image (`--list`) found them before a run did. A tier-1 test that does this belongs in the next step.
6. **Three traps found and fixed:** Xsuite compiles C kernels into its working directory, so the container's working directory is `/tmp` and the repository is mounted read-only; a worker keeps its traceback in `result.json`, not on stderr, so the runner prints that; the Ocelot worker asked for the distribution `ocelot-collab` while the image has the same code as `ocelot-desy`.

7. **One archived number does not reproduce: the elegant `ring_grad` reference.** The Jacobian and endpoint state that elegant gives for the five-knob case are 2.3e-4 and 5.2e-7 (absolute) away from the archived ones, with the same case file, the same lattice hash and the same executable path as the archived run. The image (elegant 2026.1.0) and native Windows (2026.3.0, run again from this repository's worker) agree with each other to 6e-14, and the Slipstream and Xsuite arrays reproduce the archive (1e-15, exactly), so only the elegant row moved. Three archived `vs reference` columns of Slipstream and one of Cheetah follow it; Slipstream's distance to the reference fell from 2.2e-4 to 4e-7, i.e. it agrees with the rerun of elegant better than the paper's table says. The four metrics are XFAIL in `KNOWN` with this reason. The one lead: the archived `w_elegant.py` hash differs from the file in the old report folder, so the worker was edited after the 26 October campaign. Before the paper quotes the `ring_grad` distance to elegant, rerun the old and new workers side by side or regenerate the archived row; the self-consistency gate (AD against the code's own central difference, 3.88e-9) did not move.

## 11. Open decisions

1. **Public or private?** The remote exists with an Apache 2.0 licence. The archived evidence includes GPT output and CLARA-derived numbers; confirm that is releasable before the repository is public.
2. **elegant version.** `ring_tunes` is unaffected (identical to four figures) and so is `ring_grad` (image 2026.1.0 and native 2026.3.0 agree to 6e-14). Rebuild at 2026.3.0 only if a later case, such as CSR or the Pelegant baseline, shows a difference.
3. **Package tests and CI.** Whether the Slipstream repository adds the markers of section 9 (a change to the package, not done here).
4. **Data deposit.** The raw results (1.8 GB benchmark, 12 GB story output) need a persistent identifier for R4; this repository holds the compact part only.

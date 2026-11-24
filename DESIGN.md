# slipstream-validation: design

Status: 8 November 2026. The `claims` review (tier 0) and the `benchmark` review run; the benchmark and validation work of the report folder has moved here (section 12); the injector review is planned.

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
6. **Archived evidence for codes that cannot run.** *(new)* GPT (licence), ImpactX (needs a source build) and JuTrack (Enzyme on Linux) are not in the image. Their results are saved once, hashed, with the code version, and used as the reference. A machine that has the code sets `platform` in `bench.local.toml` and reruns it.
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
| Bmad-X, SciBmad, HELIX | conda environments in the extended image, built from `envs/*.yml` (8 Nov) | Bmad-X `563c825`, SciBmad 0.6.0+26.10.14 (Julia 1.12 inside), HELIX v1.20.0: the paper's versions | Run in the image and reproduce the archive |
| JuTrack | no | Julia 1.10.4 on Windows | Enzyme, a dependency, does not precompile on Linux with Julia 1.10.4 or 1.10.10 (WSL and Docker, `Missing Julia object for global`). Archived on Linux; runs on the workstation's Windows platform |
| OPAL-X | no (10 Nov) | 1.0.0 (tag `OPALX-1.0.0`): the official CPU image `ghcr.io/opalx-project/opalx:cpu`, and a CUDA source build for the RTX 3050 | Not in the image: the official image is Ubuntu 24.04 (glibc 2.39) against AlmaLinux 9 (2.34), so its binary cannot be copied in, and it has no GPU build. `setup/install_opalx.sh` builds the CUDA and OpenMP versions and pulls the image; the worker reaches them through `BENCH_OPALX_*` (platform `wsl` or `linux`, default `skip`). See `reviews/benchmark/codes/CODES.md` |
| ImpactX | no | a source build at `e4279a4b` (26.10 plus 20 commits) | `beam.at_fixed_t()`, which the worker needs, is not in conda-forge's 26.10. Archived on Linux until a release has it, or build the source |

What this means for the paper's reproducibility item (R5 in `PLAN.md`): a second PC with Docker reproduces the accuracy gates and
the figures that depend on them. Performance claims stay tied to the measured workstation unless a second GPU host reruns the
timing cases; the report puts the hosts side by side and gives ratios, and the paper's wording can be narrowed to what was
measured. Floating-point results differ between hosts at about 1e-12 in float64 on a CPU (BLAS threading) and more in float32 on a
GPU, so tolerances for reruns are set from the spread between two hosts, not from one run.

## 4. Layout

```text
slipstream-validation/
├── README.md  DESIGN.md  AGENTS.md           what it is, why, rules for coding agents
├── pyproject.toml  .gitattributes            package slipstream_validation; pytest markers; cases and evidence byte for byte
├── bench.toml                                how each code is reached (platform per code); bench.local.toml (not committed) overrides it
├── bench.local.example.toml                  the settings of a Windows workstation with WSL
├── docker/Dockerfile  .dockerignore          simcodes image + conda environments of the extra codes (context: the repository root)
├── envs/*.yml  setup/*.sh                    one conda environment file per code family; installers (also for a Linux host without Docker)
├── src/slipstream_validation/                runner.py (plan, run, checkout; platforms), check.py (compare with the archive), export.py (to the report)
├── reviews/
│   ├── claims/                               verify_claims.py (453 quoted numbers), test_claims.py, evidence/ (compact, committed)
│   ├── benchmark/                            cases/, workers/, cases.toml, workers.toml, analysis, 40-odd control scripts, run.py, study.py, codes/CODES.md
│   └── injector/                             (planned) CLARA 20/100/250 pC, ASTRA and GPT archived
├── tests/                                    tier-0 tests of the framework itself
└── .github/workflows/tier0.yml               tier 0 on push and pull request
```

Not committed: `results/`, `runs/`, `work/`, `checkouts/`, `bench*.local.toml`. Committed evidence is compact (JSON, CSV, a few text
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
2. `docker build -t slipstream-validation:local -f docker/Dockerfile .` (about 15 minutes and 15 GB on top of the simcodes image: pytest, soliday.sdds, and the conda environments of Bmad-X, SciBmad and HELIX). On a host without Docker: `bash setup/install_envs.sh` and `platform = "conda"` in `bench.local.toml`.
3. `python -m slipstream_validation.runner run benchmark --execute --tag pc2`, with `--slipstream checkouts/slipstream-NAME` (a `git archive` export of the commit or branch to test; it needs read access to the Slipstream repository) and `gpus = true` in `bench.local.toml` on a GPU host.
4. `python -m slipstream_validation.check compare archived pc2` lists what moved beyond the cross-host tolerance; `check report pc2` writes the markdown tables with the host, versions and image id on top.
5. Send back `results/pc2/` (compact JSON) or commit it under `reviews/benchmark/evidence/`.

Steps 3 and 4 exist for all four cases (section 10, phase 3): `python -m slipstream_validation.check compare TAG`.

## 8. What moves from the report folder

Done on 8 November 2026 (section 12). The report keeps the manuscript, drafts, figures and builders, and reads the evidence from here.

| Moved to this repository | Stays in the report folder |
| --- | --- |
| `data/2026-10-26_R1/` and the four CLARA files the claims need (`reviews/claims/evidence/`; the report's `data/` is now a copy made by `python -m slipstream_validation.export`) | `overleaf_git/`, `drafts/`, `final/`, `overleaf/`, `figures/`, `references/` |
| `benchmarks/` without `results/`: harness, cases, workers, analysis, all control scripts, `BENCHMARK_STUDY.md`, `codes/CODES.md`, with `registry.toml` rewritten as `bench.toml` and the uncommitted `bench.local.toml`; 2.2 GB of `results/` copied (gitignored) | `build/`, `reference_reports/`, `sources/`, `archive/` |
| the claims verifier (`scripts/verify_v8_numbers.py` is now `reviews/claims/verify_claims.py`) | `scripts/make_*` (figure makers; two of them read this repository through `scripts/validation_repo.py`), bibliography builders, `validate_vN.py` (draft checks), and the two audits of the 26 October study (`verify_study.py`, `verify_source_export.py`, kept as a record) |
| `review/vN/claims.csv` as the archived reference of the claims | `review/` evidence of the draft reviews |
| not yet: the CLARA convergence runners and bridges of `Stories/CLARA_InjectorConvergence/` and `slipstream_dev/benchmark/` (outputs, 12 GB, stay local); six control scripts that read them are here but need `SSV_WORKBENCH` | the story write-ups |

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
| 3b | 8 Nov: the move from the report folder (section 12): harness, controls, results, platforms, environment files, extended image, claims against the current evidence | done |
| 4 | Second PC: the same campaign on a Linux box with Docker (`docker build`, `runner run`, `check compare`); cross-host tolerances; replace the unreproducible performance wording | 0.5 day plus a machine |
| 5 | `injector` review with ASTRA in the image and GPT archived; the CLARA and ELENA bridges that still need the workbench (`SSV_WORKBENCH`) | 1 to 2 days |

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

7. **The first `ring_grad` comparison pointed at a superseded archive row, and found the elegant build.** `check.py` first read `ring_grad.csv` of 26 October, where the elegant reference is 5.2e-7 (endpoint state) and 2.3e-4 (Jacobian) away from a rerun. That row is the one before the 27 October `p_central` fix of the elegant worker (the archived `w_elegant.py` hash differs for that reason); `corrected_reference.json` holds the corrected table, and `check.py` now compares with it. Against the corrected row, native Windows elegant 2026.3.0 with this worker reproduces the archived arrays exactly (difference 0.0), and the image's 2026.1.0 differs by 6e-14 in the endpoint state and 7.6e-9 in the Jacobian. Every `ring_grad` metric agrees within the ratchet except Slipstream's worst-column error against elegant (4.7e-5 against 2.1e-5), a difference of two numbers that small, XFAIL in `KNOWN`. So the elegant build matters at the 1e-8 level of a finite-difference Jacobian, which only a worst-column error can see. Nothing in the archive was regenerated or overwritten, and no number in the paper changes.

## 11. Open decisions

1. **Public or private?** The remote exists with an Apache 2.0 licence. The archived evidence includes GPT output and CLARA-derived numbers; confirm that is releasable before the repository is public.
2. **elegant version.** `ring_tunes` is unaffected (identical to four figures) and `ring_grad` moves in one metric only (Slipstream's worst-column error, 4.7e-5 against 2.1e-5; the Jacobians differ by 7.6e-9). Rebuild at 2026.3.0 if that XFAIL is to go, or if a later case, such as CSR or the Pelegant baseline, shows a difference.
3. **Package tests and CI.** Whether the Slipstream repository adds the markers of section 9 (a change to the package, not done here).
4. **Data deposit.** The raw results (1.8 GB benchmark, 12 GB story output) need a persistent identifier for R4; this repository holds the compact part only.
5. **JuTrack on Linux.** Enzyme, a dependency of JuTrack, does not precompile with Julia 1.10.4 or 1.10.10 on WSL or in Docker (`Missing Julia object for global '+Core.Float64'`, with the default and the generic CPU target); it works on Windows with Julia 1.10.4 and the same Manifest. Options: find the cause (an Enzyme version in the Manifest against Linux LLVM), pin another Enzyme, or keep JuTrack archived on Linux. Until then the plan leaves it out of a plain Linux run. `setup/julia.sh` and `setup/jutrack.sh` are the recipe to retry.
6. **ImpactX.** The `sc_drift` worker needs `beam.at_fixed_t()`, which conda-forge's 26.10 lacks (it came in with `256c808f`, after the release). The paper's run is a source build at `e4279a4b`. Either script the source build (compilers, CMake, AMReX from source: tens of minutes) or wait for the next release and use `envs/impactx.yml`.
7. **Image size.** The extended image is 31.7 GB (the simcodes base is 16.7 GB; SciBmad's conda package alone is 1.2 GB). A second image for the extra codes would halve the pull for someone who needs only ASTRA and elegant; one image keeps one image id in every campaign. Revisit if the pull is a problem.
8. **Report copy of the evidence.** `slipstream_dev/reports/Slipstream/data/` is a copy of `reviews/claims/evidence/` made by `export`; the figure scripts read it. If it should disappear, the scripts need the `validation_repo.py` indirection (two already have it).

## 12. The move from the report folder, 8 November 2026

Decision: every benchmark and validation run for the report is made here, on this PC and on a Linux server; the report folder keeps the
manuscript and reads. This section records what was done and how it was checked, so the next person can tell what is trusted.

### What moved

| From `slipstream_dev/reports/Slipstream/` | To this repository |
| --- | --- |
| `benchmarks/` (117 tracked files) | `reviews/benchmark/`: `common.py`, `analyse*.py`, `cases/`, `workers/` (now with `w_scibmad.py`, `scibmad_server.jl` and the Slipstream 1.1.0 rungs of 7 November), `study.py`, `publish_study.py`, `test_study.py`, 45 control scripts, `prototypes/`, `codes/`, `BENCHMARK_STUDY.md`. `registry.toml` became `bench.toml` plus the uncommitted `bench.local.toml`; `launch/wsl_run.sh` became `launch/conda_run.sh`; the two `chain_*.ps1` pass launchers (Windows paths) were dropped |
| `benchmarks/run.py` | `reviews/benchmark/run.py`, a thin layer over `runner.py` with the same command line and the `run.command / run.plat / run.REG` API the controls use |
| `benchmarks/results/` (3.1 GB untracked) | `results/` (2.2 GB: the 54 campaigns of 26 October to 7 November; not the 867 MB of Slipstream 1.0.0 output, the 22 October rehearsals and two staging folders). The report's copy is left for the owner to delete |
| `data/2026-10-26_R1/` | `reviews/claims/evidence/2026-10-26_R1/` (16 files updated, `scibmad_scaling.json` added); the report's `data/` is now a copy made by `python -m slipstream_validation.export` (a dry run reports 0 differences) |
| `scripts/verify_v8_numbers.py` | `reviews/claims/verify_claims.py` (453 checks; it was 401 before the SciBmad and Slipstream 1.1.0 changes) |

New in this repository for the move: `runner.py` now starts a worker on five platforms (`docker`, `conda`, `wsl`, `windows`, `linux`) behind one interface and resolves
`--config` globs from each worker's `--list`; `plan --check` verifies every planned configuration exists; `envs/*.yml` and `setup/*.sh`; the extended image;
`export.py`; `archived_sources.json`; `.gitattributes`; and tests of the commands each platform builds.

### How it was checked (all on this PC, 8 November, Slipstream 1.1.0 `96607b5`)

1. Tier 0: 8 tests pass; the 453 claims recompute from the current evidence.
2. A control rerun from the copied raw output: `corrected_reference.py` with `SSV_EVIDENCE` pointing at a copy of the evidence writes a `corrected_reference.json` equal to the committed one in every number (only the hash of the script itself differs). `test_study.py`: 25 passed.
3. Workstation platforms (native Windows, WSL; campaigns `ws1`, `ws2`): `ring_tunes` (elegant 2026.3.0, SciBmad, Xsuite, Slipstream, JuTrack), `ring_grad` (all eight code and rung rows, including Slipstream `o6q8s1b8`: `rel_fro` 4.108e-7, equal to the archive) and `linear_fodo` all agree with the archive. With native elegant 2026.3.0 the single `ring_grad` XFAIL of the image does not appear.
4. The extended image (campaign `img2`): SciBmad (Julia 1.12 inside the conda package), Bmad-X (after `scipy` was added to its file: the first run failed with `No module named 'scipy'`), HELIX and the Slipstream FODO rungs run in it and reproduce the archive. The same Bmad-X, SciBmad and HELIX versions as the paper's.
5. The full image campaign `img3` (32 jobs: every code the plan keeps on a plain Linux host) finished; `python -m slipstream_validation.check compare img3` (16 November): 28 rows `ok` or `new`, 3 `XFAIL`, 1 `MOVED`. `ring_tunes`, `sc_drift` (ASTRA `3d64h1.4` is new) and `linear_fodo` reproduce the archive. In `ring_grad` the `rel_col` of Slipstream and of both SciBmad rungs falls by the same 2.2x: the image's elegant 2026.1.0 is the reference (see `KNOWN`), and their `rel_fro` and own AD/FD columns equal the archive. Open: Bmad-X `ad_vs_fd_col` is 1.15e-5 against 1.88e-5 (`ad_vs_fd_fro` 5.0e-9 against 4.2e-9, below the limit; its `rel_fro` against the reference is equal to the WSL run to 2e-9): finite-difference round-off, not yet traced to a package.

### State per code

| Code | Plain Linux host | This workstation | Notes |
| --- | --- | --- | --- |
| Slipstream | docker | windows | CPU rungs only without a card; `gpus = true` adds the GPU rows |
| elegant | docker (2026.1.0) | windows (2026.3.0) | the paper cites 2026.3.0; open decision 2 |
| ASTRA | docker (Linux serial 4.0) | windows (`Astra.exe`) | identical turn-by-turn moments on two rungs |
| Xsuite, Cheetah, Ocelot | docker | wsl, windows, windows | GPU Xsuite: `envs/xsuite-gpu.yml` |
| SciBmad, Bmad-X, HELIX | docker | windows, wsl, windows | the image carries them (`/opt/envs/*`) |
| JuTrack | archived | windows | Enzyme does not precompile on Linux (open decision 5) |
| ImpactX | archived | wsl (source build) | needs `beam.at_fixed_t()` (open decision 6) |
| OPAL-X | skip | wsl (CUDA source build, CPU image through docker) | `setup/install_opalx.sh`; `sc_drift` only; not in the paper yet |
| GPT | archived | windows | licence; install it yourself and set `platform`, `BENCH_GPT_BIN`, `GPTLICENSE` |

### Not moved

`Stories/CLARA_InjectorConvergence/` and `slipstream_dev/benchmark/` (the injector drivers, 12 GB of output) stay in the workbench until the injector review (phase 5). Six control scripts that read them
(`clara_release_bridge.py`, `clara_gpt_profile.py`, `ibs_elena_bridge.py`, `injector_*_summary.py`) are here and need `SSV_WORKBENCH`. `scripts/validate_vN.py` (draft checks), the figure makers and
the two audits of the 26 October study stay in the report.

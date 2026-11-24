# benchmark review

The cross-code benchmark behind the Slipstream paper: four frozen cases, one worker per code, the numerical controls that sit
around them, and the gates that compare a rerun with the archived campaign. Since 8 November 2026 this is the **only** place where
benchmark and validation runs for the report are made. The report folder (`slipstream_dev/reports/Slipstream/`) keeps the
manuscript and reads the compact results from `reviews/claims/evidence/`.

| Case | Question | Codes | Gates |
| --- | --- | --- | --- |
| `ring_tunes` | amplitude-dependent tunes of a 48 m ring over 1024 turns | Slipstream, elegant, Xsuite, Cheetah, JuTrack, SciBmad, Bmad-X | G1 conventions, G2 nonlinear ring, G3 precision |
| `ring_grad` | five-knob Jacobian by automatic differentiation against finite differences | Slipstream, elegant, Xsuite, Cheetah, JuTrack, SciBmad, Bmad-X | G11 |
| `sc_drift` | free-drift space charge on one frozen bunch | Slipstream, ASTRA, Cheetah, Ocelot, ImpactX, GPT, OPAL-X | G4 |
| `linear_fodo` | proton FODO, analytic control | Slipstream, HELIX | G1 |

`cases.toml` lists the configurations a campaign runs (`cpu`; `gpu` when the host has a card), `workers.toml` the worker of each
code, `bench.toml` how each code is reached. The complete frozen design of the paper (more rungs, GPU, three processes per rung)
is `study.py`; the ladders and tables it produced are in [BENCHMARK_STUDY.md](BENCHMARK_STUDY.md). Installation notes and the
benchmark standard frame (how each code's coordinates map to the common ones) are in [codes/CODES.md](codes/CODES.md).

## Where a code runs

Each code has a `platform` in `bench.toml` (committed; the defaults of a Linux host with Docker) and `bench.local.toml` (not
committed; one machine). The worker, the cases and the result layout are the same on every platform, so the same campaign tag
means the same thing on every host.

| Platform | Used for | Set up with |
| --- | --- | --- |
| `docker` | the whole stack in one image: ASTRA, elegant, Xsuite (CPU), Cheetah, Ocelot from simcodes, plus Bmad-X, ImpactX, SciBmad, JuTrack (Julia 1.10) and HELIX as conda environments built from [envs/](../../envs/) | `docker build -t slipstream-validation:local -f docker/Dockerfile .` |
| `conda` | a Linux host without Docker; the same environment files | `bash setup/install_envs.sh` |
| `wsl` | a Windows host with the codes in a WSL distribution (Xsuite with a GPU, Bmad-X, ImpactX) | `bash setup/install_envs.sh` inside WSL |
| `windows` | a Windows host with native codes (elegant and ASTRA executables, Julia, GPT) | paths in `bench.local.toml` |
| `archived` | a code the host cannot run (GPT without a licence): its frozen result is the reference | nothing |

GPT is the one code that must be installed by hand (licence). Set `platform = "linux"` (or `"windows"`), `BENCH_GPT_BIN` and the
`GPTLICENSE` variable in `bench.local.toml`; until then the plan leaves GPT out and `check` compares the other codes.

Two profiles of the development workstation are kept as uncommitted files: `bench.local.toml` (native Windows plus WSL, the setup
of the paper's timings) and `bench.image.local.toml` (the Docker image only, `SSV_BENCH=bench.image.local.toml`).

## Run

```bash
python -m slipstream_validation.runner checkout ../slipstream HEAD --name head          # git archive of the commit under test
python -m slipstream_validation.runner plan benchmark --check                          # the jobs, and that every worker knows its configurations
python -m slipstream_validation.runner run benchmark --execute --tag pc2 --slipstream checkouts/slipstream-head
python -m slipstream_validation.check compare pc2                                      # against the archived campaign
```

`--case`, `--code` and `--config` (names or globs) narrow a run; `--procs 3` repeats each configuration in three independent
processes, as the paper's timings did (accuracy needs one). A tag that exists resumes: finished jobs are skipped. A campaign for the
report uses the same commands with a new tag; the old `python run.py --case C --campaign TAG --codes a,b --configs finest` form still
works (it is a thin layer over the runner) and is what the control scripts use. Timing campaigns: never run two at once, and keep GPU
use within the 8 GiB card.

## Controls

The control scripts answer one question each, read `results/<campaign>/` (raw arrays, local, not committed; 2.2 GB of the paper's
campaigns were copied here on 8 November 2026) and write compact JSON to `reviews/claims/evidence/<campaign>/`, which `reviews/claims`
recomputes the quoted numbers of the paper from. A control rerun for the paper's campaign changes tracked files: `git diff` is the audit
trail, and a changed number needs a new campaign tag. Run them from this folder with the Python of an environment that has the codes
they use; set `SSV_SLIPSTREAM` to the Slipstream checkout under test. `SSV_RESULTS` and `SSV_EVIDENCE` move the input and output folders
(`corrected_reference.py` run with `SSV_EVIDENCE` pointing at a copy of the evidence reproduces `corrected_reference.json` exactly, 8 November 2026).

| Question | Scripts |
| --- | --- |
| Adapter and reference correctness | `corrected_reference.py`, `bend_attribution.py`, `xsuite_model_control.py`, `mass_fix_control.py`, `element_attribution.py`, `element_substitution_control.py`, `probe_elements.py` |
| Precision and device | `gpu_control.py`, `cpu_control.py`, `precision_control.py`, `precision_probe.py`, `xsuite_gpu_scaling.py`, `xsuite_gpu_scaling_analysis.py`, `xsuite_gpu_profile.py` |
| Space-charge references and convergence | `astra_refinement.py`, `native_mesh_control.py`, `native_box_control.py`, `space_charge_box_probe.py`, `injector_step_control.py`, `richardson_reference.py`, `convergence_extension.py`, `longitudinal_gates.py`, `injector_scaling_control.py`, `charge_off_control.py`, `astra_mesh_cap_probe.py`, `astra_particle_energy_control.py` |
| Derivatives and optimisation | `gradient_acceptance.py`, `fd_ad_endpoint_control.py`, `fd_ad_knob_ladder.py`, `optimisation_budget_control.py`, `optimisation_convergence_control.py`, `injector_fd_analysis.py`, `injector_fd_full_analysis.py` |
| Cost, start-up and memory | `injector_cost_control.py`, `startup_accounting.py`, `memory_tree_control.py`, `memory_wsl_control.py`, `particle_scaling_control.py` |
| SciBmad | `scibmad_scaling.py`, `scibmad_scaling_analysis.py`; the Slipstream step-model prototypes are in `prototypes/` |
| CLARA and injector studies (need the workbench: `SSV_WORKBENCH` is the `neg_coating_rwi` checkout, and its 12 GB of story output) | `clara_release_bridge.py`, `clara_gpt_profile.py`, `injector_ad_summary.py`, `injector_optimiser_summary.py`, `injector_tolerance_summary.py`, `ibs_elena_bridge.py` |

The injector derivative, jitter and tuning studies of the paper (Sec. IV D and E, campaign `2026-11-21_injector_image`, CLARA 5 pC deck, cathode image on) are run
here. The drivers are `injector_clara_ad.py` (autograd against finite differences, `--image on`; `--fd-knobs` shards the knobs and `--ad-json` shares the one
autograd pass between them), `injector_clara_profile.py`, `injector_clara_energy_spread.py`, `injector_clara_tolerance.py`, `injector_clara_optimiser.py` (timing: run it
alone, three methods at once) and `injector_clara_memory.py` (GPU). They read the deck from the workbench (`SSV_WORKBENCH`). The summaries `injector_fd_full_analysis.py`,
`injector_ad_summary.py`, `injector_tolerance_summary.py` and `injector_optimiser_summary.py` take `--campaign TAG` and write `reviews/claims/evidence/TAG/`; without it they
rebuild the archived image-off numbers of `2026-10-26_R1`. **Memory:** reverse mode over the whole line holds 9 to 14 GiB of host memory per process (the autograd graph
grows with the steps even with checkpointing); forward-only jobs hold under 1 GiB. Run every long CPU job under `memguard.py --max-gib N`, one reverse-mode job at a time:
ten at once overran the 63 GiB of the workstation, whose pagefile is on a hard disk, and made it unusable (21 November 2026). `injector_clara_memory.py` caps itself
below the free GPU memory and skips a run without checkpointing that would not fit, because a run past the card spills into host memory (3 November 2026).

`publish_study.py` writes the per-case summaries of a campaign to the evidence folder and refreshes `BENCHMARK_STUDY.md`; `test_study.py`
holds the numerical controls of the harness itself (no external code). `launch/conda_run.sh` is how `wsl` and `conda` workers start;
`launch/wsl_tree_rss.py` samples the resident set of a worker tree for the memory controls.

## Gates

`check compare TAG` is the gate of a rerun. It analyses every case in the campaign with the case's own script and compares it with the
archived campaign: a metric agrees when it moved by no more than max(25 % of the archived value, a floor) from the archived value in
`reviews/claims/evidence/2026-10-26_R1/`. This is a ratchet from one host pair, to be re-measured on a second host. The physics gate of
each case (tune error up to 1e-5, first-turn residual up to 1e-8, Jacobian agreement) is in the case's json and in `PLAN.md` of the
report folder (G1 to G13). `CASES` in `check.py` says which columns are gated, the floor of each case and its absolute limits:

| Case | Gated columns (ratchet) | Floor | Also |
| --- | --- | --- | --- |
| `ring_tunes` | `dq_ref` | 1e-9 | |
| `ring_grad` | `ad_vs_fd_fro`, `ad_vs_fd_col`, `rel_fro`, `rel_col`, `dy0` | 1e-9 | `ad_vs_fd_fro` must stay below `jac_rel` = 1e-5 (G11), whatever the archive says |
| `sc_drift` | `size`, `length`, `emit`, `spread` against the reference, `dEkin_change_eV` | 1e-9 | `pass` flag equal |
| `linear_fodo` | `position_abs_m`, `momentum_abs` | 1e-15 | `pass` flag equal; `cpu-f64-o4k4b8` fails the 1e-11 gate by design, in the archive too |

Rows: `ok`, `new` (a rung with no archived row on purpose, `NOREF_OK`), `NOREF` (no archived row and not allowed: a hole), `MOVED`,
`LIMIT`, `FAILED`, and `XFAIL` (moved, with a reason in `KNOWN`). Exit status is 1 for `MOVED`, `LIMIT` and `FAILED`.
`tests/test_framework.py` checks that every planned job has an archived row and that the gate can fail. A Slipstream CPU float64 row
without a CPU archive row is compared with the archived GPU row of the same model (`ALIASES`; printed as `[vs ...]`). Rung-to-rung
columns (`*_vs_prev`) are not gated because they depend on which rungs a campaign ran. Wall times are printed beside the archived ones and
never gated.

The archived campaign is the paper's `2026-10-26_R1` as corrected on 27 October (elegant `p_central`) and rerun for Slipstream 1.1.0 on
7 November (per-kind kick counts, the ring setting `o6q8s1b8`), with SciBmad added in place of Bmad-X: `corrected_reference.json` holds the
ring tables, the CSVs the space-charge and FODO cases. The earlier `ring_tunes.csv` and `ring_grad.csv` of 26 October carry an elegant
reference that was wrong and are not used for `ring_grad`.

## What is kept from the move

The workers, `common.py` and the analysis scripts are the report folder's, with machine paths removed and `RESULTS`/`EVIDENCE`
constants instead of folders next to the script. Results record the hash of the file they ran with, so
[archived_sources.json](archived_sources.json) holds the hashes of the originals of the files that were edited, and a reanalysis of a
retained campaign checks against that (`common.archived_source_sha256`). Case files are byte-for-byte the originals: the hashes in the archived results were taken on a Windows checkout (CRLF for some of them), so the reanalysis
controls that assert them are meant for that workstation, while a new campaign records the hashes of the files it ran with. The 867 MB of Slipstream 1.0.0 results (`_superseded_slipstream_v100`),
the rehearsals of 22 October and the staging folders were not copied.

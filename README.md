# slipstream-validation

Cross-code validation, benchmarks and archived evidence for [Slipstream](https://github.com/astec-stfc/slipstream).

The unit tests in the Slipstream repository check the package on its own and need no other code. This repository holds
everything that needs more: the other codes (ASTRA, elegant, Xsuite, Ocelot, Cheetah, GPT), the frozen benchmark cases, the
CLARA injector comparisons and the saved results that every number in the paper is recomputed from. Every check is a **gate**:
a quantity, a reference and a tolerance. A gate that fails means a bug, or a documented exception written next to it.

It follows [simba-integration-tests](https://github.com/astec-stfc/simba-integration-tests): a review is a folder, the codes run
in one Docker image, a campaign is kept under a tag, and `check` compares a campaign with another.

## Status

8 November 2026: this repository is the **only** place where benchmark and validation runs for the Slipstream report are made. The
harness, the frozen cases, the workers for all eleven codes, the 40-odd control scripts and the campaign output of the report folder
moved here (see [reviews/benchmark](reviews/benchmark/)). Tier 0 runs the 453 quoted-number checks of the paper against the current
evidence and the framework tests, with numpy only. A campaign runs on this PC through four platforms (Docker image, WSL, native
Windows, conda on Linux) and `check compare` compares it with the paper's archived campaign. The rest is planned in
[DESIGN.md](DESIGN.md), which also says what a second PC can and cannot reproduce.

## Quick start (any PC, no simulation code)

```bash
git clone https://github.com/astec-stfc/slipstream-validation
cd slipstream-validation
pip install numpy pytest
pytest -m tier0
```

This recomputes every number quoted in the paper from `reviews/claims/evidence/` and writes `results/claims.csv`.

## Gates against Slipstream (any PC, no simulation code)

```bash
SSV_SLIPSTREAM=../slipstream pytest -m slipstream      # without the variable: the installed package
```

CI runs the fast ones on every push against Slipstream `develop`. The `Nightly` workflow runs all of them and a campaign in the
simulation image; start it by hand to test another branch or a pull request (`refs/pull/N/head`).

## Running the codes (a Linux host with Docker)

```bash
docker pull ghcr.io/astec-stfc/simcodes-docker:master
docker build -t slipstream-validation:local -f docker/Dockerfile .      # the simcodes image plus a conda environment per extra code (envs/*.yml)
python -m slipstream_validation.runner checkout ../slipstream HEAD --name head
python -m slipstream_validation.runner plan benchmark --check            # the jobs, and that every worker knows its configurations
python -m slipstream_validation.runner run benchmark --execute --tag pc2 --slipstream checkouts/slipstream-head
python -m slipstream_validation.check compare pc2
```

`bench.toml` says how each code is reached; copy it to `bench.local.toml` (not committed) to change it for one machine. The same
campaign runs on other platforms by changing a code's `platform` there:

| Platform | When | Set up with |
| --- | --- | --- |
| `docker` | default; the whole stack in one image, GPU with `gpus = true` | `docker build` as above |
| `conda` | a Linux host without Docker | `bash setup/install_envs.sh` (one `environment.yml` per code family in [envs/](envs/), plus the official Julia and JuTrack) |
| `wsl` | the runner is on Windows, the codes in a WSL distribution | `bash setup/install_envs.sh` inside WSL |
| `windows` | native Windows executables and Julia (this workstation's paper setup) | paths in `bench.local.toml`, see `bench.local.example.toml` |
| `archived` | a code the host cannot run: GPT (licence) unless you install it | nothing |

GPT is the one code that needs a manual install (licence): set its `platform` and `BENCH_GPT_BIN` in `bench.local.toml` and export
`GPTLICENSE`. ASTRA and elegant come with the simcodes image (or a native simcodes install, `BENCH_ASTRA_EXE` and `BENCH_ELEGANT_EXE`).
To refresh the report folder after a campaign: `python -m slipstream_validation.export <report folder>`.

## Reviews

| Review | What it checks | Needs | State |
| --- | --- | --- | --- |
| [claims](reviews/claims/) | every number quoted in the paper follows from the archived evidence (453 checks) | numpy | runs (tier 0) |
| [benchmark](reviews/benchmark/) | the four frozen cases (ring tunes, five-knob Jacobian, space-charge drift, linear FODO) across eleven codes, and the numerical controls around them | the image, or the platforms above | runs; `check compare` against the archived campaign. Workstation smoke campaigns reproduce the archive (elegant, SciBmad, Xsuite, HELIX, Slipstream FODO); see its README for the state per code |
| injector | CLARA 20, 100 and 250 pC against refined ASTRA and GPT | the image, GPT archived | planned |

## Licence

Apache 2.0, see [LICENSE](LICENSE).

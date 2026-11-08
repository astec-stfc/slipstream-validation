# Working in slipstream-validation

For coding agents and the people who drive them. [README.md](README.md) says how to run things, [DESIGN.md](DESIGN.md) why it is built
this way. This file holds the rules and traps that are not in the code.

## What this repository is for

It shows, with gates, how far Slipstream agrees with other codes and with its own archived evidence, on this PC and on others. It must
**show disagreements, not hide them**. A bug in Slipstream is never worked around here: it becomes an issue in `astec-stfc/slipstream`
and the failing case stays as `xfail` with the issue number in the reason.

## Rules

- **Runs for the report are made here**, never in `slipstream_dev/reports/Slipstream/` (since 8 November 2026). That folder only reads: `python -m slipstream_validation.export <report folder>` copies the compact evidence into its `data/`, and two figure scripts find this repository through `scripts/validation_repo.py` (`SLIPSTREAM_VALIDATION`).
- **Slipstream is a peer, not the reference.** ASTRA, GPT, elegant and the others are compared as peers; a gap is reported in percent with the denominator stated. Keep the tone non-harsh.
- **Do not loosen a tolerance to make a code pass.** A tolerance is a target (what the physics should give) or a ratchet (about twice today's spread, to catch regressions); say which.
- **Accuracy reproduces, timing does not.** Never gate a wall time across hosts; gate a ratio on one host and report the host.
- **Archived evidence is read-only.** `reviews/*/evidence/` is written by a campaign, never edited by hand. A changed number needs a new campaign name.
- **No machine path in a committed file.** Paths live in `bench.local.toml`.
- **Never exceed the 8 GiB GPU.** The driver spills to RAM and the PC freezes. Cap with `torch.cuda.set_per_process_memory_fraction` and skip a run by a probe.
- **No dependency on the Cheetah package in new Slipstream work.** Cheetah here is a code to compare with, not a library to build on.
- **Do not run two timing campaigns at once**, and do not edit `reviews/*/workers/` while one runs.
- **Commit, push, publish and file issues only when the person asks.**

## Traps

- Slipstream's repository is private, so the image cannot clone it. Mount a `git archive` export of the commit under test (`checkouts/slipstream-NAME`, `[paths].slipstream`), never a working tree: on the Windows workstation `slipstream/examples` is a junction to `slipstream_examples`, which Docker does not follow (21 false test failures, 6 November 2026).
- A background job started with `nohup ... &` inside a `wsl.exe` call is killed when the call returns. Run the whole `wsl.exe` command as a background task of your tool, and write its output to a file with `python -u`.
- The conda environment on the Windows workstation imports Slipstream from `C:\...\Documents\GitHub\slipstream`, not from the `neg_coating_rwi` submodule. Set `PYTHONPATH` to the checkout under test.
- On Windows, set `CXX=clang-cl` before any `torch.compile` run. In the image `g++` is used and needs nothing.
- The image carries elegant 2026.1.0; the paper cites 2026.3.0. A result records the version it came from. The two differ by 7.6e-9 in the `ring_grad` finite-difference Jacobian (endpoint state 6e-14), visible only in a worst-column error.
- The tables of `2026-10-26_R1` (`ring_grad.csv`, `ring_tunes.csv`) are the superseded elegant reference (before the 27 October `p_central` fix). Compare with `corrected_reference.json`; a "moved reference" in an archived table is first a question of which table.
- `wsl.exe` expands `$VAR` in its arguments: put multi-command WSL work in a script. Set `MSYS_NO_PATHCONV=1` for Docker paths from Git Bash.
- Docker on the workstation is a native `dockerd` inside Ubuntu-22.04 (no systemd), not Docker Desktop. Start it as a background task: `wsl.exe -d Ubuntu-22.04 -u root -- dockerd`. Image builds: `docker build -t slipstream-validation:local -f docker/Dockerfile .` from the repository root (the context is the root; `.dockerignore` lets only `envs/` and `setup/` in). Keep `slipstream-validation:pc1` as the known-good fallback tag.
- Python's `Path.write_text` on Windows writes CRLF. `setup/*.sh`, `launch/conda_run.sh` and the `Dockerfile` must be LF (`set: pipefail: invalid option name` in the build is the symptom); write them with the Write tool, or give `write_text` a `newline` argument, or write bytes. Case files must not be re-saved or re-encoded: results record their hashes, and the archived ones were taken on a Windows (CRLF) checkout; `.gitattributes` only fixes the line endings of `*.sh` and the `Dockerfile`. `grep` in Git Bash ignores the CR and reports 0 for a CRLF file; count bytes in Python.
- A reanalysis control that checks the hash of a worker compares with `reviews/benchmark/archived_sources.json` (the originals), not with the file now: the workers lost their machine paths in the move (`common.archived_source_sha256`).
- A campaign that lacks the reference rung (elegant `o6s16b32` for the rings, ASTRA `3d32h1.4` or `3d64h1.4` for `sc_drift`) is not compared: the analysis would measure every row against some other reference. `check compare` says so instead of printing a false MOVED.
- Controls write into `reviews/claims/evidence/`, which the paper's numbers are recomputed from. Rerunning one for `2026-10-26_R1` changes tracked files; look at `git diff` and use a new campaign tag for a changed number. `SSV_EVIDENCE` redirects the output (copy the evidence folder first: a control also reads files it did not write).
- Heredocs in the shell tool mangle backslashes and the backslash-n of an f-string (it becomes a real line break); write scripts with the Write tool and run them.

## Style

Files, commit messages and issues are plain English prose. Keep diffs small.

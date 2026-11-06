# Working in slipstream-validation

For coding agents and the people who drive them. [README.md](README.md) says how to run things, [DESIGN.md](DESIGN.md) why it is built
this way. This file holds the rules and traps that are not in the code.

## What this repository is for

It shows, with gates, how far Slipstream agrees with other codes and with its own archived evidence, on this PC and on others. It must
**show disagreements, not hide them**. A bug in Slipstream is never worked around here: it becomes an issue in `astec-stfc/slipstream`
and the failing case stays as `xfail` with the issue number in the reason.

## Rules

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
- The image carries elegant 2026.1.0; the paper cites 2026.3.0. A result records the version it came from.
- `wsl.exe` expands `$VAR` in its arguments: put multi-command WSL work in a script. Set `MSYS_NO_PATHCONV=1` for Docker paths from Git Bash.

## Style

Files, commit messages and issues are plain English prose. Keep diffs small.

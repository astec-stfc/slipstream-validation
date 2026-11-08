"""Run a benchmark campaign: the command line and the small API the control scripts were written against.

    python run.py --case ring_tunes --campaign 2026-11-08_R2 --codes slipstream,elegant --configs finest --procs 3 --repeats 3
    python run.py --case ring_scale_n500 --campaign scratch --codes xsuite --configs "cuda-*" --procs 1 --repeats 1

This is a thin layer over ``slipstream_validation.runner``, which knows the platforms (docker, conda, wsl, windows, linux) from
bench.toml / bench.local.toml.  Every (code, config) is an independent process, repeated ``--procs`` times, all jobs interleaved in a
seeded random order.  ``--configs``: ``finest`` (each code's converged rung), ``all``, or names/globs.  Existing ok results are skipped,
so an interrupted campaign resumes.  The Slipstream under test is ``$SSV_SLIPSTREAM`` or ``[paths].slipstream`` of bench.toml.
"""
import argparse
import os
import pathlib
import sys
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import common  # noqa: E402
from slipstream_validation import runner  # noqa: E402

BENCH = runner.load_bench()
REG = {code: dict(cfg) for code, cfg in BENCH.get("codes", {}).items()}   # per-code settings: platform, python, env, timeout, ...
WORKERS = tomllib.loads((HERE / "workers.toml").read_text())
CASES = tomllib.loads((HERE / "cases.toml").read_text())
_slip = os.environ.get("SSV_SLIPSTREAM") or BENCH.get("paths", {}).get("slipstream", "")
SLIPSTREAM = pathlib.Path(_slip).resolve() if _slip else None
WORKSPACE = SLIPSTREAM  # the checkout under test (the old name)


def target(code):
    return runner.Target(BENCH, code, SLIPSTREAM, common.RESULTS, "benchmark")


def plat(code, p) -> str:
    """Host path ``p`` as the code's worker sees it."""
    return target(code).path(p)


def worker_path(code, case_id):
    """Worker file (relative to this folder) of ``code`` for the kind of ``case_id``; None if the code has none."""
    return WORKERS.get(code, {}).get(runner.kind_of(CASES, case_id))


def command(code, args, threads, case_id):
    """(argv, environment) that starts the worker of ``code`` for ``case_id`` with ``args`` in the worker's path style."""
    return target(code).argv(HERE / worker_path(code, case_id), args, threads)


def list_configs(code, case_id):
    return runner.list_configs(BENCH, "benchmark", WORKERS, code, runner.kind_of(CASES, case_id), SLIPSTREAM, common.RESULTS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--codes", help="comma-separated; default: every code with a worker for the case that runs on this machine")
    ap.add_argument("--configs", default="finest")
    ap.add_argument("--procs", type=int, default=3, help="independent process runs per (code, config)")
    ap.add_argument("--repeats", type=int, default=3, help="warm repeats inside one process")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads for BLAS/OpenMP in every worker")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=None, help="seconds per job; default: the code's `timeout` in bench.toml, else 3600")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print the jobs and start nothing")
    a = ap.parse_args(argv)
    kind = runner.kind_of(CASES, a.case)
    codes = a.codes.split(",") if a.codes else [c for c in WORKERS if kind in WORKERS[c] and runner.platform_of(BENCH, c) not in ("archived", "skip")]
    cmd = ["run", "benchmark", "--case", a.case, "--tag", a.campaign, "--config", a.configs, "--procs", str(a.procs), "--repeats", str(a.repeats),
           "--threads", str(a.threads), "--seed", str(a.seed)]
    for c in codes:
        cmd += ["--code", c]
    if SLIPSTREAM:
        cmd += ["--slipstream", str(SLIPSTREAM)]
    if a.timeout:
        cmd += ["--timeout", str(a.timeout)]
    if a.force:
        cmd += ["--force"]
    if not a.plan:
        cmd += ["--execute"]
    runner.main(cmd)


if __name__ == "__main__":
    main()

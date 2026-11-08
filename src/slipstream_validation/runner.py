"""Plan and run the cases of a review, one isolated worker process per job.

    python -m slipstream_validation.runner checkout ../slipstream HEAD --name head     # git archive export of the commit to test
    python -m slipstream_validation.runner plan benchmark --case ring_tunes            # prints the jobs, starts nothing
    python -m slipstream_validation.runner run benchmark --execute --tag pc1 --slipstream checkouts/slipstream-head
    python -m slipstream_validation.runner run benchmark --execute --tag t --case ring_scale_n500 --code xsuite --config "cuda-*"

Results go to results/TAG/CASE/CODE/CONFIG/pK in the layout the analysis scripts of the review read.  Jobs run one at a time
(timing); an interrupted campaign resumes because a finished job (status ok) is skipped.

bench.toml says how each code is reached, bench.local.toml (not committed; or the file named by $SSV_BENCH) overrides it for one machine.  A code runs on one of
five platforms, and the worker, the cases and the result layout are the same on all of them:

    docker    the validation image (Docker on this machine, or in WSL when the runner is on Windows)
    conda     a conda environment on this Linux machine (environment.yml files in envs/)
    wsl       a conda environment in a WSL distribution; the runner is on Windows
    windows   a native Windows interpreter and executables
    linux     the Python that runs the runner
    archived  the code cannot run here; its frozen result is the reference
"""
import argparse
import fnmatch
import json
import os
import pathlib
import platform as _platform
import random
import subprocess
import sys
import time
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
GLOB = set("*?[")


def load_bench() -> dict:
    cfg = tomllib.loads((ROOT / "bench.toml").read_text())
    local = pathlib.Path(os.environ["SSV_BENCH"]) if os.environ.get("SSV_BENCH") else ROOT / "bench.local.toml"   # SSV_BENCH: another profile
    if local.exists():
        def merge(a, b):
            for k, v in b.items():
                a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
            return a
        cfg = merge(cfg, tomllib.loads(local.read_text()))
    return cfg


def review_dir(review: str) -> pathlib.Path:
    return ROOT / "reviews" / review


def platform_of(bench, code) -> str:
    return bench.get("codes", {}).get(code, {}).get("platform", bench.get("platform", "docker"))


def kind_of(cases, case_id) -> str:
    """Kind of a case: from cases.toml, else from the id (ring_scale_n500 is a ring case, sc_drift_n2000s1 an sc_drift case)."""
    if case_id in cases:
        return cases[case_id]["kind"]
    for k in ("ring", "sc_drift", "linear_fodo"):
        if case_id == k or case_id.startswith(k + "_"):
            return k
    raise ValueError(f"unknown case {case_id}: add it to cases.toml")


# ------------------------------------------------------------------------------------------------ where a worker runs
class Target:
    """How to start one code's worker.  ``path`` maps a host path to the path the worker sees; ``argv`` builds the command."""

    def __init__(self, bench, code, slip=None, runs=None, review="benchmark"):
        self.bench, self.code, self.review = bench, code, review
        self.cfg = bench.get("codes", {}).get(code, {})
        self.kind = platform_of(bench, code)
        self.slip = pathlib.Path(slip).resolve() if slip else None
        self.runs = pathlib.Path(runs).resolve() if runs else pathlib.Path(bench.get("docker", {}).get("runs") or ROOT / "results").resolve()
        self.docker = bench.get("docker", {})
        self.wsl = bench.get("wsl", {})
        self.conda = bench.get("conda", {})

    # -- paths
    def path(self, p) -> str:
        p = pathlib.Path(p)
        if self.kind == "docker":
            for host, inside in ((self.runs, "/runs"), (self.slip, "/slipstream"), (ROOT, "/work")):   # results may sit inside the repository: most specific first
                if host is not None:
                    try:
                        return inside + ("/" + "/".join(p.resolve().relative_to(host).parts) if p.resolve() != host else "")
                    except ValueError:
                        pass
            raise ValueError(f"{p} is not under the repository, the results folder or the Slipstream checkout")
        if self.kind == "wsl":
            q = pathlib.PureWindowsPath(p.resolve())
            return f"/mnt/{q.drive[0].lower()}/" + "/".join(q.parts[1:])
        return str(p.resolve())

    def docker_host_path(self, p) -> str:
        """The path as the Docker daemon sees it: /mnt/c/... when Docker is in WSL and the runner is on Windows."""
        p = pathlib.Path(p).resolve()
        if os.name == "nt" and self.docker.get("wsl_distro"):
            return f"/mnt/{p.drive[0].lower()}/" + "/".join(p.parts[1:])
        return str(p)

    # -- environment
    def env(self, threads) -> dict:
        e = {**{k: str(threads) for k in THREAD_VARS}, "SSV_RESULTS": self.path(self.runs), "PYTHONDONTWRITEBYTECODE": "1"}
        if self.slip is not None:
            e["PYTHONPATH"] = self.path(self.slip) + ("/src" if self.kind in ("docker", "wsl") else os.sep + "src")
            e["SSV_SLIPSTREAM"] = self.path(self.slip)
        if self.kind == "docker":
            e["SSV_EVIDENCE"] = "/work/reviews/claims/evidence"
        e.update({k: str(v) for k, v in self.cfg.get("env", {}).items()})
        return e

    # -- command
    def argv(self, script, args, threads=1):
        """(argv, process environment) that runs ``script`` (a host path) with ``args`` (already in the worker's path style)."""
        env, script = self.env(threads), self.path(script)
        if self.kind == "docker":
            d = self.docker
            cmd = ["docker", "run", "--rm", "--entrypoint", self.cfg.get("python", "python")]
            if d.get("gpus"):
                cmd += ["--gpus", "all"]
            user = d.get("user") or (f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") and not d.get("wsl_distro") else "")
            if user:
                cmd += ["--user", user, "-e", "HOME=/tmp"]
            mounts = [(ROOT, "/work", "ro"), (self.runs, "/runs", "rw")] + ([(self.slip, "/slipstream", "ro")] if self.slip else [])
            for host, inside, mode in mounts:
                cmd += ["-v", f"{self.docker_host_path(host)}:{inside}:{mode}"]
            for k, v in env.items():
                cmd += ["-e", f"{k}={v}"]
            # the repository is read-only and Xsuite compiles its kernels into the working directory, so work in /tmp
            cmd += ["-w", "/tmp", self.cfg.get("image", d["image"]), "-u", script, *args]
            return (["wsl.exe", "-d", d["wsl_distro"], "--", *cmd] if d.get("wsl_distro") else cmd), os.environ.copy()
        if self.kind in ("wsl", "conda"):
            kv = [f"{k}={v}" for k, v in env.items()]
            launcher = self.path(ROOT / "reviews" / self.review / "launch" / "conda_run.sh")
            base = self.conda.get("base") or self.wsl.get("conda_base") or "~/miniforge3"
            cmd = ["bash", launcher, self.cfg["conda_env"], str(threads), f"SSV_CONDA_BASE={base}", *kv, "--", script, *args]
            if self.kind == "wsl":
                cmd = ["wsl.exe", "-d", self.wsl.get("distro") or self.cfg["distro"], "--", *cmd]
            return cmd, os.environ.copy()
        full = dict(os.environ, **env)
        py = self.cfg.get("python") or sys.executable
        return [py, "-u", script, *args], full


def worker_script(review, workers, code, kind) -> pathlib.Path:
    return review_dir(review) / workers[code][kind]


# ------------------------------------------------------------------------------------------------ plan
def list_configs(bench, review, workers, code, kind, slip=None, runs=None):
    """Configurations a worker offers on this machine (its ``--list``), in rank order."""
    t = Target(bench, code, slip, runs, review)
    cmd, env = t.argv(worker_script(review, workers, code, kind), ["--list"], 1)
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("[")]
    if not line:
        raise RuntimeError(f"{code}: cannot list configs\n{out.stdout[-500:]}\n{out.stderr[-1500:]}")
    return [c["name"] for c in json.loads(line[-1])]


def select(names, spec):
    """``finest`` (the converged rung), ``all``, or names/globs."""
    if spec == "all":
        return names
    if spec == "finest":
        return names[-1:]
    return [n for n in names if any(fnmatch.fnmatch(n, s) for s in spec.split(","))]


def jobs_for(review, bench, case=None, code=None, config=None, procs=1, seed=1, slip=None, runs=None):
    """(case, code, config, k) for every job.  Without ``config`` the configurations are those of cases.toml (codes whose platform
    is archived or skip are left out); with ``config`` (names, globs, ``all``, ``finest``) they are matched against what the worker
    lists on this machine, so a case that is not in cases.toml (ring_scale_n500) and a code that is is not listed for a case can be run."""
    d = review_dir(review)
    cases = tomllib.loads((d / "cases.toml").read_text())
    workers = tomllib.loads((d / "workers.toml").read_text())
    gpu = bool(bench.get("docker", {}).get("gpus"))
    config = [x for c in config for x in c.split(",")] if config else None   # --config a,b and --config a --config b are the same
    jobs = []
    for cid in case or list(cases):
        kind = kind_of(cases, cid)
        table = {}
        if cid in cases:
            table = dict(cases[cid]["cpu"])
            if gpu:
                for k, v in cases[cid].get("gpu", {}).items():
                    table[k] = table.get(k, []) + v
        for cd in code or list(table):
            plat = platform_of(bench, cd)
            if plat in ("archived", "skip"):
                continue
            if kind not in workers.get(cd, {}):
                raise ValueError(f"no {kind} worker for {cd} in workers.toml")
            if config is None:
                names = table.get(cd, [])
            elif all(not (set(c) & GLOB) and c not in ("all", "finest") for c in config):
                names = [n for n in config if not table.get(cd) or n in table[cd] or code]
            else:
                names = select(list_configs(bench, review, workers, cd, kind, slip, runs), ",".join(config))
            jobs += [(cid, cd, n, k) for n in names for k in range(procs)]
    random.Random(seed).shuffle(jobs)
    return jobs, cases, workers


def image_id(bench) -> str | None:
    d = bench["docker"]
    cmd = ["docker", "image", "inspect", "--format", "{{.Id}}", d["image"]]
    if d.get("wsl_distro"):
        cmd = ["wsl.exe", "-d", d["wsl_distro"], "--", *cmd]
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


def cpu_name() -> str | None:
    try:
        if os.name == "nt":
            r = subprocess.run(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True)
            return r.stdout.strip() or None
        for ln in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return _platform.processor() or None


def git_state(path) -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True).stdout.strip()
    return dict(path=str(path), commit=g("rev-parse", "HEAD") or None, dirty=bool(g("status", "--porcelain")))


def check_names(bench, review, cases, workers, jobs, slip, runs) -> int:
    """Ask each (code, kind) worker for its configurations and report planned names it does not have.  Returns the number missing."""
    pairs = sorted({(cd, kind_of(cases, cid)) for cid, cd, _, _ in jobs})
    bad = 0
    for cd, kind in pairs:
        have = set(list_configs(bench, review, workers, cd, kind, slip, runs))
        want = sorted({n for cid, c, n, _ in jobs if c == cd and kind_of(cases, cid) == kind})
        miss = [n for n in want if n not in have]
        bad += len(miss)
        print(f"  {cd:10} {kind:12} {len(want) - len(miss)}/{len(want)} configurations exist" + (f"; MISSING {miss}" if miss else ""))
    return bad


def cmd_checkout(a):
    dest = ROOT / "checkouts" / f"slipstream-{a.name}"
    if dest.exists():
        sys.exit(f"{dest} exists; use another --name")
    src = pathlib.Path(a.source).resolve()
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", a.ref], capture_output=True, text=True, check=True).stdout.strip()
    dest.mkdir(parents=True)
    tar = subprocess.Popen(["git", "-C", str(src), "archive", commit], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=tar.stdout, check=True)
    (dest / "COMMIT").write_text(commit + "\n")
    print(f"{dest}  ({commit[:12]}, tracked files only)")


def cmd_run(a, execute: bool):
    bench = load_bench()
    slip_arg = a.slipstream or bench.get("paths", {}).get("slipstream", "")
    slip = pathlib.Path(slip_arg) if slip_arg else None
    runs = pathlib.Path(bench["docker"].get("runs") or ROOT / "results")
    jobs, cases, workers = jobs_for(a.review, bench, a.case, a.code, a.config, a.procs, a.seed, slip, runs)
    where = {cd: platform_of(bench, cd) for _, cd, _, _ in jobs}
    print(f"{len(jobs)} jobs, review {a.review}, codes {dict(sorted(where.items()))}, gpus={bench['docker'].get('gpus')}")
    if not execute:
        for cid, cd, n, k in jobs:
            print(f"  {cid:14} {cd:10} {n} p{k}")
        if a.check:
            sys.exit(check_names(bench, a.review, cases, workers, jobs, slip, runs))
        print("nothing started (add --execute to run; --check asks each worker for its configurations)")
        return
    if slip is None and "docker" in where.values():
        sys.exit("--slipstream CHECKOUT is required for the image (python -m slipstream_validation.runner checkout ...)")
    if slip is not None and not slip.is_dir():
        sys.exit(f"{slip} is not a directory")
    if slip is None:
        print("no --slipstream: the workers import the Slipstream installed in their environment (recorded in each result)")
    commit = ((slip / "COMMIT").read_text().strip() if (slip / "COMMIT").exists() else git_state(slip)["commit"]) if slip else None
    root = runs / a.tag
    root.mkdir(parents=True, exist_ok=True)
    repos = {cd: [git_state(p) for p in bench.get("codes", {}).get(cd, {}).get("repos", [])] for cd in where}
    env_doc = json.dumps(dict(
        runner=sys.version.split()[0], platform=sys.platform, cpu=cpu_name(), image=bench["docker"]["image"], image_id=image_id(bench),
        slipstream=dict(path=str(slip), commit=commit, dirty=git_state(slip)["dirty"] if slip and (slip / ".git").exists() else None),
        platforms=where, repos=repos, args=vars(a), time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1, default=str)
    (root / "environment.json").write_text(env_doc)
    for cid in {j[0] for j in jobs}:  # the per-case copy the study builder reads
        (root / cid).mkdir(exist_ok=True)
        if not (root / cid / "environment.json").exists():
            (root / cid / "environment.json").write_text(env_doc)
    for i, (cid, cd, name, k) in enumerate(jobs, 1):
        out = root / cid / cd / name / f"p{k}"
        done = out / "result.json"
        if done.exists() and json.loads(done.read_text()).get("status") == "ok" and not a.force:
            print(f"[{i}/{len(jobs)}] skip {cid}/{cd}/{name}/p{k}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        t = Target(bench, cd, slip, runs, a.review)
        (out / "job.json").write_text(json.dumps(dict(
            case_path=t.path(review_dir(a.review) / "cases" / f"{cid}.json"), config=name, repeats=a.repeats, threads=a.threads,
            out_dir=t.path(out)), indent=1))
        cmd, env = t.argv(worker_script(a.review, workers, cd, kind_of(cases, cid)), ["--job", t.path(out / "job.json")], a.threads)
        limit = a.timeout or bench.get("codes", {}).get(cd, {}).get("timeout", 3600)
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=limit, env=env)
            lines = [ln for ln in (r.stdout + r.stderr).splitlines() if not ln.startswith("wsl:")]
            rc, tail = r.returncode, "\n".join(lines)[-1500:]
        except subprocess.TimeoutExpired:
            rc, tail = -1, f"timeout after {limit} s"
        wall = time.perf_counter() - t0
        (out / "wall.json").write_text(json.dumps(dict(wall_s=wall, returncode=rc, stderr_tail=tail)))
        if not done.exists():
            done.write_text(json.dumps(dict(code=cd, case=cid, config=dict(name=name), rank=0, status="error", error=tail, warnings=[]), indent=1))
        res = json.loads(done.read_text())
        print(f"[{i}/{len(jobs)}] {cid}/{cd}/{name}/p{k}: {res.get('status')}, wall {wall:.1f} s", flush=True)
        if res.get("status") != "ok":  # the worker keeps its traceback in result.json, not on stderr
            text = (res.get("error") or tail).strip()
            print("   ", text.splitlines()[-1] if text else "(no output)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checkout", help="git archive export of the Slipstream commit under test")
    c.add_argument("source")
    c.add_argument("ref")
    c.add_argument("--name", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("review")
        p.add_argument("--case", action="append")
        p.add_argument("--code", action="append")
        p.add_argument("--config", action="append", help="name or glob; ``all``/``finest`` ask the worker (--list)")
        p.add_argument("--procs", type=int, default=1, help="independent process runs per configuration (the paper used 3)")
        p.add_argument("--repeats", type=int, default=2, help="warm repeats inside one process")
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--timeout", type=float, default=None, help="seconds per job; default: the code's `timeout` in bench.toml, else 3600")
        p.add_argument("--slipstream")
        p.add_argument("--tag")
        p.add_argument("--execute", action="store_true")
        p.add_argument("--check", action="store_true", help="plan only: ask each worker for its configurations and report unknown names")
        p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "checkout":
        cmd_checkout(a)
    else:
        if a.cmd == "run" and a.execute and not a.tag:
            sys.exit("--tag is required")
        cmd_run(a, execute=a.cmd == "run" and a.execute)


if __name__ == "__main__":
    main()

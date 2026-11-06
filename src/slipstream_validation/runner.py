"""Plan and run the cases of a review, one isolated worker process (one container) per job.

    python -m slipstream_validation.runner checkout ../slipstream HEAD --name head     # git archive export of the commit to test
    python -m slipstream_validation.runner plan benchmark --case ring_tunes            # prints the jobs, starts nothing
    python -m slipstream_validation.runner run benchmark --execute --tag pc1 --slipstream checkouts/slipstream-head

Results go to results/TAG/CASE/CODE/CONFIG/pK in the layout the analysis scripts of the review read.  Jobs run one at a time
(timing); an interrupted campaign resumes because a finished job (status ok) is skipped.  bench.toml says how a code is reached,
bench.local.toml (not committed) overrides it for one machine.
"""
import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import time
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")


def load_bench() -> dict:
    cfg = tomllib.loads((ROOT / "bench.toml").read_text())
    local = ROOT / "bench.local.toml"
    if local.exists():
        def merge(a, b):
            for k, v in b.items():
                a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
            return a
        cfg = merge(cfg, tomllib.loads(local.read_text()))
    return cfg


def review_dir(review: str) -> pathlib.Path:
    return ROOT / "reviews" / review


def jobs_for(review, bench, case=None, code=None, config=None, procs=1, seed=1):
    """(case, code, config, k) for every job; codes whose platform is archived or skip are left out."""
    d = review_dir(review)
    cases = tomllib.loads((d / "cases.toml").read_text())
    workers = tomllib.loads((d / "workers.toml").read_text())
    gpu = bool(bench.get("docker", {}).get("gpus"))
    jobs = []
    for cid, c in cases.items():
        if case and cid not in case:
            continue
        table = dict(c["image"])
        if gpu:
            for k, v in c.get("gpu", {}).items():
                table[k] = table.get(k, []) + v
        for cd, names in table.items():
            plat = bench.get("codes", {}).get(cd, {}).get("platform", bench.get("platform", "docker"))
            if plat in ("archived", "skip") or (code and cd not in code):
                continue
            if c["kind"] not in workers.get(cd, {}):
                raise ValueError(f"no {c['kind']} worker for {cd} in workers.toml")
            jobs += [(cid, cd, n, k) for n in names if not config or n in config for k in range(procs)]
    random.Random(seed).shuffle(jobs)
    return jobs, cases, workers


def host_path(p, bench) -> str:
    """The path as the Docker daemon sees it: /mnt/c/... when Docker is in WSL and the runner is on Windows."""
    p = pathlib.Path(p).resolve()
    if os.name == "nt" and bench["docker"].get("wsl_distro"):
        return f"/mnt/{p.drive[0].lower()}/" + "/".join(p.parts[1:])
    return str(p)


def docker_argv(bench, args: list, mounts: dict, env: dict, gpus: bool, workdir: str) -> list:
    d = bench["docker"]
    cmd = ["docker", "run", "--rm", "--entrypoint", "python"]
    if gpus:
        cmd += ["--gpus", "all"]
    user = d.get("user") or (f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") and not d.get("wsl_distro") else "")
    if user:
        cmd += ["--user", user, "-e", "HOME=/tmp"]
    for host, (inside, mode) in mounts.items():
        cmd += ["-v", f"{host_path(host, bench)}:{inside}:{mode}"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["-w", workdir, d["image"], "-u", *args]
    return ["wsl.exe", "-d", d["wsl_distro"], "--", *cmd] if d.get("wsl_distro") else cmd


def container_cmd(bench, slip, runs, worker_args, threads, code, review):
    env = {"PYTHONPATH": "/slipstream/src", "PYTHONDONTWRITEBYTECODE": "1", "SSV_RESULTS": "/runs",
           **{k: str(threads) for k in THREAD_VARS}, **{k: str(v) for k, v in bench.get("codes", {}).get(code, {}).get("env", {}).items()}}
    mounts = {ROOT: ("/work", "ro"), runs: ("/runs", "rw"), slip: ("/slipstream", "ro")}
    # the repository is read-only; Xsuite compiles its kernels into the working directory, so work in /tmp
    return docker_argv(bench, worker_args, mounts, env, bool(bench["docker"].get("gpus")), "/tmp")


def image_id(bench) -> str | None:
    d = bench["docker"]
    cmd = ["docker", "image", "inspect", "--format", "{{.Id}}", d["image"]]
    if d.get("wsl_distro"):
        cmd = ["wsl.exe", "-d", d["wsl_distro"], "--", *cmd]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip() or None


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
    jobs, cases, workers = jobs_for(a.review, bench, a.case, a.code, a.config, a.procs, a.seed)
    print(f"{len(jobs)} jobs, review {a.review}, image {bench['docker']['image']}, gpus={bench['docker'].get('gpus')}")
    if not execute:
        for cid, cd, n, k in jobs:
            print(f"  {cid:12} {cd:10} {n} p{k}")
        print("nothing started (add --execute to run)")
        return
    slip = pathlib.Path(a.slipstream or bench.get("paths", {}).get("slipstream", ""))
    if not slip.is_dir():
        sys.exit("--slipstream CHECKOUT is required (python -m slipstream_validation.runner checkout ...)")
    commit = (slip / "COMMIT").read_text().strip() if (slip / "COMMIT").exists() else None
    runs = pathlib.Path(bench["docker"].get("runs") or ROOT / "results")
    root = runs / a.tag
    root.mkdir(parents=True, exist_ok=True)
    (root / "environment.json").write_text(json.dumps(dict(
        runner=sys.version.split()[0], platform=sys.platform, image=bench["docker"]["image"], image_id=image_id(bench),
        slipstream=dict(path=str(slip), commit=commit), args=vars(a), time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        indent=1, default=str))
    for i, (cid, cd, name, k) in enumerate(jobs, 1):
        out = root / cid / cd / name / f"p{k}"
        done = out / "result.json"
        if done.exists() and json.loads(done.read_text()).get("status") == "ok" and not a.force:
            print(f"[{i}/{len(jobs)}] skip {cid}/{cd}/{name}/p{k}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        inside = f"/runs/{a.tag}/{cid}/{cd}/{name}/p{k}"
        (out / "job.json").write_text(json.dumps(dict(
            case_path=f"/work/reviews/{a.review}/cases/{cid}.json", config=name, repeats=a.repeats, threads=a.threads, out_dir=inside), indent=1))
        worker = workers[cd][cases[cid]["kind"]]
        cmd = container_cmd(bench, slip, runs, [f"/work/reviews/{a.review}/{worker}", "--job", f"{inside}/job.json"], a.threads, cd, a.review)
        limit = bench.get("codes", {}).get(cd, {}).get("timeout", 3600)
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
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
        p.add_argument("--config", action="append")
        p.add_argument("--procs", type=int, default=1, help="independent process runs per configuration (the paper used 3)")
        p.add_argument("--repeats", type=int, default=2, help="warm repeats inside one process")
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--slipstream")
        p.add_argument("--tag")
        p.add_argument("--execute", action="store_true")
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

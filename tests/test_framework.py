"""Tier 0: the cases, the workers, bench.toml and the archived evidence agree.  No simulation code is started."""
import re
import subprocess
import tomllib

import pytest

from slipstream_validation import check, runner

BENCH = runner.ROOT / "reviews" / "benchmark"
CASES = tomllib.loads((BENCH / "cases.toml").read_text())
WORKERS = tomllib.loads((BENCH / "workers.toml").read_text())


@pytest.mark.tier0
def test_every_case_has_a_file_a_worker_and_an_analysis():
    for cid, c in CASES.items():
        assert (BENCH / "cases" / f"{cid}.json").is_file(), cid
        assert cid in check.ANALYSIS and (BENCH / check.ANALYSIS[cid]).is_file(), cid
        for code in {**c["cpu"], **c.get("gpu", {})}:
            assert (BENCH / WORKERS[code][c["kind"]]).is_file(), (cid, code)


@pytest.mark.tier0
def test_no_machine_path_is_committed():
    for p in [runner.ROOT / "bench.toml", *BENCH.rglob("*.py"), *BENCH.rglob("*.toml")]:
        text = p.read_text(errors="ignore")
        assert "C:/Users" not in text and "C:" + chr(92) + "Users" not in text, p


@pytest.mark.tier0
def test_git_ignores_no_evidence():
    """An ignored evidence file exists on this PC and nowhere else: `runs/` once hid one the claims read, and CI failed on every push."""
    if not (runner.ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    r = subprocess.run(["git", "-C", str(runner.ROOT), "ls-files", "--others", "--ignored", "--exclude-standard", "--", "reviews/claims/evidence"],
                       capture_output=True, text=True, check=True)
    assert not r.stdout, r.stdout


@pytest.mark.tier0
def test_plan_lists_jobs_for_the_image_codes_only():
    bench = tomllib.loads((runner.ROOT / "bench.toml").read_text())
    jobs, _, _ = runner.jobs_for("benchmark", bench, case=["ring_tunes"])
    codes = {j[1] for j in jobs}
    assert codes == {"slipstream", "elegant", "xsuite", "cheetah", "scibmad", "bmadx"}   # GPT, JuTrack and ImpactX are archived on a plain Linux host
    assert not any(j[2].startswith("cuda") for j in jobs)             # no GPU unless bench.local.toml asks
    gpu = dict(bench, docker=dict(bench["docker"], gpus=True))
    assert any(j[2].startswith("cuda") for j in runner.jobs_for("benchmark", gpu, case=["ring_tunes"])[0])


@pytest.mark.tier0
def test_the_base_image_is_pinned_and_the_nightly_plans_only_codes_of_the_core_image(monkeypatch):
    docker = (runner.ROOT / "docker" / "Dockerfile").read_text()
    assert re.search(r"^ARG SIMCODES_IMAGE=\S+@sha256:[0-9a-f]{64}$", docker, re.M)
    assert " AS core" in docker
    monkeypatch.setenv("SSV_BENCH", str(runner.ROOT / "bench.ci.toml"))
    codes = {j[1] for j in runner.jobs_for("benchmark", runner.load_bench())[0]}
    assert codes == {"slipstream", "elegant", "astra", "xsuite", "cheetah", "ocelot"}


@pytest.mark.tier0
def test_every_image_job_has_an_archived_row_to_compare_with():
    assert set(check.CASES) == set(CASES)
    for cid, c in CASES.items():
        old = check.archived(cid)
        for code, names in c["cpu"].items():
            for n in names:
                found = (code, n) in old or check.ALIASES.get((code, n)) in old
                assert found or (cid, code, n) in check.NOREF_OK, (cid, code, n)


GOOD = dict(code="slipstream", config="cpu-f64-o6k8b16", status="ok", dq_ref="1.6e-9")


@pytest.mark.tier0
def test_the_gate_can_fail():
    old = dict(dq_ref="1.6e-9")
    judge = lambda x, o=old, case="ring_tunes", code="slipstream", cfg="c", lim={}: check.judge(case, code, cfg, x, o, lim)[0]  # noqa: E731
    assert judge(GOOD) == "ok"
    assert judge(dict(GOOD, dq_ref="1.7e-9")) == "ok"                    # inside the floor
    assert judge(dict(GOOD, dq_ref="5e-6")) == "MOVED"                   # a 3000-fold change in the tune error
    assert judge(dict(GOOD, dq_ref="")) == "FAILED"                      # job produced no number
    assert judge(dict(GOOD, status="error")) == "FAILED"
    assert judge(GOOD, o=None) == "NOREF"
    # ratchet: 30 % of a value above the floor moves, 20 % does not
    big = dict(ad_vs_fd_fro="1e-6", ad_vs_fd_col="1e-6", rel_fro="1e-3", rel_col="1e-3", dy0="1e-3")
    assert judge(dict(big, rel_fro="1.2e-3"), big, "ring_grad") == "ok"
    assert judge(dict(big, rel_fro="1.3e-3"), big, "ring_grad") == "MOVED"
    # a metric with its own floor (finite-difference round-off between hosts): only that one
    small = dict(big, ad_vs_fd_fro="3e-8", ad_vs_fd_col="3e-8")
    assert judge(dict(small, ad_vs_fd_col="1.2e-7"), small, "ring_grad") == "ok"
    assert judge(dict(small, ad_vs_fd_col="5e-7"), small, "ring_grad") == "MOVED"
    assert judge(dict(small, ad_vs_fd_fro="1.2e-7"), small, "ring_grad") == "MOVED"
    # absolute limit of the case beats the archive: the archive itself may not break G11
    assert judge(dict(big, ad_vs_fd_fro="2e-5"), dict(big, ad_vs_fd_fro="2e-5"), "ring_grad", lim={"jac_rel": 1e-5}) == "LIMIT"
    # a flag that changed
    sc = dict(size_vs_ref="0.01", length_vs_ref="0.01", emit_vs_ref="0.01", spread_vs_ref="0.01", dEkin_change_eV="50", **{"pass": "True"})
    assert judge(dict(sc, **{"pass": "False"}), sc, "sc_drift") == "MOVED"
    # a known difference is XFAIL and says why; the same move on another row is not
    key = ("ring_grad", "slipstream", "cpu-f64-o6q8s1b8")
    moved = dict(big, rel_col="1e-6")
    assert check.judge(*key, moved, big, {})[0] == "XFAIL"
    assert check.judge(*key, dict(moved, rel_fro="1e-6"), big, {})[0] == "MOVED"        # one unexplained metric spoils it
    assert check.judge("ring_grad", "xsuite", "cpu-f64-yosk8b16", moved, big, {})[0] == "MOVED"


def _bench(**codes):
    return dict(docker=dict(image="img:1", wsl_distro="", gpus=False), wsl=dict(distro="Ubuntu"), conda=dict(base="~/mf"), codes=codes)


@pytest.mark.tier0
def test_every_platform_builds_a_command_for_the_same_job(tmp_path):
    import os
    script = runner.ROOT / "reviews" / "benchmark" / "workers" / "w_elegant.py"
    runs, slip = runner.ROOT / "results", tmp_path
    inside = runs / "t" / "ring_tunes" / "elegant" / "c" / "p0" / "job.json"
    # docker: the repository, the results folder and the Slipstream checkout are mounted at /work, /runs, /slipstream; results sit inside the repository
    t = runner.Target(_bench(a=dict(platform="docker", python="/opt/envs/x/bin/python")), "a", slip, runs)
    cmd, _ = t.argv(script, ["--job", t.path(inside)], 2)
    assert cmd[:2] == ["docker", "run"] and "/opt/envs/x/bin/python" in cmd
    assert t.path(inside) == "/runs/t/ring_tunes/elegant/c/p0/job.json" and "/work/reviews/benchmark/workers/w_elegant.py" in cmd
    assert "SSV_RESULTS=/runs" in cmd and "OMP_NUM_THREADS=2" in cmd and "PYTHONPATH=/slipstream/src" in cmd
    # conda: a bash launcher in the environment, variables as KEY=VALUE before `--`
    t = runner.Target(_bench(a=dict(platform="conda", conda_env="ssv-x", env=dict(K="v"))), "a", slip, runs)
    cmd, _ = t.argv(script, ["--job", "j"], 1)
    assert cmd[0] == "bash" and cmd[2:4] == ["ssv-x", "1"] and "K=v" in cmd and cmd[cmd.index("--") + 1] == t.path(script)
    # windows / linux: the interpreter and the worker, variables in the process environment
    for plat in ("windows", "linux"):
        t = runner.Target(_bench(a=dict(platform=plat, python="py", env=dict(K="v"))), "a", slip, runs)
        cmd, env = t.argv(script, ["--list"], 1)
        assert cmd[0] == "py" and cmd[-1] == "--list" and env["K"] == "v" and env["SSV_RESULTS"] == str(runs.resolve())
    if os.name == "nt":   # a WSL path of a Windows path
        t = runner.Target(_bench(a=dict(platform="wsl", conda_env="e")), "a", slip, runs)
        assert t.path(script).startswith("/mnt/") and "\\" not in t.path(script)
        assert t.argv(script, [], 1)[0][:4] == ["wsl.exe", "-d", "Ubuntu", "--"]


@pytest.mark.tier0
def test_a_case_that_is_not_in_cases_toml_still_has_a_kind():
    cases = tomllib.loads((BENCH / "cases.toml").read_text())
    assert runner.kind_of(cases, "ring_scale_n500") == "ring" and runner.kind_of(cases, "sc_drift_n2000s1") == "sc_drift"
    assert runner.kind_of(cases, "ring_grad_b") == "ring" and runner.kind_of(cases, "linear_fodo") == "linear_fodo"
    with pytest.raises(ValueError):
        runner.kind_of(cases, "nothing")
    for case in (p.stem for p in (BENCH / "cases").glob("*.json") if "lat" not in p.stem and p.stem != "mls_ring"):
        runner.kind_of(cases, case)   # every frozen case file has a worker kind

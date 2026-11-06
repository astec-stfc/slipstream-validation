"""Tier 0: the cases, the workers, bench.toml and the archived evidence agree.  No simulation code is started."""
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
        for code in {**c["image"], **c.get("gpu", {})}:
            assert (BENCH / WORKERS[code][c["kind"]]).is_file(), (cid, code)


@pytest.mark.tier0
def test_no_machine_path_is_committed():
    for p in [runner.ROOT / "bench.toml", *BENCH.rglob("*.py"), *BENCH.rglob("*.toml")]:
        text = p.read_text(errors="ignore")
        assert "C:/Users" not in text and "C:" + chr(92) + "Users" not in text, p


@pytest.mark.tier0
def test_plan_lists_jobs_for_the_image_codes_only():
    bench = tomllib.loads((runner.ROOT / "bench.toml").read_text())
    jobs, _, _ = runner.jobs_for("benchmark", bench, case=["ring_tunes"])
    codes = {j[1] for j in jobs}
    assert codes == {"slipstream", "elegant", "xsuite", "cheetah"}   # JuTrack and Bmad-X are archived
    assert not any(j[2].startswith("cuda") for j in jobs)             # no GPU unless bench.local.toml asks
    gpu = dict(bench, docker=dict(bench["docker"], gpus=True))
    assert any(j[2].startswith("cuda") for j in runner.jobs_for("benchmark", gpu, case=["ring_tunes"])[0])


@pytest.mark.tier0
def test_every_image_job_has_an_archived_row_to_compare_with():
    assert set(check.CASES) == set(CASES)
    for cid, c in CASES.items():
        old = check.archived(cid)
        for code, names in c["image"].items():
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
    # absolute limit of the case beats the archive: the archive itself may not break G11
    assert judge(dict(big, ad_vs_fd_fro="2e-5"), dict(big, ad_vs_fd_fro="2e-5"), "ring_grad", lim={"jac_rel": 1e-5}) == "LIMIT"
    # a flag that changed
    sc = dict(size_vs_ref="0.01", length_vs_ref="0.01", emit_vs_ref="0.01", spread_vs_ref="0.01", dEkin_change_eV="50", **{"pass": "True"})
    assert judge(dict(sc, **{"pass": "False"}), sc, "sc_drift") == "MOVED"
    # a known difference is XFAIL and says why; the same move on another row is not
    key = ("ring_grad", "slipstream", "cpu-f64-o6k8b16")
    moved = dict(big, rel_col="1e-6")
    assert check.judge(*key, moved, big, {})[0] == "XFAIL"
    assert check.judge(*key, dict(moved, rel_fro="1e-6"), big, {})[0] == "MOVED"        # one unexplained metric spoils it
    assert check.judge("ring_grad", "xsuite", "cpu-f64-yosk8b16", moved, big, {})[0] == "MOVED"

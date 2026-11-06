"""Tier 0: the cases, the workers, bench.toml and the archived evidence agree.  No simulation code is started."""
import pathlib
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
def test_every_ring_tunes_job_has_an_archived_row_to_compare_with():
    old = check.archived_ring_tunes()
    for code, names in CASES["ring_tunes"]["image"].items():
        for n in names:
            assert check.ALIASES.get((code, n), (code, n)) in old, (code, n)

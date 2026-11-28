"""The gate that needs Slipstream and no other code: the frozen `linear_fodo` case against its analytic reference and the archive,
through the same runner and `check` as a campaign."""
import pathlib

import pytest

from slipstream_validation import check, runner

slipstream = pytest.importorskip("slipstream")
pytest.importorskip("torch")


@pytest.mark.slipstream
def test_linear_fodo_runs_and_agrees_with_the_archive(tmp_path, monkeypatch):
    bench = tmp_path / "bench.toml"
    bench.write_text(f'[docker]\nruns = "{tmp_path.as_posix()}"\n[codes.slipstream]\nplatform = "linux"\n')   # linux: this Python
    monkeypatch.setenv("SSV_BENCH", str(bench))
    root = pathlib.Path(slipstream.__file__).resolve().parents[2]
    source = ["--slipstream", str(root)] if (root / "src" / "slipstream").is_dir() else []   # an installed package has no checkout
    runner.main(["run", "benchmark", "--execute", "--tag", "gate", "--case", "linear_fodo", "--code", "slipstream", *source])
    assert len(check.ran(tmp_path, "gate", "linear_fodo")) == 2   # a campaign with no rows would compare as agreeing
    assert check.compare("gate", "linear_fodo", tmp_path) == 0

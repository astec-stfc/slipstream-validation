"""Tier 0: every number quoted in the paper still follows from the archived evidence."""
import csv
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.claims
@pytest.mark.tier0
def test_every_quoted_number_follows_from_the_evidence():
    r = subprocess.run([sys.executable, str(HERE / "verify_claims.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1000:]
    rows = list(csv.DictReader((HERE.parents[1] / "results" / "claims.csv").open()))
    assert len(rows) >= 400, f"only {len(rows)} checks ran"
    assert all(x["status"] == "ok" for x in rows)

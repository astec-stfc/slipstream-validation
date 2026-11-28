"""Tests marked `slipstream` import the Slipstream under test: the checkout named by SSV_SLIPSTREAM, else the installed package.
The report header says which one, with its commit."""
import importlib.util
import os
import pathlib
import subprocess
import sys

if os.environ.get("SSV_SLIPSTREAM"):
    sys.path.insert(0, str(pathlib.Path(os.environ["SSV_SLIPSTREAM"]).resolve() / "src"))


def pytest_report_header(config):
    spec = importlib.util.find_spec("slipstream")   # finds it without importing torch
    if spec is None:
        return "slipstream: not found (tests marked slipstream skip)"
    folder = pathlib.Path(spec.origin).parent
    commit = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return f"slipstream: {folder} {commit[:12] or '(not a git checkout)'}"

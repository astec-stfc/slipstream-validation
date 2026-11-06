"""JuTrack.jl adapter.  JuTrack is Julia, so one persistent Julia process (jutrack_server.jl) is driven over stdin/stdout:
``import_s`` is the package load, ``cold_s`` includes Julia's JIT, ``warm`` is a repeat call in the same session.
JuTrack is an Accelerator-Toolbox-style code: KQUAD/KSEXT use the 4th-order symplectic multipole pass (NumIntSteps slices),
SBEND the AT bend pass with the hard-edge fringes switched off.  Coordinates (x, px, y, py, z, delta) have px = Px/P0, as in BSF.
Its ``energy`` argument is kinetic, so E0 - m is passed."""
import atexit
import json
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from common import run_worker  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
JULIA = os.environ.get("BENCH_JULIA", "julia")
PROJECT = os.environ.get("BENCH_JUTRACK", "")

# (NumIntSteps per quad/sext, NumIntSteps per bend)
_RUNGS = [(4, 8), (8, 16), (16, 32), (32, 64)]
CONFIGS = [dict(name=f"cpu-f64-n{k}b{b}", device="cpu", dtype="float64", n_steps=k, n_steps_bend=b) for k, b in _RUNGS]
_S = {}


def _reply(proc):
    for ln in proc.stdout:
        if ln.startswith("@@"):
            if ln.startswith("@@ERR"):
                raise RuntimeError("JuTrack: " + ln[5:] + "".join(proc.stdout.readline() for _ in range(15)))
            return ln[2:].strip()
    raise RuntimeError("julia exited")


def _ask(cmd):
    p = _S["proc"]
    p.stdin.write(cmd + "\n")
    p.stdin.flush()
    return _reply(p)


def setup():
    env = dict(os.environ, JULIA_NUM_THREADS="1")
    p = subprocess.Popen([JULIA, f"--project={PROJECT}", "-O2", str(HERE / "jutrack_server.jl")], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    _S["proc"] = p
    atexit.register(lambda: (p.stdin.close(), p.wait(timeout=30)) if p.poll() is None else None)
    _S["ready"] = _reply(p)  # READY import_s=... julia=... threads=...


def build(lat, case, cfg):
    d = pathlib.Path(tempfile.mkdtemp(prefix="bench_jutrack_"))
    (d / "lat.json").write_text(json.dumps(lat))
    r = _ask(f"BUILD {(d / 'lat.json').as_posix()} {cfg['n_steps']} {cfg['n_steps_bend']} {case['energy_eV']!r} {case['mass_eV']!r}")
    return dict(dir=d), json.loads(r[3:])


def track(model, init, turns, cfg):
    d = model["dir"]
    np.ascontiguousarray(init, dtype="<f8").tofile(d / "init.bin")
    r = _ask(f"TRACK {(d / 'init.bin').as_posix()} {(d / 'out.bin').as_posix()} {turns}")
    if "lost=0" not in r:
        raise RuntimeError(f"JuTrack lost particles: {r}")
    return np.fromfile(d / "out.bin", dtype="<f8").reshape(len(init), turns, 4)


def versions():
    return dict(jutrack_ready=_S.get("ready"), project=PROJECT,
                jutrack_commit=subprocess.run(["git", "-C", PROJECT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    run_worker("jutrack", CONFIGS, build, track, versions=versions, setup=setup)

"""SciBmad adapter (SciBmad 0.6.0: BeamTracking.jl, Beamlines.jl, GTPSA.jl; the conda distribution ``scibmad`` bundles Julia 1.12.7).
SciBmad is Julia, so one persistent Julia process (scibmad_server.jl) is driven over stdin/stdout, as for JuTrack: ``import_s`` is the
package load, ``cold_s`` includes Julia's JIT, ``warm`` is a repeat call in the same session.
Coordinates are Bmad's (x, px, y, py, z, pz) with px = Px/P0 (canonical) and pz = delta, i.e. already the benchmark standard frame.
Rungs: ``std`` is the shipped default (SciBmadStandard); ``sym*`` uses BeamTracking.Symplectic(order, n_steps) on quadrupoles and sextupoles
with the hard-edge fringes off and an exact sector bend, the same element models as the other codes' nonlinear settings.
Derivatives: the Jacobian of the tracked state with respect to the family strengths uses ForwardDiff through ``track`` (SciBmad accepts
ForwardDiff.Dual and GTPSA.TPS element parameters)."""
import atexit
import json
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from common import p0c_eV, run_worker  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
JULIA = os.environ.get("BENCH_SCIBMAD_JULIA", "julia")
GPU_PROJECT = HERE.parent / "codes" / "scibmad_gpu"   # Julia project holding CUDA.jl (the conda distribution has none)

# (symplectic order, steps per quadrupole/sextupole)
_RUNGS = [(4, 4), (4, 16), (6, 8), (6, 32)]
CONFIGS = [dict(name="cpu-f64-std", device="cpu", dtype="float64", method="std", order=4, n_steps=1)]
CONFIGS += [dict(name=f"cpu-f64-o{o}k{k}", device="cpu", dtype="float64", method="sym", order=o, n_steps=k) for o, k in _RUNGS]
CONFIGS += [dict(name=f"cpu-f64-{m}-o4k4", device="cpu", dtype="float64", method=m, order=4, n_steps=4) for m in ("symy", "dk", "dky")]   # which algorithmic choice matters
CONFIGS += [dict(name=f"cuda-f{t[-2:]}-o{o}k{k}", device="cuda", dtype=t, method="sym", order=o, n_steps=k) for t in ("float32", "float64") for o, k in ((6, 8), (4, 4))]
_S = {}


def _reply(proc):
    for ln in proc.stdout:
        if ln.startswith("@@"):
            if ln.startswith("@@ERR"):
                raise RuntimeError("SciBmad: " + ln[5:] + "".join(proc.stdout.readline() for _ in range(15)))
            return ln[2:].strip()
    raise RuntimeError("julia exited")


def _ask(cmd):
    p = _S["proc"]
    p.stdin.write(cmd + "\n")
    p.stdin.flush()
    return _reply(p)


def _wants_cuda():
    """setup() takes no configuration, so read the job file's config name: CUDA configs start Julia in the GPU project."""
    argv = sys.argv
    if "--job" in argv:
        job = json.loads(pathlib.Path(argv[argv.index("--job") + 1]).read_text())
        return str(job.get("config", "")).startswith("cuda")
    return False


def setup():
    env = dict(os.environ, JULIA_NUM_THREADS="1")
    cmd = [JULIA]
    if _wants_cuda():
        env.update(BENCH_SCIBMAD_CUDA="1", JULIA_CUDA_HARD_MEMORY_LIMIT="4GiB")   # the 8 GiB card also runs the desktop; never spill to host RAM
        cmd.append(f"--project={GPU_PROJECT}")
    p = subprocess.Popen(cmd + [str(HERE / "scibmad_server.jl")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=env)
    _S["proc"] = p
    atexit.register(lambda: (p.stdin.close(), p.wait(timeout=30)) if p.poll() is None else None)
    _S["ready"] = _reply(p)  # READY import_s=... julia=... scibmad=... beamtracking=... beamlines=... gtpsa=...


def build(lat, case, cfg):
    d = pathlib.Path(tempfile.mkdtemp(prefix="bench_scibmad_"))
    (d / "lat.json").write_text(json.dumps(lat))
    r = _ask(f"BUILD {(d / 'lat.json').as_posix()} {cfg['method']} {cfg['order']} {cfg['n_steps']} {case['energy_eV']!r} {case['mass_eV']!r} {cfg['device']} {cfg['dtype']}")
    return dict(dir=d), json.loads(r[3:])


def track(model, init, turns, cfg):
    d = model["dir"]
    np.ascontiguousarray(init, dtype="<f8").tofile(d / "init.bin")
    r = _ask(f"TRACK {(d / 'init.bin').as_posix()} {(d / 'out.bin').as_posix()} {turns}")
    if "lost=0" not in r:
        raise RuntimeError(f"SciBmad lost particles: {r}")
    return np.fromfile(d / "out.bin", dtype="<f8").reshape(len(init), turns, 4)


def ad_build(lat, case, cfg, knobs):
    d = pathlib.Path(tempfile.mkdtemp(prefix="bench_scibmad_ad_"))
    (d / "lat.json").write_text(json.dumps(lat))
    (d / "knobs.json").write_text(json.dumps(knobs))
    _ask(f"ADBUILD {(d / 'lat.json').as_posix()} {(d / 'knobs.json').as_posix()} {cfg['method']} {cfg['order']} {cfg['n_steps']} "
         f"{case['energy_eV']!r} {case['mass_eV']!r}")
    return dict(dir=d)


def ad_jacobian(model, init, turns, cfg, knobs):
    d = model["dir"]
    np.ascontiguousarray(init, dtype="<f8").tofile(d / "init.bin")
    _ask(f"ADJAC {(d / 'init.bin').as_posix()} {(d / 'out.bin').as_posix()} {turns}")
    a = np.fromfile(d / "out.bin", dtype="<f8")
    K = len(knobs)
    return a[:4], a[4:].reshape(K, 4).T


def versions():
    return dict(scibmad_ready=_S.get("ready"), julia_exe=JULIA)


if __name__ == "__main__":
    run_worker("scibmad", CONFIGS, build, track, versions=versions, setup=setup, ad=(ad_build, ad_jacobian))

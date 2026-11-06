"""ASTRA adapter for ``sc_drift``.  Writes the deck and the frozen macroparticle file, runs ``Astra.exe`` in a scratch directory, reads the
``.Xemit/.Yemit/.Zemit`` tables (fixed-time snapshots, normalised rms emittance) and interpolates them to the case stations.
Timing covers the whole subprocess plus parsing.  Configs: axisymmetric (Nrad, Nlong) meshes at several time steps, and the 3-D solver."""
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from astra_deck import write_deck  # noqa: E402
from common import launches, run_worker  # noqa: E402

EXE = pathlib.Path(os.environ.get("BENCH_ASTRA_EXE", "Astra"))
# (Nrad, Nlong_in, time step [ps]); the 3-D rung is last so that the code's own finest setting is the 3-D solver
_RUNGS = [(8, 16, 7.0), (16, 32, 3.5), (32, 64, 1.4), (64, 128, 0.7)]
CONFIGS = [dict(name=f"cpu-f64-n{n}l{nlong}h{h:g}", device="cpu", dtype="float64", nrad=n, nlong=nlong, h_ns=h * 1e-3, three_d=False) for n, nlong, h in _RUNGS]
CONFIGS += [dict(name=f"cpu-f64-3d{n}h{h:g}", device="cpu", dtype="float64", nrad=0, nlong=0,
                 h_ns=h * 1e-3, three_d=True, n3d=n) for n, h in ((64, 1.4), (32, 0.7))]
# Convergence extension: finer 3-D meshes at the original 1.4 ps step.
CONFIGS += [dict(name=f"cpu-f64-3d{n}h1.4", device="cpu", dtype="float64", nrad=0, nlong=0, h_ns=1.4e-3, three_d=True, n3d=n) for n in (96, 128)]
CONFIGS.append(dict(name="cpu-f64-3d32h1.4", device="cpu", dtype="float64", nrad=0, nlong=0, h_ns=1.4e-3, three_d=True, n3d=32))


def setup():
    pass


def build(lat, case, cfg):
    d = pathlib.Path(tempfile.mkdtemp(prefix="bench_astra_"))
    deck = write_deck(d, launches(case), case, cfg["nrad"], cfg["nlong"], cfg["h_ns"], cfg["three_d"], cfg.get("n3d", 32))
    return dict(dir=d, deck=deck, case=case), None


def read_table(path):
    a = np.loadtxt(path)
    return a


def _write_energy_gain(d):
    """Particle-based mean kinetic-energy gain from the full-precision phase-space file at ZSTOP, written as ``energy_gain.json`` beside the result
    (the ``Zemit`` table prints the mean energy in 100 eV steps).  Uses ASTRA's electron mass for both files, so the gain is mass-independent."""
    import json
    argv = sys.argv
    final = sorted(d.glob("sc_drift.[0-9][0-9][0-9][0-9].001"))
    if "--job" not in argv or not final:
        return

    def ekin(t):
        ref, p = t[0], t[1:]
        pt = np.sqrt(p[:, 3] ** 2 + p[:, 4] ** 2 + (p[:, 5] + ref[5]) ** 2)
        return float((np.sqrt(pt ** 2 + 511003.0 ** 2) - 511003.0).mean())

    out = pathlib.Path(json.loads(pathlib.Path(argv[argv.index("--job") + 1]).read_text())["out_dir"])
    e0, e1 = ekin(np.loadtxt(d / "beam.ini")), ekin(np.loadtxt(final[-1]))
    (out / "energy_gain.json").write_text(json.dumps(dict(ekin_initial_eV=e0, ekin_final_eV=e1, gain_eV=e1 - e0, file=final[-1].name)))


def track(model, init, turns, cfg):
    d, case = model["dir"], model["case"]
    for f in d.glob("sc_drift.*.001"):
        f.unlink()
    r = subprocess.run([str(EXE), model["deck"].name], cwd=d, capture_output=True, text=True, timeout=3600, input="\n")
    if not (d / "sc_drift.Xemit.001").exists():
        raise RuntimeError((r.stdout + r.stderr)[-2000:])
    _write_energy_gain(d)
    X, Y, Z = (read_table(d / f"sc_drift.{c}emit.001") for c in "XYZ")
    st = np.array(case["stations_m"])
    # ASTRA columns: X/Y: z[m] t[ns] <x>[mm] rms x[mm] rms x'[mrad] emit_n[pi mm mrad] cor;  Z: z t Ekin[MeV] rms z[mm] rms dE[keV] emit[keV mm] cor
    f = lambda tab, col: np.interp(st, tab[:, 0], tab[:, col])  # noqa: E731
    return np.column_stack([st, f(X, 3) * 1e-3, f(Y, 3) * 1e-3, f(Z, 3) * 1e-3, f(X, 5) * 1e-6, f(Y, 5) * 1e-6, f(Z, 2) * 1e6, f(Z, 4) * 1e3])


def versions():
    r = subprocess.run([str(EXE)], capture_output=True, text=True, input="\n", timeout=30)
    ban = [ln.strip() for ln in r.stdout.splitlines() if "Version" in ln or "DESY" in ln]
    return dict(astra_exe=str(EXE), banner=ban)


if __name__ == "__main__":
    run_worker("astra", CONFIGS, build, track, versions=versions, setup=setup)

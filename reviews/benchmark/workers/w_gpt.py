"""GPT adapter for ``sc_drift``.  The macroparticles come from the ASTRA-format file written for ASTRA (converted with ``astra2gdf``) and are
loaded with ``setfile``; ``settdist(...,0,0)`` makes them a fixed-time snapshot, as GPT's manual prescribes for ASTRA files.  ``tout`` snapshots every
0.05 m of reference travel; ``gdfa`` reduces them to the bunch moments and the stations are interpolated on the mean z.  Licence: GPT reads it from
the ``GPTLICENSE`` environment variable (a user-level variable on this host); it is passed on the command line and never written to a result file.
Space charge: the mesh-based ``spacecharge3Dmesh`` at several ``accuracy`` levels, and ``spacecharge3D``, GPT's point-particle tree code (opening angle
theta, default; theta = 0 would be all N^2 pairs), which needs a Plummer radius from ``setrmacrodist`` (1e-12 m here, far below the particle spacing).
The local notes found the mesh solver 12 % below the point-particle model on a very short bunch, so the tree code is the second reference."""
import math
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

BIN = pathlib.Path(os.environ.get("BENCH_GPT_BIN", ""))
CONFIGS = [dict(name=f"cpu-f64-mesh-acc{a}", device="cpu", dtype="float64", solver="spacecharge3Dmesh", accuracy=a) for a in (4, 5, 6)]
CONFIGS.append(dict(name="cpu-f64-tree-acc6", device="cpu", dtype="float64", solver="spacecharge3D", accuracy=6))


def setup():
    pass


def _run(args, cwd, **kw):
    return subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True, timeout=7200, **kw)


def build(lat, case, cfg):
    d = pathlib.Path(tempfile.mkdtemp(prefix="bench_gpt_"))
    write_deck(d, launches(case), case)  # only beam.ini is used
    r = _run([BIN / "astra2gdf.exe", "-o", "beam.gdf", "beam.ini"], d)
    if not (d / "beam.gdf").exists():
        raise RuntimeError(r.stdout + r.stderr)
    mc2, ek = case["mass_eV"], case["kinetic_eV"]
    g0 = 1 + ek / mc2
    beta0 = math.sqrt(1 - 1 / g0**2)
    c = 299792458.0
    plummer = 'setrmacrodist("beam","d",1e-12);\n' if cfg["solver"] == "spacecharge3D" else ""  # the tree code needs a positive Plummer radius
    (d / "sc.in").write_text(f'setfile("beam","beam.gdf");\nsettdist("beam","u",0,0);\n{plummer}accuracy({cfg["accuracy"]});\n{cfg["solver"]}();\n'
                             f'tout(0,{(case["stations_m"][-1] + 0.05) / (beta0 * c)!r},{0.05 / (beta0 * c)!r});\n')
    return dict(dir=d, case=case), None


def track(model, init, turns, cfg):
    d, case = model["dir"], model["case"]
    lic = os.environ.get("GPTLICENSE")
    r = _run([BIN / "gpt.exe", "-j", "1", "-o", "out.gdf", "sc.in"] + ([f"GPTLICENSE={lic}"] if lic else []), d)
    if not (d / "out.gdf").exists() or r.returncode:
        raise RuntimeError((r.stdout + r.stderr).replace(lic or "\0", "<key>")[-2000:])
    cols = "time avgz stdx stdy stdz nemixrms nemiyrms avgG stdG".split()
    _run([BIN / "gdfa.exe", "-o", "a.gdf", "out.gdf", *cols], d)
    t = _run([BIN / "gdf2a.exe", "a.gdf"], d).stdout.splitlines()
    names = t[0].split()
    a = np.array([[float(v) for v in ln.split()] for ln in t[1:] if ln.strip()])
    col = lambda k: a[:, names.index(k)]  # noqa: E731
    mc2, st = case["mass_eV"], np.array(case["stations_m"])
    f = lambda k, s=1.0: np.interp(st, col("avgz"), col(k)) * s  # noqa: E731
    (d / "out.gdf").unlink()  # 185 MB of snapshots; the reduced table is what the benchmark keeps
    return np.column_stack([st, f("stdx"), f("stdy"), f("stdz"), f("nemixrms"), f("nemiyrms"), (f("avgG") - 1) * mc2, f("stdG", mc2)])


def versions():
    r = _run([BIN / "gpt.exe", "-v"], BIN)
    return dict(gpt=(r.stdout + r.stderr).splitlines()[0] if (r.stdout + r.stderr).strip() else None, bin=str(BIN))


if __name__ == "__main__":
    run_worker("gpt", CONFIGS, build, track, versions=versions, setup=setup)

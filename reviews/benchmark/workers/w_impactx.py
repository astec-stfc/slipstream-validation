"""ImpactX adapter for ``sc_drift`` (WSL conda env impactx-cpu-mpich-dev, CPU/OpenMP Python module).  Each call runs ``impactx_run.py`` in a
fresh process (AMReX initialises once per process) and reads ``diags/reduced_beam_characteristics.0.0``, ImpactX's own moments at every slice
end; the rows at the case stations are picked.  Conversions: sigma_z = beta_ref * sigma_t (ImpactX's t is c*time at fixed s); kinetic energy
and spread from pt = -gamma in units of the reference beta*gamma.  This build has no FFT support, so the 3-D Poisson solver is multigrid."""
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
# (cells x, cells z, slices per 0.05 m drift)
_RUNGS = [(16, 20, 1), (32, 40, 2), (64, 80, 4)]
CONFIGS = [dict(name=f"cpu-f64-mg{n}x{nz}s{s}", device="cpu", dtype="float64", n=n, nz=nz, nslice=s, space_charge="3D", solver="multigrid")
           for n, nz, s in _RUNGS]
CONFIGS.append(dict(name="cpu-f64-gauss3d-s4", device="cpu", dtype="float64", n=16, nz=20, nslice=4, space_charge="Gauss3D", solver=None))


def setup():
    import impactx  # noqa: F401


def build(lat, case, cfg):
    d = pathlib.Path(tempfile.mkdtemp(prefix="bench_impactx_"))
    n_steps = int(round(case["stations_m"][-1] / 0.05))
    run = dict(dir=str(d), npz=str(HERE.parent / "cases" / case["particles"]["file"]), charge_C=case["charge_C"], kin_eV=case["kinetic_eV"],
               ds=0.05, n_steps=n_steps, nslice=cfg["nslice"], n_cell=[cfg["n"], cfg["n"], cfg["nz"]], space_charge=cfg["space_charge"],
               solver=cfg["solver"], mass_eV=case["mass_eV"], max_level=1, prob_relative=[3.0, 1.1])
    (d / "run.json").write_text(json.dumps(run))
    return dict(dir=d, case=case), None


def track(model, init, turns, cfg):
    d, case = model["dir"], model["case"]
    for f in (d / "diags").glob("*") if (d / "diags").exists() else []:
        f.unlink()
    r = subprocess.run([sys.executable, "-u", str(HERE / "impactx_run.py"), str(d / "run.json")], capture_output=True, text=True, timeout=7200,
                       env=dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1")))
    f = d / "diags" / "reduced_beam_characteristics.0.0"
    if r.returncode or not f.exists():
        raise RuntimeError((r.stdout + r.stderr)[-2500:])
    names = f.read_text().splitlines()[0].split()
    a = np.loadtxt(f, skiprows=1)
    col = lambda k: a[:, names.index(k)]  # noqa: E731
    mc2, bg0 = case["mass_eV"], np.sqrt(((1 + case["kinetic_eV"] / case["mass_eV"]) ** 2) - 1)
    g0 = np.sqrt(1 + bg0**2)
    beta0 = bg0 / g0
    s = col("s")
    st = np.array(case["stations_m"])
    idx = np.array([int(np.argmin(np.abs(s - v))) for v in st])
    if np.abs(s[idx] - st).max() > 1e-9:
        raise RuntimeError(f"no diagnostic row at a station (worst {np.abs(s[idx] - st).max():.2e} m)")
    ekin = (g0 - 1 - col("mean_pt")[idx] * bg0) * mc2
    return np.column_stack([st, col("sigma_x")[idx], col("sigma_y")[idx], beta0 * col("sigma_t")[idx], col("emittance_xn")[idx],
                            col("emittance_yn")[idx], ekin, col("sigma_pt")[idx] * bg0 * mc2])


def versions():
    import impactx
    return dict(impactx=impactx.__version__, impactx_file=impactx.__file__)


if __name__ == "__main__":
    run_worker("impactx", CONFIGS, build, track, versions=versions, setup=setup)

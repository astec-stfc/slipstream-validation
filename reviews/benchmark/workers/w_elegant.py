"""elegant adapter: writes an .lte and an SDDS particle file, runs the executable, reads the WATCH output with
sdds2stream.  The executable comes from BENCH_ELEGANT_EXE (bench.toml); the banner version is recorded.
Timing covers the whole subprocess plus parsing, i.e. the cost of obtaining the turn-by-turn data from a fresh run.
Elegant works in slopes (xp, yp), t and p = beta*gamma; everything is converted explicitly here."""
import math
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from common import canon_to_slopes, launches, run_worker, slopes_to_canon  # noqa: E402

# (integration order, slices per quad/sext, slices per bend)
_RUNGS = [(2, 4, 8), (4, 4, 8), (4, 8, 16), (6, 8, 16), (6, 16, 32)]
CONFIGS = [dict(name=f"cpu-f64-o{o}s{s}b{b}", device="cpu", dtype="float64", order=o, n_slices=s, n_slices_bend=b)
           for o, s, b in _RUNGS]
EXE = pathlib.Path(os.environ.get("BENCH_ELEGANT_EXE", "elegant"))
ENV = dict(os.environ, RPN_DEFNS=os.environ.get("RPN_DEFNS", str(EXE.parent / "defns.rpn")))
if os.name == "nt":
    ENV.setdefault("OMP_NUM_THREADS", "1")


def write_lte(path, lat, cfg):
    defs, line, seen = [], ["W"], {}
    for e in lat["elements"]:
        key = (e["kind"], e["length"], e.get("k1", 0.0), e.get("k2", 0.0), e.get("angle", 0.0))
        if key not in seen:
            i = len(seen)
            k, L = e["kind"], e["length"]
            o, s, b = cfg["order"], cfg["n_slices"], cfg["n_slices_bend"]
            if k == "drift":
                defs.append(f"E{i}: DRIFT,L={L!r}")
            elif k == "quad":
                defs.append(f"E{i}: KQUAD,L={L!r},K1={e['k1']!r},N_SLICES={s},INTEGRATION_ORDER={o}")
            elif k == "sext":
                defs.append(f"E{i}: KSEXT,L={L!r},K2={e['k2']!r},N_SLICES={s},INTEGRATION_ORDER={o}")
            elif k == "bend":
                defs.append(f"E{i}: CSBEND,L={L!r},ANGLE={e['angle']!r},E1=0,E2=0,N_SLICES={b},INTEGRATION_ORDER={o}")
            else:
                raise ValueError(k)
            seen[key] = f"E{i}"
        line.append(seen[key])
    pathlib.Path(path).write_text('W: WATCH,FILENAME="%s.w1",MODE="coordinates",INTERVAL=1\n' + "\n".join(defs)
                                  + "\nRING: LINE=(" + ",".join(line) + ")\n")


def setup():
    pass


def build(lat, case, cfg):
    d = pathlib.Path(tempfile.mkdtemp(prefix="bench_elegant_"))
    write_lte(d / "ring.lte", lat, cfg)
    E0, m = case["energy_eV"], case["mass_eV"]
    p0c = math.sqrt(E0**2 - m**2)
    init = launches(case)
    xp, yp = canon_to_slopes(init[:, 1], init[:, 3], init[:, 5])
    pbg = (1 + init[:, 5]) * p0c / m
    # ponytail: tau is not mapped (t = 0); a case that records tau needs the arrival-time convention probe first
    rows = np.column_stack([init[:, 0], xp, init[:, 2], yp, np.zeros(len(init)), pbg, np.arange(1, len(init) + 1)])
    np.savetxt(d / "beam.txt", rows, fmt="%.17g")
    subprocess.run(["plaindata2sdds", "beam.txt", "beam.sdds", "-inputMode=ascii", "-outputMode=binary", "-noRowCount",
                    "-column=x,double,units=m", "-column=xp,double", "-column=y,double,units=m", "-column=yp,double",
                    "-column=t,double,units=s", "-column=p,double", "-column=particleID,long"],
                   cwd=d, check=True, capture_output=True, env=ENV)
    (d / "run.ele").write_text(
        f'&run_setup lattice="ring.lte", use_beamline="RING", rootname="run", p_central={p0c / m!r}, default_order=3 &end\n'
        f'&run_control n_passes={case["turns"]} &end\n'
        '&sdds_beam input="beam.sdds", input_type="elegant" &end\n&track &end\n')
    return dict(dir=d, n=len(init), E0=E0, m=m, p0c=p0c), None


def track(model, init, turns, cfg):
    d, n = model["dir"], model["n"]
    for f in d.glob("run.*"):
        if f.name != "run.ele":
            f.unlink()
    r = subprocess.run([str(EXE), "run.ele"], cwd=d, capture_output=True, text=True, env=ENV, timeout=3600)
    if r.returncode:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-1000:])
    s = subprocess.run(["sdds2stream", "run.w1", "-col=particleID,x,xp,y,yp,p"], cwd=d, capture_output=True, text=True,
                       check=True, env=ENV).stdout.split()
    a = np.array(s, dtype=float).reshape(-1, 6)
    if a.shape[0] != n * turns:  # ponytail: loss in the reference run is an error, not handled
        raise RuntimeError(f"elegant returned {a.shape[0]} rows, expected {n * turns} (particle loss?)")
    a = a.reshape(turns, n, 6)
    a = np.take_along_axis(a, np.argsort(a[:, :, 0], axis=1)[:, :, None], axis=1)  # order by particleID within a pass
    xp, yp, delta = a[..., 2], a[..., 4], a[..., 5] / (model["p0c"] / model["m"]) - 1
    px, py = slopes_to_canon(xp, yp, delta)
    return np.stack([a[..., 1], px, a[..., 3], py], axis=-1).transpose(1, 0, 2)  # (N, T, 4)


def versions():
    r = subprocess.run([str(EXE)], capture_output=True, text=True, env=ENV)
    banner = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("This is elegant")]
    return dict(elegant=banner[0].split(",")[0] if banner else None, exe=str(EXE))


if __name__ == "__main__":
    run_worker("elegant", CONFIGS, build, track, versions=versions, setup=setup)

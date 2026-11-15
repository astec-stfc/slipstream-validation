"""OPAL-X adapter for ``sc_drift`` (OPAL-X 1.0.0, IPPL + Kokkos; CPU/OpenMP or CUDA).  Each call writes ``opalx.in`` and the frozen macroparticle
file (``FROMFILE``: x px y py z pz, absolute beta*gamma, positions at fixed time: the case's ``lab_fixed_t`` frame), runs ``opalx`` in a scratch
directory with one MPI rank and reads its ``.stat`` file (ASCII SDDS, one row per time step with ``STATDUMPFREQ=1``).  Rows are fixed-time
statistics of all particles: ``rms_x/rms_y/rms_s`` [m], ``emit_x/emit_y`` normalised [m], ``energy`` mean kinetic energy [MeV], ``dE`` rms
kinetic-energy spread [MeV], ``s`` reference path [m], ``mean_s`` centroid offset [m].  They are interpolated to the case stations against the bunch-mean z = ``s + mean_s``, as the ASTRA
worker does against the ``z`` of its tables; station 0 is computed from the frozen particles themselves (the first stat row is already one step on).
Timing covers the whole subprocess (process start, MPI and Kokkos initialisation, CUDA context, I/O) plus parsing, like the ASTRA and GPT workers.

Where OPAL-X comes from is not in the worker: ``BENCH_OPALX_CUDA_EXE`` is the CUDA build's ``opalx`` (``setup/install_opalx.sh``), and the CPU
rungs use ``BENCH_OPALX_CPU_EXE`` (a ``PLATFORMS=OPENMP`` build) or, when that is unset and ``docker`` exists, the official image
``BENCH_OPALX_CPU_IMAGE`` (default ``ghcr.io/opalx-project/opalx:cpu``).  A configuration whose launcher is missing is not listed.
Configurations: ``<device>-f64-open<cells>h<step in ps>`` = open-boundary (Hockney, integrated Green function) FFT solver on a cells^3 mesh, solved in the
rest frame of one bin (``BINNING, MAXBINS=1``: Lorentz boost and z stretch with the bunch-mean gamma).  The suffix ``-mono`` is the monolithic solve
(``BINS=NONE``): in OPAL-X 1.0.0 it uses gamma = 1 (``CartesianPIC3DAlgorithm::solvePass``: no boost, no stretch), so it is not valid for a relativistic
bunch and is kept only as a documented disagreement (10 November 2026: transverse size 10 times ASTRA's at 3 m for this 5 MeV case).  OPAL-X is double precision only.
Traps: (1) ``FROMFILE`` cannot set the beam reference momentum (the BEAM stays at 1e-9 GeV/c), so the bin gamma must come from the particles, hence the bin
solve; (2) with WSL2 mirrored networking OpenMPI start-up hangs for ever in hwloc's X11/GL probe (TCP connect to 127.0.0.1:6001), also for ``mpirun hostname``;
``HWLOC_COMPONENTS=-gl`` (set below for native builds) cures it; (3) on a WSL2 host a stale libnvidia-ptxjitcompiler.so.580.* in /lib/x86_64-linux-gnu
(not from any package) shadows the one of the WSL driver and ``cuInit`` segfaults in it (any Kokkos CUDA program, toolkit 13.3 and 13.4, 10 November 2026);
the driver's own directory in /usr/lib/wsl/drivers goes first on LD_LIBRARY_PATH for CUDA runs.  Run the worker under ``timeout``: a wedged MPI start hangs for ever."""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from common import launches, run_worker  # noqa: E402

C = 299792458.0
IMAGE = os.environ.get("BENCH_OPALX_CPU_IMAGE", "ghcr.io/opalx-project/opalx:cpu")
# (cells per dimension, time step [ps]); the CPU rungs stop at 64^3 (a 128^3 open solve is a 256^3 FFT per step), the GPU ladder goes one rung finer
_CPU = [(32, 40.0), (64, 40.0), (64, 10.0)]
_CUDA = [(32, 40.0), (64, 40.0), (64, 10.0), (128, 40.0)]
_MONO = (32, 40.0)   # the monolithic (gamma = 1) rung, one per device


def _have_cpu():
    return bool(os.environ.get("BENCH_OPALX_CPU_EXE") or shutil.which("docker"))


CONFIGS = []
if _have_cpu():
    CONFIGS += [dict(name=f"cpu-f64-open{_MONO[0]}h{_MONO[1]:g}-mono", device="cpu", dtype="float64", n=_MONO[0], h_ps=_MONO[1], mono=True)]
    CONFIGS += [dict(name=f"cpu-f64-open{n}h{h:g}", device="cpu", dtype="float64", n=n, h_ps=h, mono=False) for n, h in _CPU]
if os.environ.get("BENCH_OPALX_CUDA_EXE"):
    CONFIGS += [dict(name=f"cuda-f64-open{_MONO[0]}h{_MONO[1]:g}-mono", device="cuda", dtype="float64", n=_MONO[0], h_ps=_MONO[1], mono=True)]
    CONFIGS += [dict(name=f"cuda-f64-open{n}h{h:g}", device="cuda", dtype="float64", n=n, h_ps=h, mono=False) for n, h in _CUDA]
_S = {}


def setup():
    if not CONFIGS:
        raise RuntimeError("no OPAL-X launcher: set BENCH_OPALX_CUDA_EXE and/or BENCH_OPALX_CPU_EXE (or install docker for the official CPU image)")


def _wsl_driver_dir():
    """The WSL2 driver directory that holds libnvidia-ptxjitcompiler.so.1 (newest), or "" on a host without one."""
    found = sorted(pathlib.Path("/usr/lib/wsl/drivers").glob("*/libnvidia-ptxjitcompiler.so.1")) if pathlib.Path("/usr/lib/wsl/drivers").exists() else []
    return str(found[-1].parent) if found else ""


def _command(cfg, d):
    """(argv, environment) that runs ``opalx opalx.in`` in directory ``d`` with one rank."""
    threads = os.environ.get("OMP_NUM_THREADS", "1")
    env = dict(os.environ, OMP_NUM_THREADS=threads, OMP_PROC_BIND="false", HWLOC_COMPONENTS=os.environ.get("HWLOC_COMPONENTS", "-gl"))   # -gl: see the traps above
    if cfg["device"] == "cuda":
        env["LD_LIBRARY_PATH"] = ":".join(x for x in (_wsl_driver_dir(), env.get("LD_LIBRARY_PATH", "")) if x)
        return [os.environ["BENCH_OPALX_CUDA_EXE"], "opalx.in", "--info", "2"], env
    exe = os.environ.get("BENCH_OPALX_CPU_EXE")
    if exe:
        return [exe, "opalx.in", "--info", "2"], env
    user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "0:0"
    return (["docker", "run", "--rm", "--user", user, "-e", f"OMP_NUM_THREADS={threads}", "-v", f"{d}:/data", IMAGE, "opalx.in", "--info", "2"],
            dict(os.environ))


def _deck(case, cfg, n_part, dt_s, zstop):
    beta0 = np.sqrt(1.0 - 1.0 / (1.0 + case["kinetic_eV"] / case["mass_eV"]) ** 2)
    max_steps = int(np.ceil(zstop / (beta0 * C * dt_s))) + 20
    n = cfg["n"]
    binning = "" if cfg["mono"] else "BINS1: BINNING, MAXBINS=1, TABLEPRINTFREQ=0;\n"
    bins = "" if cfg["mono"] else ", BINS=BINS1"
    return f"""OPTION, PSDUMPFREQ=100000000;
OPTION, STATDUMPFREQ=1;
Title, string="sc_drift {cfg['name']}";
D1: DRIFT, L={zstop:.6f}, ELEMEDGE=0.0;
myLine: LINE=(D1);
{binning}FS1: FIELDSOLVER, TYPE=OPEN{bins}, NX={n}, NY={n}, NZ={n}, PARFFTX=TRUE, PARFFTY=TRUE, PARFFTZ=TRUE,
    BCFFTX=OPEN, BCFFTY=OPEN, BCFFTZ=OPEN, GREENSF=INTEGRATED;
Dist1: DISTRIBUTION, TYPE=FROMFILE, FNAME="dist.txt", NPARTDIST={n_part};
ES1: EMISSIONSOURCE, DISTRIBUTION=Dist1;
mySources: EMISSIONSOURCELIST=(ES1);
BEAM1: BEAM, PARTICLE=ELECTRON, NALLOC={n_part}, BCHARGE={abs(case['charge_C']):.12e}, SOURCES=mySources, CHARGE=-1;
TRACK, LINE=myLine, BEAM=BEAM1, MAXSTEPS={max_steps}, DT={{{dt_s:.6e}}}, ZSTOP={zstop:.6f};
RUN, METHOD="PARALLEL", FIELDSOLVER=FS1;
ENDTRACK;
QUIT;
"""


def build(lat, case, cfg):
    d = pathlib.Path(tempfile.mkdtemp(prefix="bench_opalx_"))
    if cfg["device"] == "cuda" and cfg["n"] > 128:
        raise RuntimeError("the 8 GiB card also runs the desktop: no open-boundary mesh above 128^3")   # AGENTS.md: never spill the GPU into host RAM
    # the input files are written here, outside the timed call (as the ASTRA worker does): a call covers the process, its I/O and the parsing
    init = launches(case)
    with open(d / "dist.txt", "w", newline="\n") as fh:
        fh.write(f"{len(init)}\nx px y py z pz\n")
        np.savetxt(fh, init[:, [0, 3, 1, 4, 2, 5]], fmt="%.17e")   # case columns are x y z bgx bgy bgz
    (d / "opalx.in").write_bytes(_deck(case, cfg, len(init), cfg["h_ps"] * 1e-12, case["stations_m"][-1] + 0.02).encode())   # bytes: LF on every platform
    return dict(dir=d, case=case, cfg=cfg), None


def read_stat(path):
    """ASCII SDDS stat file -> dict column name -> array (one row per time step)."""
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    head, _, body = text.partition("&data")
    names = re.findall(r"&column\s+name=(\w+),", head)
    lines = body.split("&end", 1)[1].lstrip("\n").split("\n")[3:]   # three parameter lines (processors, revision, species) precede the rows
    a = np.array([[float(v) for v in ln.split()] for ln in lines if ln.strip()])
    if a.shape[1] != len(names):
        raise RuntimeError(f"stat file has {a.shape[1]} columns, its header {len(names)}")
    return {k: a[:, i] for i, k in enumerate(names)}


def _initial(init, mc2):
    """Station-0 row from the frozen particles: sigma_x, sigma_y, sigma_z, eps_nx, eps_ny [m], mean Ekin and rms spread [eV]."""
    x, y, z, px, py, pz = init.T
    ek = (np.sqrt(1.0 + px**2 + py**2 + pz**2) - 1.0) * mc2
    eps = lambda q, p: np.sqrt(np.var(q) * np.var(p) - np.mean((q - q.mean()) * (p - p.mean())) ** 2)  # noqa: E731
    return np.array([z.mean(), x.std(), y.std(), z.std(), eps(x, px), eps(y, py), ek.mean(), ek.std()])


def _gpu_peak(stop, out):
    """Used GPU memory [MiB] of the whole card at the start and its peak during the run, sampled with nvidia-smi (the desktop's own share is in the start value)."""
    peak, start = 0, None
    while not stop.is_set():
        try:
            used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                      timeout=10).stdout.split()[0])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            break
        start = used if start is None else start
        peak = max(peak, used)
        stop.wait(0.25)
    out.update(gpu_used_start_mib=start, gpu_used_peak_mib=peak)


def _sidecar(d, files):
    """Keep the deck, the log and the stat file beside result.json (the scratch directory is temporary)."""
    argv = sys.argv
    if "--job" not in argv:
        return
    out = pathlib.Path(json.loads(pathlib.Path(argv[argv.index("--job") + 1]).read_text())["out_dir"])
    for f in files:
        if (d / f).exists():
            shutil.copy(d / f, out / f)


def track(model, init, turns, cfg):
    d, case = model["dir"], model["case"]
    for f in {*d.glob("*.stat"), *d.glob("*.h5"), *d.glob("run.log")}:   # outputs of the previous call; the input files written by build() stay
        f.unlink()
    n_part = len(init)
    cmd, env = _command(cfg, d)
    mem, stop = {}, threading.Event()
    watcher = threading.Thread(target=_gpu_peak, args=(stop, mem), daemon=True) if cfg["device"] == "cuda" else None
    if watcher:
        watcher.start()
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=7200, env=env)
    wall = time.perf_counter() - t0
    stop.set()
    (d / "run.log").write_text(r.stdout + r.stderr)
    rev = re.search(r"git rev\. (\w+)", r.stdout + r.stderr)
    _S["opalx_git_rev"] = rev.group(1) if rev else None
    stats = sorted(d.glob("*.stat"))
    if r.returncode or not stats:
        raise RuntimeError((r.stdout + r.stderr)[-2500:])
    s = read_stat(stats[0])
    if not np.isfinite(np.column_stack(list(s.values()))).all():
        raise RuntimeError("non-finite statistics in the stat file")
    if int(s["numParticles"].min()) != n_part:
        raise RuntimeError(f"particles lost: {int(s['numParticles'].min())} of {n_part} left")
    mc2 = case["mass_eV"]
    st = np.array(case["stations_m"])
    row0 = _initial(init, mc2)
    z = np.concatenate([[row0[0]], s["s"] + s["mean_s"]])   # absolute bunch-mean z: ``s`` is the reference path, ``mean_s`` the centroid offset from it
    if np.any(np.diff(z) <= 0) or z[-1] < st[-1]:
        raise RuntimeError(f"mean z is not monotonic or stops short of the last station ({z[-1]:.4f} m)")
    cols = [np.concatenate([[row0[1]], s["rms_x"]]), np.concatenate([[row0[2]], s["rms_y"]]), np.concatenate([[row0[3]], s["rms_s"]]),
            np.concatenate([[row0[4]], s["emit_x"]]), np.concatenate([[row0[5]], s["emit_y"]]),
            np.concatenate([[row0[6]], s["energy"] * 1e6]), np.concatenate([[row0[7]], s["dE"] * 1e6])]
    _S.update(cfg=cfg, wall_s=wall, steps=len(s["s"]), **mem)
    _sidecar(d, ["opalx.in", "run.log", stats[0].name])
    return np.column_stack([st] + [np.interp(st, z, c) for c in cols])


def versions():
    cfg = _S.get("cfg", CONFIGS[-1])   # the launcher of the last job, else the finest configuration
    d = pathlib.Path(tempfile.mkdtemp(prefix="bench_opalx_v_"))
    cmd, env = _command(cfg, d)
    cmd = ["docker", "run", "--rm", IMAGE, "--version-full"] if cmd[0] == "docker" else [cmd[0], "--version-full"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env, cwd=d)
    keep = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if re.search(r"OPAL|IPPL|Kokkos|git|Last build|backend|CUDA|OpenMP", ln) and "WARNING" not in ln]
    out = dict(opalx_cuda_exe=os.environ.get("BENCH_OPALX_CUDA_EXE"), opalx_cpu_exe=os.environ.get("BENCH_OPALX_CPU_EXE"), opalx_cpu_image=IMAGE, banner=keep[:20], last_call={k: v for k, v in _S.items() if k != "cfg"})
    if shutil.which("docker"):
        img = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], capture_output=True, text=True)
        out["opalx_cpu_image_id"] = img.stdout.strip() or None
    return out


if __name__ == "__main__":
    run_worker("opalx", CONFIGS, build, track, versions=versions, setup=setup)

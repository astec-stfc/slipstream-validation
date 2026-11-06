"""Numpy-only helpers shared by every benchmark worker, the orchestrator and the analysis.

Workers run inside each code's own environment (Windows conda, WSL conda, ...), so this module must import
with nothing but numpy.  Heavy imports (torch, xtrack, cheetah, ...) belong inside the worker's ``build``.

Benchmark standard frame (BSF), the only particle representation that crosses a process boundary:
    (x, px, y, py)   x, y [m]; px, py = canonical transverse momentum / P0 (dimensionless)
    delta            (p - p0) / p0
    tau              late-positive arrival coordinate [m]  (only where a case declares it)
Every code adapter converts to and from BSF explicitly; nothing is assumed to match by name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = pathlib.Path(os.environ.get("SSV_RESULTS") or ROOT.parents[1] / "results")  # campaigns: results/TAG/CASE/CODE/CONFIG/pK
MC2_EV = 0.51099895e6
SCHEMA = 1


# ----------------------------------------------------------------------------------------- files and hashes
def sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load(path) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def lattice_inventory(lat: dict) -> dict:
    """Integrated quantities every adapter must reproduce from its native model (P0 import audit)."""
    els = lat["elements"]
    return dict(
        n=len(els),
        length=sum(e["length"] for e in els),
        angle=sum(e.get("angle", 0.0) for e in els),
        k1L_abs=sum(abs(e.get("k1", 0.0)) * e["length"] for e in els),
        k2L_abs=sum(abs(e.get("k2", 0.0)) * e["length"] for e in els),
    )


def launches(case: dict) -> np.ndarray:
    """(N, 6) float64 initial BSF states (x, px, y, py, tau, delta) from the case file.  A case with ``particles.file`` (frame
    ``lab_fixed_t``) instead returns the frozen macroparticles as (N, 6) columns ``x, y, z, bgx, bgy, bgz`` (m; beta*gamma)."""
    p = case["particles"]
    if "file" in p:
        z = np.load(ROOT / "cases" / p["file"])
        return np.column_stack([z[k] for k in ("x", "y", "z", "bgx", "bgy", "bgz")]).astype(float)
    x0 = np.atleast_1d(np.asarray(p["x0"], dtype=float))
    s = np.zeros((len(x0), 6))
    s[:, 0] = x0
    s[:, 1] = p.get("px0", 0.0)
    s[:, 2] = p.get("y0", 0.0)
    s[:, 3] = p.get("py0", 0.0)
    s[:, 5] = p.get("delta", 0.0)
    return s


# ----------------------------------------------------------------------------------------- frame conversions
def p0c_eV(E0_eV: float, m_eV: float = MC2_EV) -> float:
    return math.sqrt(E0_eV**2 - m_eV**2)


def slopes_to_canon(xp, yp, delta):
    """Slopes x' = dx/ds, y' -> canonical px/P0, py/P0 (no vector potential)."""
    xp, yp, delta = map(np.asarray, (xp, yp, delta))
    d = (1.0 + delta) / np.sqrt(1.0 + xp**2 + yp**2)
    return xp * d, yp * d


def canon_to_slopes(px, py, delta):
    px, py, delta = map(np.asarray, (px, py, delta))
    pz = np.sqrt((1.0 + delta) ** 2 - px**2 - py**2)
    return px / pz, py / pz


def delta_to_cheetah_p(delta, E0_eV, m_eV: float = MC2_EV):
    """Cheetah's last coordinate (E - E0)/(p0 c), in the cancellation-free form."""
    delta = np.asarray(delta, dtype=float)
    p0c = p0c_eV(E0_eV, m_eV)
    E = np.sqrt((p0c * (1 + delta)) ** 2 + m_eV**2)
    return p0c * (2 * delta + delta**2) / (E + E0_eV)


def cheetah_p_to_delta(p, E0_eV, m_eV: float = MC2_EV):
    p = np.asarray(p, dtype=float)
    p0c = p0c_eV(E0_eV, m_eV)
    pc2_minus = p * p0c * (2 * E0_eV + p * p0c)  # = (pc)^2 - (p0c)^2 = E^2 - E0^2
    pc = np.sqrt(p0c**2 + pc2_minus)
    return pc2_minus / (p0c * (pc + p0c))


# ----------------------------------------------------------------------------------------- station statistics
def lab_stats(x, y, z, bgx, bgy, bgz, mass_eV: float = MC2_EV, w=None):
    """Bunch statistics of a lab-frame snapshot (m, beta*gamma): sigma_x, sigma_y, sigma_z, normalised rms emittances (canonical,
    x versus bgx; mean removed), mean kinetic energy and rms total-energy spread [eV].  ``w`` are optional macroparticle weights."""
    x, y, z, bgx, bgy, bgz = (np.asarray(a, dtype=float) for a in (x, y, z, bgx, bgy, bgz))
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    w = w / w.sum()
    m = lambda a: float(np.sum(w * a))  # noqa: E731
    cov = lambda a, b: m(a * b) - m(a) * m(b)  # noqa: E731
    e = mass_eV * np.sqrt(1 + bgx**2 + bgy**2 + bgz**2)
    eps = lambda a, b: float(np.sqrt(max(cov(a, a) * cov(b, b) - cov(a, b) ** 2, 0.0)))  # noqa: E731
    return dict(sigma_x_m=cov(x, x) ** 0.5, sigma_y_m=cov(y, y) ** 0.5, sigma_z_m=cov(z, z) ** 0.5, eps_nx_m=eps(x, bgx),
                eps_ny_m=eps(y, bgy), ekin_eV=m(e) - mass_eV, sigma_E_eV=cov(e, e) ** 0.5)


# ----------------------------------------------------------------------------------------- tune estimator
def naff(z, center: float, hw: float, refine: int = 60):
    """Dominant frequency [cycles/turn] of complex series ``z`` inside center +- hw, and its amplitude.

    Hann-windowed FFT peak, then golden-section refinement of |sum w z exp(-2 pi i nu t)|.  Both rotation senses
    are tried because the sign of the (x, px) phase advance depends on the code's conventions.  The one estimator
    is applied to every code's turn-by-turn data so that estimator differences cannot masquerade as code differences.
    """
    # ponytail: first line only (no iterative subtraction); add Laskar's full NAFF if secondary lines matter
    z = np.asarray(z, dtype=complex)
    n = len(z)
    z = z - z.mean()
    t = np.arange(n)
    w = np.hanning(n)
    pad = 8 * n
    f = np.arange(pad) / pad
    win = (f >= center - hw) & (f <= center + hw)
    gr = (math.sqrt(5) - 1) / 2
    best = (center, -1.0)
    for s in (z, z.conj()):
        zs = s * w
        k = np.flatnonzero(win)[np.argmax(np.abs(np.fft.fft(zs, pad))[win])]

        def amp(nu, zs=zs):
            return abs(np.dot(zs, np.exp(-2j * np.pi * nu * t)))

        a, b = f[k] - 1.0 / n, f[k] + 1.0 / n
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = amp(c), amp(d)
        for _ in range(refine):
            if fc > fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = amp(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = amp(d)
        nu = 0.5 * (a + b)
        am = amp(nu) / w.sum()
        if am > best[1]:
            best = (nu, am)
    return best


def tunes(tbt, case: dict):
    """tbt (N, T, >=4) BSF -> (qx[N], qy[N]) with the case's search windows."""
    cx, hx = case["tune_windows"]["x"]
    cy, hy = case["tune_windows"]["y"]
    qx = [naff(t[:, 0] + 1j * t[:, 1], cx, hx)[0] for t in tbt]
    qy = [naff(t[:, 2] + 1j * t[:, 3], cy, hy)[0] for t in tbt]
    return np.array(qx), np.array(qy)


# ----------------------------------------------------------------------------------------- timing / provenance
def stats(samples) -> dict:
    a = np.asarray(samples, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return dict(n=int(a.size), median=float(med), q1=float(q1), q3=float(q3), min=float(a.min()), max=float(a.max()),
                samples=[float(v) for v in a])


def _cmd(args, cwd=None, timeout=10):
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def git_state(path) -> dict:
    """Commit, dirty flag and a hash of the uncommitted diff, so a dirty tree is still identifiable."""
    path = str(path)
    commit = _cmd(["git", "-C", path, "rev-parse", "HEAD"])
    if commit is None:
        return dict(path=path, commit=None)
    diff = _cmd(["git", "-C", path, "diff", "HEAD"], timeout=30) or ""
    return dict(path=path, commit=commit, dirty=bool(_cmd(["git", "-C", path, "status", "--porcelain"])),
                diff_sha256=hashlib.sha256(diff.encode()).hexdigest() if diff else None)


def gpu_state() -> dict | None:
    q = "name,driver_version,memory.total,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu"
    out = _cmd(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"])
    return dict(zip(q.split(","), [v.strip() for v in out.splitlines()[0].split(",")])) if out else None


def rss_peak_bytes() -> int | None:
    try:
        import psutil

        mi = psutil.Process().memory_info()
        return int(getattr(mi, "peak_wset", mi.rss))
    except Exception:  # noqa: BLE001
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:  # noqa: BLE001
            return None


def provenance() -> dict:
    thr = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}
    return dict(host=socket.gethostname(), platform=platform.platform(), python=sys.version.split()[0],
                cpu_count=os.cpu_count(), threads_env=thr, gpu=gpu_state(), argv=sys.argv,
                time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


# ----------------------------------------------------------------------------------------- gradient flow
def scaled_lattice(lat: dict, knobs: list, factors) -> dict:
    """Copy of the lattice with each knob family's strength multiplied by its factor."""
    import copy

    out = copy.deepcopy(lat)
    for kn, f in zip(knobs, factors):
        for e in out["elements"]:
            if e["name"] == kn["family"]:
                e[kn["kind"]] *= float(f)  # plain float: elegant/Julia writers use repr()
    return out


def nominal_strengths(lat: dict, knobs: list) -> np.ndarray:
    v = []
    for kn in knobs:
        vals = {e[kn["kind"]] for e in lat["elements"] if e["name"] == kn["family"]}
        assert len(vals) == 1, f"family {kn['family']} has {len(vals)} distinct strengths"
        v.append(vals.pop())
    return np.array(v)


def grad_flow(case, lat, cfg, build, track, ad, init, turns, sync, dev, out, repeats, fd_update=None, model0=None):
    """Central-difference Jacobian dy/dtheta for several relative steps (any code) and the code's own AD Jacobian.
    y = (x, px, y, py) of particle 0 at the start of turn ``turns - 1``; theta = the family strengths in ``case['knobs']``."""
    knobs, K = case["knobs"], len(case["knobs"])
    k0 = nominal_strengths(lat, knobs)
    steps = case["fd_rel_steps"]

    def y_of(factors):
        lat_f = scaled_lattice(lat, knobs, factors)
        if fd_update is not None:
            fd_update(model0, lat_f, case, cfg)
            m = model0
        else:
            m, _ = build(lat_f, case, cfg)
        return track(m, init[:1], turns, cfg)[0, -1, :4]

    y_of(np.ones(K))  # warm-up: imports, JIT, kernel caches
    sync(dev)
    res, fd, fd_s = {}, [], []
    for h in steps:
        t0 = time.perf_counter()
        J = np.zeros((4, K))
        for j in range(K):
            up, dn = np.ones(K), np.ones(K)
            up[j] += h
            dn[j] -= h
            J[:, j] = (y_of(up) - y_of(dn)) / (2 * h * k0[j])
        sync(dev)
        fd_s.append(time.perf_counter() - t0)
        fd.append(J)
    y0 = y_of(np.ones(K))
    m, _ = build(lat, case, cfg)  # fresh model at the nominal strengths for the forward timing
    sync(dev)
    t0 = time.perf_counter()
    track(m, init[:1], turns, cfg)
    sync(dev)
    res["forward_s"] = time.perf_counter() - t0
    res["fd_s"] = dict(zip(map(str, steps), fd_s))
    arrays = dict(y0=y0, J_fd=np.array(fd), steps=np.array(steps), k0=k0)
    if ad is not None:
        ad_build, ad_jac = ad
        t0 = time.perf_counter()
        am = ad_build(lat, case, cfg, knobs)
        sync(dev)
        res["ad_build_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        ya, Ja = ad_jac(am, init[:1], turns, cfg, knobs)
        sync(dev)
        res["ad_cold_s"] = time.perf_counter() - t0
        warm = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            ya, Ja = ad_jac(am, init[:1], turns, cfg, knobs)
            sync(dev)
            warm.append(time.perf_counter() - t0)
        res["ad_warm_s"] = stats(warm)
        arrays.update(y_ad=np.asarray(ya, dtype=float), J_ad=np.asarray(Ja, dtype=float))
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite gradient output: {name}")
    np.savez_compressed(out / "grad.npz", **arrays)
    return res


# ----------------------------------------------------------------------------------------- worker harness
def run_worker(code, configs, build, track, sync=lambda dev: None, peak_mem=lambda dev: {}, versions=lambda: {},
               setup=lambda: None, ad=None, fd_update=None):
    """Common worker main.

    configs     ordered list of dicts ``{name, device, dtype, ...}``; ordering is coarse -> fine, the last is the
                code's own converged reference.
    setup       imports the heavy libraries; timed separately (``import_s``) from ``build`` (``build_s``).
    ad          optional ``(ad_build, ad_jacobian)`` for cases with ``"kind": "grad"``: ``ad_build(lat, case, cfg, knobs)`` ->
                model; ``ad_jacobian(model, init, turns, cfg, knobs)`` -> (y[4], J[4, K]) by the code's own automatic
                differentiation.  Every code also gets a central finite-difference Jacobian from ``build``/``track``.
    build       build(lattice, case, cfg) -> (model, inventory dict or None)
    track       track(model, init(N,6) float64 ndarray, turns, cfg) -> (N, T, 4) float64 ndarray in BSF, state at the
                START of turns 0..T-1 (initial condition included).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--job")
    a = ap.parse_args()
    if a.list:
        print(json.dumps(configs))
        return
    job = load(a.job)
    out = pathlib.Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    case = load(job["case_path"])
    lat = load(ROOT / "cases" / case["lattice"])
    cfg = next(c for c in configs if c["name"] == job["config"])
    rec = dict(schema=SCHEMA, code=code, case=case["id"], config=cfg, rank=configs.index(cfg), job=job,
               status="error", warnings=[], provenance=provenance(), case_sha256=sha256(job["case_path"]),
               lattice_sha256=sha256(ROOT / "cases" / case["lattice"]))
    rec["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p)
                             for p in sorted(ROOT.rglob("*.py")) if "results" not in p.parts}
    if "file" in case["particles"]:
        rec["particles_sha256"] = sha256(ROOT / "cases" / case["particles"]["file"])
    try:
        dev = cfg.get("device")
        t0 = time.perf_counter()
        setup()
        t_import = time.perf_counter() - t0
        t0 = time.perf_counter()
        model, inv = build(lat, case, cfg)
        sync(dev)
        t_build = time.perf_counter() - t0
        want = lattice_inventory(lat)
        if inv is not None:
            rec["inventory"] = dict(model=inv, reference=want)
            for k, v in inv.items():
                if k in want and abs(v - want[k]) > 1e-9 * max(1.0, abs(want[k])):
                    rec["warnings"].append(f"inventory mismatch {k}: model {v!r} vs lattice {want[k]!r}")
        init = launches(case)
        turns = int(case["turns"])
        if case.get("kind") == "grad":
            rec.update(grad_flow(case, lat, cfg, build, track, ad, init, turns, sync, dev, out, int(job.get("repeats", 3)), fd_update, model))
            rec.update(status="ok", versions=versions(), timing=dict(import_s=t_import, build_s=t_build),
                       memory=dict(rss_peak_bytes=rss_peak_bytes(), **peak_mem(dev)))
            (out / "result.json").write_text(json.dumps(rec, indent=1, default=str))
            sys.exit(0)

        def timed():
            sync(dev)
            t = time.perf_counter()
            r = track(model, init, turns, cfg)
            sync(dev)
            return r, time.perf_counter() - t

        first, t_cold = timed()
        if not np.isfinite(first).all():
            raise ValueError("Non-finite tracking output")
        warm, last = [], first
        for _ in range(int(job.get("repeats", 3))):
            last, dt = timed()
            if not np.isfinite(last).all():
                raise ValueError("Non-finite warm tracking output")
            warm.append(dt)
        np.savez_compressed(out / "tbt.npz", tbt=last, init=init)
        diff = float(np.nanmax(np.abs(first - last))) if first.shape == last.shape else None
        rec.update(status="ok", versions=versions(),
                   timing=dict(import_s=t_import, build_s=t_build, cold_s=t_cold, warm_s=stats(warm) if warm else None),
                   memory=dict(rss_peak_bytes=rss_peak_bytes(), **peak_mem(dev)),
                   repeat_maxdiff=diff, tbt_shape=list(last.shape))
        if diff:
            rec["warnings"].append(f"cold and warm outputs differ by {diff:.3e} (non-deterministic or state-carrying)")
    except Exception:  # noqa: BLE001  # failures are data: record them, never drop them
        rec["error"] = traceback.format_exc()
    (out / "result.json").write_text(json.dumps(rec, indent=1, default=str))
    sys.exit(0 if rec["status"] == "ok" else 1)

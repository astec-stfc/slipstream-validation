"""Post-campaign diagnostic: localise ring disagreements to single-element maps.

Run only after the timed campaign, to avoid competing for host resources:
python probe_elements.py --campaign 2026-10-26_R1
These probes are exploratory fidelity controls, not additional timing samples.
"""
import argparse
import importlib
import itertools
import json
import pathlib
import subprocess
import sys

import numpy as np
import common
import run

HERE = pathlib.Path(__file__).resolve().parent
SETTINGS = dict(slipstream="cpu-f64-o6k8b16", elegant="cpu-f64-o6s16b32", xsuite="cpu-f64-yosk16b32",
                cheetah="cpu-f64-dri16", jutrack="cpu-f64-n32b64", bmadx="cpu-f64-q1s64", scibmad="cpu-f64-o6k8")


def worker(code, out):
    sys.path.insert(0, str(HERE / "workers"))
    w = importlib.import_module(f"w_{code}")
    cfg = next(c for c in w.CONFIGS if c["name"] == SETTINGS[code])
    case = common.load(HERE / "cases" / "ring_tunes.json")
    lat = common.load(HERE / "cases" / case["lattice"])
    states = np.array([[x, px, y, py, 0, d] for x, y, d, (px, py) in
                       itertools.product([-8e-3, -1e-3, 0., 1e-3, 8e-3], [-1e-3, 0., 1e-3], [-.01, 0., .01],
                                         [(1e-4, -2e-4), (.01, -.002), (-.01, .002)])])
    # Executable adapters freeze particles/turns at build time; supply the same
    # probe case to every builder rather than relying on track() arguments.
    case = dict(case, turns=2, particles={k: states[:, i].tolist() for k, i in
                (("x0", 0), ("px0", 1), ("y0", 2), ("py0", 3), ("delta", 5))})
    w.setup()
    results = {}
    elements = {}
    for kind in ("drift", "quad", "sext", "bend"):
        e = next(e for e in lat["elements"] if e["kind"] == kind and (kind != "sext" or e["k2"] != 0))
        model, _ = w.build(dict(elements=[e]), case, cfg)
        results[kind] = w.track(model, states, 2, cfg)[:, 1, :4]
        assert np.isfinite(results[kind]).all(), f"Non-finite {kind} output"
        elements[kind] = e
    np.savez_compressed(out / "maps.npz", initial=states, **results)
    (out / "result.json").write_text(json.dumps(dict(code=code, config=cfg, status="ok", elements=elements,
        provenance=common.provenance(), versions=w.versions(), case_sha256=common.sha256(HERE / "cases" / "ring_tunes.json"),
        probe_sha256=common.sha256(__file__)), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign")
    ap.add_argument("--worker", choices=SETTINGS)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.worker:
        worker(a.worker, pathlib.Path(a.out))
        return
    if not a.campaign:
        ap.error("--campaign is required")
    root = common.RESULTS / a.campaign / "element_probes"
    root.mkdir(exist_ok=True)
    rows = []
    for code in SETTINGS:
        out = root / code
        out.mkdir(exist_ok=True)
        args = ["--worker", code, "--out", run.plat(code, out)]
        cmd, env = run.command(code, args, 1, "ring_tunes")
        old = run.plat(code, HERE / run.worker_path(code, "ring_tunes"))
        cmd[cmd.index(old)] = run.plat(code, pathlib.Path(__file__).resolve())
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
            if r.returncode:
                raise RuntimeError(r.stderr[-3000:] or r.stdout[-3000:])
        except Exception as exc:
            (out / "result.json").write_text(json.dumps(dict(code=code, status="error", error=str(exc)), indent=2))
        print(code, common.load(out / "result.json")["status"], flush=True)
    ref = np.load(root / "elegant" / "maps.npz") if (root / "elegant" / "maps.npz").exists() else None
    if ref is not None:
        initial = ref["initial"]
        length = common.load(root / "elegant" / "result.json")["elements"]["drift"]["length"]
        pz = np.sqrt((1 + initial[:, 5])**2 - initial[:, 1]**2 - initial[:, 3]**2)
        drift_exact = initial[:, :4].copy()
        drift_exact[:, 0] += length * initial[:, 1] / pz
        drift_exact[:, 2] += length * initial[:, 3] / pz
    for code in SETTINGS:
        rec = common.load(root / code / "result.json")
        if rec["status"] != "ok" or ref is None:
            rows.append(dict(code=code, status=rec["status"], error=rec.get("error")))
            continue
        z = np.load(root / code / "maps.npz")
        for kind in ("drift", "quad", "sext", "bend"):
            diff = np.abs(z[kind] - ref[kind])
            on = ref["initial"][:, 5] == 0
            row = dict(code=code, element=kind, config=SETTINGS[code],
                max_component_errors=dict(zip(("x_m", "px", "y_m", "py"), diff.max(axis=0).tolist())),
                on_momentum_errors=dict(zip(("x_m", "px", "y_m", "py"), diff[on].max(axis=0).tolist())))
            if kind == "drift":
                row["analytic_errors"] = dict(zip(("x_m", "px", "y_m", "py"), np.abs(z[kind] - drift_exact).max(axis=0).tolist()))
            rows.append(row)
    evidence = dict(reference="elegant", interpretation="Exploratory single-element residuals; no performance ranking or full-ring closure implied.", rows=rows)
    (root / "summary.json").write_text(json.dumps(evidence, indent=2))
    dest = common.EVIDENCE / a.campaign / "element_probes.json"
    dest.write_text(json.dumps(evidence, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()

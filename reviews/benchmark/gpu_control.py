"""Exploratory matched-model GPU control; run AFTER the primary timed campaign.

Motivation (declared before its outputs): native CPU Cartesian 3-D is much
closer to the ASTRA longitudinal spread than the cylindrical GPU settings.
Hold mesh, step and refresh fixed; vary device/precision, without changing R1.
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import common
from analyse_sc import OBS, rel

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="2026-10-26_R1", help="primary reference campaign")
    a = ap.parse_args()
    name = a.campaign + "_GPU_control"
    root = common.RESULTS / name
    root.mkdir(exist_ok=True)
    design = dict(classification="exploratory follow-up; primary R1 design unchanged", reference_campaign=a.campaign,
                  question="3-D Cartesian parity and cost at fixed 32^3 mesh, 1 ps, refresh every step",
                  configs=["cuda-f32-3d32t1c1", "cuda-f64-3d32t1c1"], procs=3, repeats=2,
                  gates="primary size/length/emittance gates unchanged; spread and CPU64 parity reported separately")
    (root / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    subprocess.run([sys.executable, "-u", str(HERE / "run.py"), "--case", "sc_drift", "--campaign", name,
                    "--codes", "slipstream", "--configs", ",".join(design["configs"]), "--procs", "3", "--repeats", "2", "--threads", "1"], check=True)
    original = common.RESULTS / a.campaign / "sc_drift"
    case = common.load(HERE / "cases" / "sc_drift.json")
    ref = np.load(original / "astra" / "cpu-f64-3d32h1.4" / "p0" / "tbt.npz")["tbt"]
    cpu = np.load(original / "slipstream" / "cpu-f64-3d32t1c1" / "p0" / "tbt.npz")["tbt"]
    rows = []
    for config in design["configs"]:
        jobs = []
        for p in sorted((root / "sc_drift" / "slipstream" / config).glob("p*/result.json")):
            r = common.load(p)
            if r["status"] == "ok":
                assert r["case_sha256"] == common.sha256(HERE / "cases" / "sc_drift.json")
                state = np.load(p.parent / "tbt.npz")["tbt"]
                jobs.append(dict(result=r, versus_astra={k: rel(state, ref, case["columns"], v) for k, v in OBS.items()},
                                 versus_cpu64={k: rel(state, cpu, case["columns"], v) for k, v in OBS.items()}))
            else:
                jobs.append(dict(result=r))
        good = [j for j in jobs if "versus_astra" in j]
        row = dict(config=config, jobs=jobs, n_ok=len(good))
        if good:
            row.update(warm_s=float(np.median([j["result"]["timing"]["warm_s"]["median"] for j in good])),
                       versus_astra={k: max(j["versus_astra"][k] for j in good) for k in OBS},
                       versus_cpu64={k: max(j["versus_cpu64"][k] for j in good) for k in OBS})
        rows.append(row)
    out = common.EVIDENCE / a.campaign / "gpu_control.json"
    out.write_text(json.dumps(dict(design=design, rows=rows), indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()

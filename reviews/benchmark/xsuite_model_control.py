"""Controlled correction of Xsuite model defaults; run after the primary campaign.

R1 explicitly set exact quadrupole splitting but left Drift/Sextupole adaptive.
Installed drift.h resolves adaptive to expanded when the legacy exact flag is
unset. Compare explicit exact drifts/sextupoles without changing R1 or its gates.
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import common
from analyse_grad import best_fd, errs

HERE = pathlib.Path(__file__).resolve().parent
DESIGN = {
    "ring_tunes": ["cpu-f64-yosk8b16-exact", "cpu-f64-yosk16b32-exact", "cuda-f64-yosk8b16-exact"],
    "ring_grad": ["cpu-f64-yosk8b16-exact"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="2026-10-26_R1")
    a = ap.parse_args()
    name = a.campaign + "_Xsuite_exact"
    root = common.RESULTS / name
    root.mkdir(exist_ok=True)
    design = dict(classification="exploratory adapter model correction; original R1 preserved", cases=DESIGN,
                  reference_campaign=a.campaign, procs=3, repeats=2, threads=1,
                  change="Force Drift exact and Sextupole drift-kick-drift-exact; other settings and error gates unchanged",
                  source_evidence="installed xtrack beam_elements/elements_src/drift.h: adaptive resolves to expanded unless XTRACK_USE_EXACT_DRIFTS is defined")
    (root / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    summaries = []
    for case_id, configs in DESIGN.items():
        subprocess.run([sys.executable, "-u", str(HERE / "run.py"), "--case", case_id, "--campaign", name,
                        "--codes", "xsuite", "--configs", ",".join(configs), "--procs", "3", "--repeats", "2", "--threads", "1", "--seed", "20261026"], check=True)
        case = common.load(HERE / "cases" / f"{case_id}.json")
        reference = common.RESULTS / a.campaign / case_id / "elegant" / "cpu-f64-o6s16b32" / "p0"
        reference_record = common.load(reference / "result.json")
        assert reference_record["status"] == "ok"
        if case_id == "ring_tunes":
            ref = np.load(reference / "tbt.npz")["tbt"]
            qx, qy = common.tunes(ref, case)
        else:
            zref = np.load(reference / "grad.npz")
            Jref, _ = best_fd(zref["J_fd"], zref["steps"])
        for config in configs:
            jobs = []
            for p in sorted((root / case_id / "xsuite" / config).glob("p*/result.json")):
                r = common.load(p)
                assert r.get("case_sha256", reference_record["case_sha256"]) == reference_record["case_sha256"]
                j = dict(result=r)
                if r["status"] == "ok":
                    if case_id == "ring_tunes":
                        state = np.load(p.parent / "tbt.npz")["tbt"]
                        x, y = common.tunes(state, case)
                        j.update(dq_ref=float(max(np.abs(x-qx).max(), np.abs(y-qy).max())),
                                 map_components=np.abs(state[:, 1] - ref[:, 1]).max(axis=0).tolist())
                    else:
                        z = np.load(p.parent / "grad.npz")
                        J, _ = best_fd(z["J_fd"], z["steps"])
                        j["rel_fro"], j["rel_col"] = errs(J, Jref)
                jobs.append(j)
            good = [j for j in jobs if j["result"]["status"] == "ok"]
            row = dict(case=case_id, config=config, jobs=jobs, n_ok=len(good))
            if good and case_id == "ring_tunes":
                row.update(warm_s=float(np.median([j["result"]["timing"]["warm_s"]["median"] for j in good])),
                           dq_ref=max(j["dq_ref"] for j in good), map_components=np.max([j["map_components"] for j in good], axis=0).tolist())
                row["pass"] = row["dq_ref"] <= case["tolerance"]["tune_abs"]
            elif good:
                row.update(fd_s=float(np.median([np.mean(list(j["result"]["fd_s"].values())) for j in good])),
                           rel_fro=max(j["rel_fro"] for j in good), rel_col=max(j["rel_col"] for j in good))
            summaries.append(row)
    out = common.EVIDENCE / a.campaign / "xsuite_model_control.json"
    out.write_text(json.dumps(dict(design=design, rows=summaries), indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()

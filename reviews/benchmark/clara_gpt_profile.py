"""Beam moments along the CLARA injector from the GPT 'profile' run -> data/2026-10-26_R1/clara_gpt_profile.json.

Input: slipstream_dev/Stories/CLARA_InjectorConvergence/output/gpt/100pC_profile_262144/out.gdf, written by
``python -m slipstream_dev.Stories.CLARA_InjectorConvergence.gpt_runs run 100pC --tag profile --particles 262144 --jobs 12``
(the default 100 pC GPT settings of Table III, snapshots every 0.2 ns, the last at the arrival time of the reference particle at z = 3.37147 m).
Per snapshot: mean z, rms size, and the normalised horizontal emittance from the canonical momentum p_x + q A_x with A_x = -B_z y / 2 (the
convention of the Slipstream trajectory and, inside the solenoids, the one that matches ASTRA) and from the kinetic momentum alone.
The last snapshot is compared with the end values of the earlier default run (same settings, snapshots only near the end).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import common

HERE = Path(__file__).resolve().parent
STORY = common.WORKBENCH / "slipstream_dev" / "Stories" / "CLARA_InjectorConvergence" / "output" / "gpt"
RUN = STORY / "100pC_profile_262144"
EARLIER = STORY / "100pC_default_262144.json"
MC2_EV = 510998.95
C = 299792458.0
ME, QE = 9.1093837015e-31, -1.602176634e-19


def emit(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(np.sqrt(max((a * a).mean() * (b * b).mean() - (a * b).mean() ** 2, 0.0)))


def moments(blk):
    p = {c["name"]: np.asarray(c["value"]) for c in blk["children"]}
    x, y, G, bz = p["x"], p["y"], p["G"], p["fBz"]
    ux, uy = p["Bx"] * G, p["By"] * G
    k = QE / (ME * C)
    ucx, ucy = ux + k * (-bz * y / 2), uy + k * (bz * x / 2)
    return dict(z_m=float(p["z"].mean()), t_ns=1e9 * float(blk["value"]), n=int(len(x)), sigma_x_mm=1e3 * float(x.std()), sigma_y_mm=1e3 * float(y.std()),
                emit_can_x_mm_mrad=1e6 * emit(x, ucx), emit_kin_x_mm_mrad=1e6 * emit(x, ux), emit_can_y_mm_mrad=1e6 * emit(y, ucy), Ekin_MeV=1e-6 * float(((G - 1) * MC2_EV).mean()))


def main():
    import easygdf

    blocks = [b for b in easygdf.load(str(RUN / "out.gdf"))["blocks"] if b["name"] in ("time", "position")]
    rows = sorted((moments(b) for b in blocks), key=lambda r: r["z_m"])
    earlier = json.loads(EARLIER.read_text())["end"]
    last = rows[-1]
    check = {k: dict(profile_last_snapshot=last[v], earlier_run_end=earlier[e], pct=100 * (last[v] / earlier[e] - 1))
             for k, v, e in (("eps_x", "emit_can_x_mm_mrad", "eps_x"), ("sigma_x", "sigma_x_mm", "sigma_x"), ("Ekin", "Ekin_MeV", "Ekin"))}
    out = dict(design=dict(run="100pC_profile_262144", settings="default GPT settings, 262144 macroparticles, snapshots every 0.2 ns", snapshots=len(rows), consistency_with_earlier_default_run=check,
                           run_in_sha256=hashlib.sha256((RUN / "run.in").read_bytes()).hexdigest(), analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), rows=rows)
    (common.EVIDENCE / "2026-10-26_R1" / "clara_gpt_profile.json").write_text(json.dumps(out, indent=1) + "\n")
    print(len(rows), "snapshots; z", round(rows[0]["z_m"], 3), "to", round(last["z_m"], 4))
    for k, v in check.items():
        print(f"  {k}: profile {v['profile_last_snapshot']:.5f}  earlier {v['earlier_run_end']:.5f}  ({v['pct']:+.2f}%)")


if __name__ == "__main__":
    main()

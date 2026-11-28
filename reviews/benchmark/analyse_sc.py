"""Summarise a station-table campaign (``sc_drift``): for every (code, config) the worst relative difference over the stations from the declared
reference's finest configuration, per observable, plus the change from the previous rung of the same code (convergence evidence) and the cost.

    python analyse_sc.py --campaign 2026-10-22_R1 --case sc_drift

Observables (columns of the station table): transverse size (x and y, worse of the two), bunch length, normalised emittance (worse of x and y),
mean kinetic energy (absolute, eV) and rms energy spread.  Differences are |a - b| / b at each station after the first (station 0 is the input).
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import phase_space  # noqa: E402

OBS = dict(size=("sigma_x_m", "sigma_y_m"), length=("sigma_z_m",), emit=("eps_nx_m", "eps_ny_m"), spread=("sigma_E_eV",))
PHASE = ("shape", "halo", "shape_z", "halo_z")   # distances of the final distributions from the reference's (phase_space.py)
SNAPSHOT_CODES = ("slipstream", "astra")        # their workers save one: a row of theirs without it is reported, not skipped


def solver_family(cfg):
    """Do not call switching from cylindrical to Cartesian (or analytic to PIC) refinement."""
    return tuple(cfg.get(k) for k in ("three_d", "geometry", "space_charge", "solver"))


def energy_gaps(table, ref, cols):
    i = cols.index("ekin_eV")
    return dict(dEkin_eV=float(np.abs(table[1:, i] - ref[1:, i]).max()),
                baseline_dEkin_eV=float(table[0, i] - ref[0, i]),
                dEkin_change_eV=float(np.abs((table[1:, i] - table[0, i]) - (ref[1:, i] - ref[0, i])).max()))


def rel(a, b, cols, names):
    i = [cols.index(n) for n in names]
    return float(np.max(np.abs(a[1:, i] - b[1:, i]) / np.abs(b[1:, i])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--case", default="sc_drift")
    a = ap.parse_args()
    root = common.RESULTS / a.campaign / a.case
    case = common.load(HERE / "cases" / f"{a.case}.json")
    cols = case["columns"]
    tol = case["tolerance"]
    runs = {}
    for res in sorted(root.glob("*/*/p*/result.json")):
        r = json.loads(res.read_text())
        runs.setdefault((r["code"], r["config"]["name"]), []).append((r, res.parent))
    rows, tabs, folders = [], {}, {}
    for (code, name), rs in runs.items():
        ok = [(r, d) for r, d in rs if r["status"] == "ok"]
        row = dict(code=code, config=name, rank=rs[0][0]["rank"], n_proc=len(rs), n_ok=len(ok),
                   solver_family=solver_family(rs[0][0]["config"]))
        if ok:
            folders[(code, name)] = ok[0][1]
            tabs[(code, name)] = np.load(ok[0][1] / "tbt.npz")["tbt"]
            med = lambda f: float(np.median([f(r) for r, _ in ok]))  # noqa: E731
            row.update(warm_s=med(lambda r: r["timing"]["warm_s"]["median"]) if ok[0][0]["timing"]["warm_s"] else None,
                       cold_s=med(lambda r: r["timing"]["cold_s"]), build_s=med(lambda r: r["timing"]["build_s"]), import_s=med(lambda r: r["timing"]["import_s"]),
                       rss_mb=med(lambda r: (r["memory"].get("rss_peak_bytes") or 0) / 2**20),
                       cuda_mb=med(lambda r: (r["memory"].get("cuda_peak_allocated", r["memory"].get("cuda_pool_total_bytes", 0))) / 2**20),
                       warnings="; ".join(sorted({w for r, _ in ok for w in r["warnings"]})))
        else:
            row["error"] = (rs[0][0].get("error") or "").strip().splitlines()[-1:] or "no result"
        rows.append(row)
    rows.sort(key=lambda r: (r["code"], r["rank"]))
    ref_code = case["reference"]["code"]
    cand = [r for r in rows if r["code"] == ref_code and (ref_code, r["config"]) in tabs]
    ref_row = max(cand, key=lambda r: r["rank"]) if cand else None
    ref = tabs[(ref_code, ref_row["config"])] if ref_row else None
    ref_snapshot = folders[(ref_code, ref_row["config"])] / "snapshot.npz" if ref_row else None
    for r in rows:
        t = tabs.get((r["code"], r["config"]))
        if t is None:
            continue
        prev = max((q for q in rows if q["code"] == r["code"] and q["rank"] < r["rank"] and (q["code"], q["config"]) in tabs and q["config"].split("-")[:2] == r["config"].split("-")[:2]
                    and q["solver_family"] == r["solver_family"]),
                   key=lambda q: q["rank"], default=None)
        for k, names in OBS.items():
            if ref is not None:
                r[f"{k}_vs_ref"] = rel(t, ref, cols, names)
            if prev:
                r[f"{k}_vs_prev"] = rel(t, tabs[(prev["code"], prev["config"])], cols, names)
        if ref is not None:
            r.update(energy_gaps(t, ref, cols))
            r["pass"] = r["size_vs_ref"] <= tol["sigma_rel"] and r["emit_vs_ref"] <= tol["eps_rel"] and r["length_vs_ref"] <= tol["sigma_rel"]
        if ref_snapshot is not None and ref_snapshot.exists():
            snapshot = folders[(r["code"], r["config"])] / "snapshot.npz"
            if snapshot.exists():
                found = phase_space.compare(np.load(snapshot), np.load(ref_snapshot), case["stations_m"][-1])
                r.update({f"{k}_vs_ref": found[k] for k in PHASE})
            elif r["code"] in SNAPSHOT_CODES:
                r.update({f"{k}_vs_ref": "missing" for k in PHASE})
    keys = ["code", "config", "rank", "n_proc", "n_ok", "warm_s", "cold_s", "build_s", "import_s", "rss_mb", "cuda_mb"] + \
           [f"{k}_vs_ref" for k in OBS] + [f"{k}_vs_prev" for k in OBS] + ["dEkin_eV", "baseline_dEkin_eV", "dEkin_change_eV", "pass", "warnings", "error"] + [f"{k}_vs_ref" for k in PHASE]
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    g = lambda v: f"{v:.3g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else ("-" if v is None else str(v))  # noqa: E731
    md = [f"# {case['title']}", "",
          (f"Reference: `{ref_code}/{ref_row['config']}`. " if ref_row else "No reference result. ") +
          f"Differences are the worst relative difference over stations 1.. ({len(case['stations_m']) - 1} stations); tolerances: size and length {tol['sigma_rel']:g}, emittance {tol['eps_rel']:g}. "
          "`prev` is the change within the same solver family, device and precision. Times are medians over independent processes. "
          "CSV energy columns separate the absolute mean-energy gap, entrance baseline offset and discrepancy in energy change, so a constant GPT offset is not read as an incorrect force.", "",
          "| code | config | warm [s] | cold [s] | size vs ref | length vs ref | emit vs ref | spread vs ref | dEkin [eV] | emit vs prev | shape vs ref | halo vs ref | pass | warnings |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['code']} | {r['config']} | {g(r.get('warm_s'))} | {g(r.get('cold_s'))} | {g(r.get('size_vs_ref'))} | {g(r.get('length_vs_ref'))} | {g(r.get('emit_vs_ref'))} | "
                  f"{g(r.get('spread_vs_ref'))} | {g(r.get('dEkin_eV'))} | {g(r.get('emit_vs_prev'))} | {g(r.get('shape_vs_ref'))} | {g(r.get('halo_vs_ref'))} | {r.get('pass', '-')} | {r.get('warnings') or r.get('error') or ''} |")
    md += ["", "## Cheapest passing configuration per code (by warm time)", ""]
    for c in sorted({r["code"] for r in rows}):
        okc = [r for r in rows if r["code"] == c and r.get("pass") and r.get("warm_s") is not None]
        best = min(okc, key=lambda r: r["warm_s"]) if okc else None
        md.append(f"- {c}: " + (f"`{best['config']}` at {best['warm_s']:.3g} s (size {best['size_vs_ref']:.2g}, emit {best['emit_vs_ref']:.2g})" if best else "no configuration passes"))
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

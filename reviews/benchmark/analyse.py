"""Summarise a campaign for the paper: accuracy against a declared reference, cost, and the cheapest passing configuration.

    python analyse.py --campaign 2026-10-22_R1 --case ring_tunes

Writes results/<campaign>/<case>/summary.csv, summary.md and pareto.png.  Accuracy uses the one estimator in common.py
for every code (never a code's own tune routine); first-turn and one-turn residuals are reported per component.
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

COMP = ("x", "px", "y", "py")


def load_runs(root):
    runs = {}
    for res in sorted(root.glob("*/*/p*/result.json")):
        r = json.loads(res.read_text())
        wall = json.loads((res.parent / "wall.json").read_text()) if (res.parent / "wall.json").exists() else {}
        r["wall_s"] = wall.get("wall_s")
        r["dir"] = res.parent
        runs.setdefault((r["code"], r["config"]["name"]), []).append(r)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--case", required=True)
    a = ap.parse_args()
    root = common.RESULTS / a.campaign / a.case
    case = common.load(HERE / "cases" / f"{a.case}.json")
    tol = case["tolerance"]
    runs = load_runs(root)

    rows, data = [], {}
    for (code, name), rs in runs.items():
        ok = [r for r in rs if r["status"] == "ok"]
        row = dict(code=code, config=name, rank=rs[0]["rank"], n_proc=len(rs), n_ok=len(ok),
                   model_family=rs[0]["config"].get("method"))
        if ok:
            z = np.load(ok[0]["dir"] / "tbt.npz")
            tbt, init = z["tbt"], z["init"]
            qx, qy = common.tunes(tbt, case)
            data[(code, name)] = dict(tbt=tbt, qx=qx, qy=qy)
            med = lambda f: float(np.median([f(r) for r in ok]))  # noqa: E731
            row.update(
                wall_s=med(lambda r: r["wall_s"]), import_s=med(lambda r: r["timing"]["import_s"]),
                build_s=med(lambda r: r["timing"]["build_s"]), cold_s=med(lambda r: r["timing"]["cold_s"]),
                warm_s=med(lambda r: r["timing"]["warm_s"]["median"]) if ok[0]["timing"]["warm_s"] else None,
                rss_mb=med(lambda r: (r["memory"].get("rss_peak_bytes") or 0) / 2**20),
                cuda_mb=med(lambda r: (r["memory"].get("cuda_peak_allocated", r["memory"].get("cuda_pool_total_bytes", 0))) / 2**20),
                init_roundtrip=float(np.abs(tbt[:, 0, :4] - init[:, :4]).max()),
                qx=qx.tolist(), qy=qy.tolist(),
                warnings="; ".join(sorted({w for r in ok for w in r["warnings"]})),
                versions=json.dumps(ok[0].get("versions")))
        else:
            row["error"] = (rs[0].get("error") or "").strip().splitlines()[-1:] or "no result"
        rows.append(row)
    rows.sort(key=lambda r: (r["code"], r["rank"]))

    # A failed reference must not silently become a consensus of the other codes.
    ref_code = case["reference"]["code"]
    finest = {c: max((r for r in rows if r["code"] == c and (c, r["config"]) in data), key=lambda r: r["rank"], default=None)
              for c in {r["code"] for r in rows}}
    if finest.get(ref_code):
        ref = data[(ref_code, finest[ref_code]["config"])]
        ref_name = f"{ref_code}/{finest[ref_code]['config']}"
    else:
        raise RuntimeError(f"No successful declared reference: {ref_code}; no ring comparison published")
    for r in rows:
        d = data.get((r["code"], r["config"]))
        if not d:
            continue
        own = data[(r["code"], finest[r["code"]]["config"])]
        r["dq_ref"] = float(max(np.abs(d["qx"] - ref["qx"]).max(), np.abs(d["qy"] - ref["qy"]).max()))
        r["dq_self"] = float(max(np.abs(d["qx"] - own["qx"]).max(), np.abs(d["qy"] - own["qy"]).max()))
        prev = max((q for q in rows if q["code"] == r["code"] and q["rank"] < r["rank"] and (q["code"], q["config"]) in data
                    and q["config"].split("-")[0] == r["config"].split("-")[0] and q["config"].split("-")[1] == r["config"].split("-")[1]
                    and q["model_family"] == r["model_family"]),
                   key=lambda q: q["rank"], default=None)  # previous rung on the same device and dtype: convergence evidence
        if prev:
            pd = data[(prev["code"], prev["config"])]
            r["dq_prev"] = float(max(np.abs(d["qx"] - pd["qx"]).max(), np.abs(d["qy"] - pd["qy"]).max()))
        for i, c in enumerate(COMP):  # separate components: positions and momenta are never mixed into one number
            r[f"onturn_{c}"] = float(np.abs(d["tbt"][:, 1, i] - ref["tbt"][:, 1, i]).max())
        r["pass"] = r["dq_ref"] <= tol["tune_abs"]

    keys = ["code", "config", "rank", "n_proc", "n_ok", "wall_s", "import_s", "build_s", "cold_s", "warm_s", "rss_mb", "cuda_mb",
            "dq_ref", "dq_self", "dq_prev", "onturn_x", "onturn_px", "onturn_y", "onturn_py", "init_roundtrip", "pass", "warnings", "error",
            "qx", "qy", "versions"]
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    g = lambda v, f="{:.3g}": f.format(v) if isinstance(v, (int, float)) and v is not None else "-"  # noqa: E731
    md = [f"# {case['title']}", "", f"Reference: `{ref_name}`. Tune tolerance {tol['tune_abs']:g} (absolute, both planes). "
          "Times are medians over independent processes; `warm` is per call of the full 1024-turn tracking.", "",
          "| code | config | warm [s] | cold [s] | wall [s] | max abs dq vs ref | dq vs own finest | dq vs previous rung | pass | warnings |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['code']} | {r['config']} | {g(r.get('warm_s'))} | {g(r.get('cold_s'))} | {g(r.get('wall_s'))} | "
                  f"{g(r.get('dq_ref'))} | {g(r.get('dq_self'))} | {g(r.get('dq_prev'))} | {r.get('pass', '-')} | {r.get('warnings') or r.get('error') or ''} |")
    md += ["", "## Cheapest passing configuration per code (by warm time)", ""]
    for c in sorted(finest):
        ok = [r for r in rows if r["code"] == c and r.get("pass") and r.get("warm_s") is not None]
        best = min(ok, key=lambda r: r["warm_s"]) if ok else None
        md.append(f"- {c}: " + (f"`{best['config']}` at {best['warm_s']:.3g} s (dq {best['dq_ref']:.2g})" if best else "no configuration passes"))
    (root / "summary.md").write_text("\n".join(md) + "\n")

    try:  # the plot is optional: a second PC needs numpy only
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in sorted(finest):
            pts = [(r["warm_s"], max(r["dq_ref"], 1e-12)) for r in rows if r["code"] == c and r.get("warm_s") and "dq_ref" in r]
            if pts:
                ax.plot(*zip(*pts), "o", label=c)
        ax.axhline(tol["tune_abs"], color="k", ls="--", lw=0.8)
        ax.set(xscale="log", yscale="log", xlabel="warm time per run [s]", ylabel="max |tune error| vs reference",
               title=f"{a.case}: error against cost")
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "pareto.png", dpi=150)
    except ImportError:
        pass
    print("\n".join(md))


if __name__ == "__main__":
    main()

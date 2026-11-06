"""Summarise a ``ring_grad`` campaign: Jacobian error against a declared reference, AD against the code's own central difference,
and the cost of one Jacobian by finite differences (2K tracks, rebuilt each time) against one by the code's own AD.

    python analyse_grad.py --campaign 2026-10-22_R1 --case ring_grad

y is the state at the start of turn ``turns - 1`` for the single launch in the case; theta are the family strengths in the case.
Errors are reported two ways because a norm over the whole matrix hides small columns: ``rel_fro`` (Frobenius, whole matrix) and
``rel_col`` (worst column, 2-norm of the column error over the reference column norm).
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


def best_fd(J, steps):
    """Best finite-difference Jacobian: Richardson extrapolation of consecutive central differences (steps 10x apart, truncation
    error ~ h^2, so R = (100 J(h/10) - J(h)) / 99), choosing the pair whose own estimate changes least. Returns (J, label)."""
    R = [(100 * J[i + 1] - J[i]) / 99 for i in range(len(J) - 1)]
    d = [np.abs(R[i] - R[i + 1]).max() for i in range(len(R) - 1)]
    d.append(d[-1])
    d = np.array(d)
    i = int(np.flatnonzero(d == d.min())[-1])  # last minimum: the smaller step wins a tie
    return R[i], f"R({steps[i]:g},{steps[i + 1]:g})"


def errs(J, Jref):
    col = np.linalg.norm(J - Jref, axis=0) / np.linalg.norm(Jref, axis=0)
    return float(np.linalg.norm(J - Jref) / np.linalg.norm(Jref)), float(col.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--case", default="ring_grad")
    a = ap.parse_args()
    root = common.RESULTS / a.campaign / a.case
    case = common.load(HERE / "cases" / f"{a.case}.json")
    rows = []
    for res in sorted(root.glob("*/*/p*/result.json")):
        r = json.loads(res.read_text())
        row = dict(code=r["code"], config=r["config"]["name"], rank=r["rank"], status=r["status"])
        if r["status"] == "ok" and (res.parent / "grad.npz").exists():
            z = np.load(res.parent / "grad.npz")
            Jb, label = best_fd(z["J_fd"], z["steps"])
            row.update(J_fd=Jb, y0=z["y0"], h_best=label, forward_s=r["forward_s"], fd_s=float(np.mean(list(r["fd_s"].values()))))
            if "J_ad" in z:
                row.update(J_ad=z["J_ad"], ad_warm_s=r["ad_warm_s"]["median"], ad_cold_s=r["ad_cold_s"])
                row["ad_vs_fd_fro"], row["ad_vs_fd_col"] = errs(z["J_ad"], row["J_fd"])
        rows.append(row)
    ok = [r for r in rows if "J_fd" in r]
    ref_code = case["reference"]["code"]
    cand = [r for r in ok if r["code"] == ref_code]
    ref = max(cand, key=lambda r: r["rank"]) if cand else None
    for r in ok:
        J = r.get("J_ad", r["J_fd"])
        if ref is not None:
            r["rel_fro"], r["rel_col"] = errs(J, ref["J_fd"])
            r["dy0"] = float(np.abs(r["y0"] - ref["y0"]).max())
    keys = ["code", "config", "status", "h_best", "forward_s", "fd_s", "ad_cold_s", "ad_warm_s", "ad_vs_fd_fro", "ad_vs_fd_col", "rel_fro", "rel_col", "dy0"]
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    g = lambda v: f"{v:.3g}" if isinstance(v, (int, float)) else "-"  # noqa: E731
    md = [f"# {case['title']}", "",
          f"Reference: `{ref['code']}/{ref['config']}` central difference, {ref['h_best']}. "
          f"Endpoint after {case['turns'] - 1} completed turns (the trajectory adapter executes {case['turns']}), {len(case['knobs'])} knobs: " + ", ".join(f"{k['family']}.{k['kind']}" for k in case["knobs"]) + ". "
          "`ad/fd` compares the code's own AD Jacobian with its own best central difference; `vs ref` compares the AD (else FD) Jacobian with the reference. "
          "`dy0` is the largest component difference of the tracked state itself, so a large `vs ref` with a large `dy0` is a trajectory (tune) disagreement, not a derivative error.", "",
          "| code | config | forward [s] | FD Jacobian [s] | AD cold [s] | AD warm [s] | FD/AD time | ad/fd (fro, worst col) | vs ref (fro, worst col) | dy0 |",
          "|---|---|---|---|---|---|---|---|---|---|"] if ref else []
    for r in ok:
        ratio = r["fd_s"] / r["ad_warm_s"] if "ad_warm_s" in r else None
        md.append(f"| {r['code']} | {r['config']} | {g(r['forward_s'])} | {g(r['fd_s'])} | {g(r.get('ad_cold_s'))} | {g(r.get('ad_warm_s'))} | {g(ratio)} | "
                  f"{g(r.get('ad_vs_fd_fro'))}, {g(r.get('ad_vs_fd_col'))} | {g(r.get('rel_fro'))}, {g(r.get('rel_col'))} | {g(r.get('dy0'))} |")
    for r in rows:
        if r["status"] != "ok":
            md.append(f"| {r['code']} | {r['config']} | failed | | | | | | | |")
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

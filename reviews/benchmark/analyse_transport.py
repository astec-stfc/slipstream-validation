"""Compare tiny basis launches against independent thick-quadrupole matrices."""
import argparse
import csv
import json
import pathlib

import numpy as np
import common

HERE = pathlib.Path(__file__).resolve().parent


def reference(lat):
    M = np.eye(4)
    for e in lat["elements"]:
        A = np.eye(4)
        for i, sign in [(0, 1), (2, -1)]:
            k, L = sign * e.get("k1", 0), e["length"]
            if k == 0:
                B = [[1, L], [0, 1]]
            else:
                q = np.sqrt(abs(k))
                c = np.cos(q * L) if k > 0 else np.cosh(q * L)
                s = (np.sin(q * L) if k > 0 else np.sinh(q * L)) / q
                B = [[c, s], [-k * s, c]]
            A[i:i+2, i:i+2] = B
        M = A @ M
    assert np.allclose(np.linalg.det(M), 1, atol=1e-12)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--case", default="linear_fodo")
    a = ap.parse_args()
    case = common.load(HERE / "cases" / f"{a.case}.json")
    M = reference(common.load(HERE / "cases" / case["lattice"]))
    expected = common.launches(case)[:, :4] @ M.T
    root = common.RESULTS / a.campaign / a.case
    rows = []
    for p in sorted(root.glob("*/*/p*/result.json")):
        r = common.load(p)
        row = dict(code=r["code"], config=r["config"]["name"], status=r["status"])
        if r["status"] == "ok":
            state = np.load(p.parent / "tbt.npz")["tbt"][:, 1]
            err = np.abs(state - expected)
            row.update(position_abs_m=float(err[:, [0, 2]].max()), momentum_abs=float(err[:, [1, 3]].max()),
                       warm_s=r["timing"]["warm_s"]["median"])
            row["pass"] = row["position_abs_m"] <= case["tolerance"]["position_abs_m"] and row["momentum_abs"] <= case["tolerance"]["momentum_abs"]
        rows.append(row)
    with (root / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["code", "config", "status", "position_abs_m", "momentum_abs", "warm_s", "pass"])
        w.writeheader()
        w.writerows(rows)
    (root / "summary.md").write_text("# Linear FODO control\n\nIndependent paraxial matrix; position and momentum errors have separate units.\n\n" + "\n".join(str(r) for r in rows) + "\n")
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

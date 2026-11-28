"""Read a campaign: analyse it with the review's own script, then compare it with the archived evidence of the paper.

    python -m slipstream_validation.check compare pc1                      # every case in the campaign
    python -m slipstream_validation.check compare pc1 --case ring_grad     # one case

A metric agrees when it moved by no more than REL of the archived value or the case's floor, whichever is larger.  That is a ratchet,
not a target: it comes from one host pair (this PC, native GPU or WSL against the image) and is to be re-measured from a second host.
A case may also have absolute limits, from its json (ring_grad: AD against its own central difference, `jac_rel`) or given in CASES.  Wall times are
printed beside each other and never gated, because they do not reproduce between hosts.  A difference that has an explanation goes
into KNOWN with the reason; it is then XFAIL, not hidden.
"""
import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "reviews" / "benchmark"
EVIDENCE = ROOT / "reviews" / "claims" / "evidence" / "2026-10-26_R1"
REL = 0.25

# What is compared per case: the metrics (two-sided ratchet), the floor below which a difference is numerical noise, flags that must
# be equal, the timing column that is only printed, and absolute limits {metric: key of the case's `tolerance`}.
# Rung-to-rung columns (`*_vs_prev`) are left out: they depend on which rungs a campaign ran.
CASES = {
    "ring_tunes": dict(metrics=["dq_ref"], floor=1e-9, time="warm_s"),
    "ring_grad": dict(metrics=["ad_vs_fd_fro", "ad_vs_fd_col", "rel_fro", "rel_col", "dy0"], floor=1e-9, time="fd_s",
                      limit={"ad_vs_fd_fro": "jac_rel"}),
    "sc_drift": dict(metrics=["size_vs_ref", "length_vs_ref", "emit_vs_ref", "spread_vs_ref", "dEkin_change_eV"], floor=1e-9,
                     flags=["pass"], time="warm_s",
                     # ratchets on the final distributions against the reference's (phase_space.py): about twice the largest value of the
                     # ASTRA and Slipstream CPU rungs on 28 November 2026.  No archived campaign has them; a campaign gates itself.
                     limit={"shape_vs_ref": 0.011, "halo_vs_ref": 0.016, "shape_z_vs_ref": 0.015, "halo_z_vs_ref": 0.073}),
    "linear_fodo": dict(metrics=["position_abs_m", "momentum_abs"], floor=1e-15, flags=["pass"], time="warm_s"),
}
# A campaign is compared only if it ran one of these rungs: the analysis measures every other row against the reference code's finest rung
# that the campaign has, so without it a row would be compared with something other than the archived reference (or with itself).
REFERENCE = {"ring_tunes": [("elegant", "cpu-f64-o6s16b32")], "ring_grad": [("elegant", "cpu-f64-o6s16b32")],
             "sc_drift": [("astra", "cpu-f64-3d32h1.4"), ("astra", "cpu-f64-3d64h1.4")]}
ANALYSIS = {"ring_tunes": "analyse.py", "ring_grad": "analyse_grad.py", "sc_drift": "analyse_sc.py", "linear_fodo": "analyse_transport.py"}

# the image has no GPU in a plain run: the same model on the CPU is compared with the archived GPU row of the same name
ALIASES = {("slipstream", "cpu-f64-o6q8s1b8"): ("slipstream", "cuda-f64-o6q8s1b8"),        # the paper's ring setting (Slipstream 1.1.0)
           ("slipstream", "cpu-f64-o6k8b16"): ("slipstream", "cuda-f64-o6k8b16"),
           ("slipstream", "cpu-f64-o6k16b32"): ("slipstream", "cuda-f64-o6k16b32"),
           ("slipstream", "cpu-f64-r32z128t2c5"): ("slipstream", "cuda-f32-r32z128t2c5"),     # f64 on the CPU against f32 on the GPU
           ("slipstream", "cpu-f64-r64z256t1c2"): ("slipstream", "cuda-f32-r64z256t1c2")}
# image jobs that have no archived row on purpose (a rung added after the archived campaign)
NOREF_OK = {("sc_drift", "astra", "cpu-f64-3d64h1.4"),
            ("sc_drift", "opalx", "cpu-f64-open32h40"), ("sc_drift", "opalx", "cpu-f64-open64h40")}   # OPAL-X: added 10 November 2026, after the archived campaign

_ELEGANT_BUILD = ("the image's elegant 2026.1.0 gives a finite-difference Jacobian 7.6e-9 (absolute) away from the paper's 2026.3.0 (endpoint state "
                  "6e-14); the worst-column error of Slipstream is 4.7e-5 against 2026.3.0 and 2.1e-5 against 2026.1.0, the difference of two "
                  "numbers that small; native 2026.3.0 with this worker reproduces the archived row exactly")
KNOWN: dict = {   # (case, code, config, metric) -> reason
    ("ring_grad", "slipstream", "cpu-f64-o6q8s1b8", "rel_col"): _ELEGANT_BUILD,
    # SciBmad: same reference shift (rel_col falls 2.2x like Slipstream's); its rel_fro and ad_vs_fd_col are equal to the archive
    ("ring_grad", "scibmad", "cpu-f64-o4k4", "rel_col"): _ELEGANT_BUILD,
    ("ring_grad", "scibmad", "cpu-f64-o6k32", "rel_col"): _ELEGANT_BUILD,
}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def archived_ring_tunes() -> dict:
    d = json.loads((EVIDENCE / "corrected_reference.json").read_text())
    out = {(r["code"], r["config"]): dict(dq_ref=r["dq_ref"], warm_s=r.get("warm_s")) for r in d["ring_tunes"]}
    for r in d["xsuite"]["rows"]:
        if r["case"] == "ring_tunes":
            out[("xsuite", r["config"])] = dict(dq_ref=r["dq_ref"], warm_s=r.get("warm_s"))
    cpu = {r["case"]: r for r in d["cpu"]["rows"]}["ring_tunes"]
    out[("slipstream", d["cpu"]["design"]["config"])] = dict(dq_ref=cpu["dq_ref"], warm_s=cpu.get("warm_s"))
    return out


def archived_csv(case: str) -> dict:
    """The archived summary of the case: one row per (code, config), the first of the three processes (accuracy is the same in all)."""
    out = {}
    for r in csv.DictReader((EVIDENCE / f"{case}.csv").open()):
        out.setdefault((r["code"], r["config"]), r)
    return out


def archived_ring_grad() -> dict:
    """The corrected `ring_grad` table (campaign 2026-10-27_momentum: elegant p_central matches input p/(mc)).  The CSV of 2026-10-26_R1
    holds the superseded elegant reference and is not used."""
    out = {}
    for r in json.loads((EVIDENCE / "corrected_reference.json").read_text())["ring_grad"]:
        out.setdefault((r["code"], r["config"]), r)
    return out


def archived(case: str) -> dict:
    return {"ring_tunes": archived_ring_tunes, "ring_grad": archived_ring_grad}.get(case, lambda: archived_csv(case))()


def judge(case, code, cfg, x, o, limits):
    """Status of one new summary row x against the archived row o (None: no archive), the moved metrics and the limit breaches."""
    spec = CASES[case]
    moved, why = [], []   # (metric, archived, new); text
    failed = x.get("status", "ok") != "ok" or bool(x.get("error")) or (
        o is not None and any(num(x.get(m)) is None and num(o.get(m)) is not None for m in spec["metrics"]))
    for m in spec["metrics"]:
        n, a = num(x.get(m)), num(o.get(m)) if o else None
        if n is not None and a is not None and abs(n - a) > max(REL * abs(a), spec["floor"]):
            moved.append((m, a, n))
    for m in spec.get("flags", []):
        if o is not None and str(x.get(m)) != str(o.get(m)):
            moved.append((m, o.get(m), x.get(m)))
    for m, key in spec.get("limit", {}).items():
        n = num(x.get(m))
        cap = key if isinstance(key, float) else limits.get(key)
        if x.get(m) == "missing":
            why.append(f"{m}: the worker wrote no particle snapshot")
        elif n is not None and cap is not None and n > cap:
            why.append(f"{m} {n:.3g} above the limit {cap:g}" + ("" if isinstance(key, float) else f" ({key})"))
    if failed:
        status = "FAILED"
    elif why:
        status = "LIMIT"
    elif moved and all((case, code, cfg, m) in KNOWN for m, _, _ in moved):
        status = "XFAIL"
    elif moved:
        status = "MOVED"
    else:
        status = "ok" if o is not None else "new" if (case, code, cfg) in NOREF_OK else "NOREF"
    return status, moved, why


def ran(results: pathlib.Path, tag: str, case: str) -> set:
    """(code, config) of every job of the campaign that finished ok."""
    out = set()
    for f in (results / tag / case).glob("*/*/p*/result.json"):
        if json.loads(f.read_text()).get("status") == "ok":
            out.add((f.parents[2].name, f.parents[1].name))
    return out


def compare(tag: str, case: str, results: pathlib.Path) -> int:
    spec = CASES[case]
    need = REFERENCE.get(case)
    if need and not any(k in ran(results, tag, case) for k in need):
        print(f"\n{case}: skipped, the campaign has no reference rung ({' or '.join(c + '/' + n for c, n in need)}); rerun with it to compare")
        return 0
    r = subprocess.run([sys.executable, str(BENCH / ANALYSIS[case]), "--campaign", tag, "--case", case], capture_output=True, text=True,
                       env={**os.environ, "SSV_RESULTS": str(results), "PYTHONDONTWRITEBYTECODE": "1"})
    if r.returncode:
        print(f"{case}: the analysis script failed\n{r.stderr[-800:]}")
        return 1
    new = {(x["code"], x["config"]): x for x in csv.DictReader((results / tag / case / "summary.csv").open())}
    old = archived(case)
    limits = json.loads((BENCH / "cases" / f"{case}.json").read_text()).get("tolerance", {})
    env = json.loads((results / tag / "environment.json").read_text()) if (results / tag / "environment.json").exists() else {}
    print(f"\n{case}  campaign {tag}  image {str(env.get('image_id'))[:19]}  slipstream {str(env.get('slipstream', {}).get('commit'))[:12]}")
    print(f"{'code':10} {'config':26} {'status':7} {'largest change':44} {'time archived':>13} {'time new':>9}")
    bad = 0
    for (code, cfg), x in sorted(new.items()):
        o = old.get((code, cfg)) or old.get(ALIASES.get((code, cfg), ()))   # an alias only where the archive has no row of that name
        status, moved, why = judge(case, code, cfg, x, o, limits)
        bad += status in ("FAILED", "LIMIT", "MOVED")
        change = max(((abs(num(x[m]) - num(o[m])) / (abs(num(o[m])) or 1), m, num(o[m]), num(x[m])) for m in spec["metrics"]
                      if o and num(x.get(m)) is not None and num(o.get(m)) is not None), default=None)
        text = f"{change[1]} {change[2]:.3e} -> {change[3]:.3e}" if change else "-"
        if o is not None and (code, cfg) not in old:
            text += f"  [vs {ALIASES[(code, cfg)][1]}]"
        t_old, t_new = (num(o.get(spec["time"])) if o else None), num(x.get(spec["time"]))
        g = lambda v: "-" if v is None else format(v, ".3g")  # noqa: E731
        print(f"{code:10} {cfg:26} {status:7} {text:44} {g(t_old):>13} {g(t_new):>9}")
        for m, a, n in moved:
            print(f"    {m}: archived {a} -> new {n}")
        for reason in dict.fromkeys(KNOWN[(case, code, cfg, m)] for m, _, _ in moved if (case, code, cfg, m) in KNOWN):
            print(f"    known: {reason}")
        for w in why:
            print(f"    {w}")
    only = sorted({c for c, _ in old} - {c for c, _ in new})
    print(f"archived, not rerun here: {', '.join(only) or 'none'}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("tag")
    c.add_argument("--case", action="append", choices=list(CASES), help="default: every case that has results in the campaign")
    c.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    res = pathlib.Path(a.results)
    cases = a.case or [k for k in CASES if (res / a.tag / k).is_dir()]
    if not cases:
        sys.exit(f"no case folder in {res / a.tag}")
    bad = sum(compare(a.tag, k, res) for k in cases)
    print("\n" + (f"{bad} row(s) need attention" if bad else "all rows agree with the archive") +
          f"; tolerance max({REL:g} * archived, floor of the case); times are not gated")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

"""Read a campaign: analyse it with the review's own script, then compare it with the archived evidence of the paper.

    python -m slipstream_validation.check compare pc1 --case ring_tunes

A configuration agrees when its error against the case's reference moved by no more than REL of the archived value or ABS, whichever
is larger.  That is a ratchet, not a target: it comes from one host pair (this PC, native GPU or WSL against the image) and is to be
re-measured from a second host.  Wall times are printed beside each other and never gated, because they do not reproduce between hosts.
A difference that has an explanation goes into KNOWN with the reason (and an issue number when there is one); it is then XFAIL, not hidden.
"""
import argparse
import csv
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "reviews" / "benchmark"
EVIDENCE = ROOT / "reviews" / "claims" / "evidence" / "2026-10-26_R1"
REL, ABS = 0.25, 1e-9

# the image has no GPU in a plain run: the same model on the CPU is compared with the archived GPU row of the same name
ALIASES = {("slipstream", "cpu-f64-o6k8b16"): ("slipstream", "cuda-f64-o6k8b16"),
           ("slipstream", "cpu-f64-o6k16b32"): ("slipstream", "cuda-f64-o6k16b32")}
KNOWN: dict = {}   # (case, code, config) -> reason

ANALYSIS = {"ring_tunes": "analyse.py", "ring_grad": "analyse_grad.py", "sc_drift": "analyse_sc.py", "linear_fodo": "analyse_transport.py"}


def archived_ring_tunes() -> dict:
    d = json.loads((EVIDENCE / "corrected_reference.json").read_text())
    out = {(r["code"], r["config"]): dict(dq_ref=r["dq_ref"], warm_s=r.get("warm_s")) for r in d["ring_tunes"]}
    for r in d["xsuite"]["rows"]:
        if r["case"] == "ring_tunes":
            out[("xsuite", r["config"])] = dict(dq_ref=r["dq_ref"], warm_s=r.get("warm_s"))
    cpu = {r["case"]: r for r in d["cpu"]["rows"]}["ring_tunes"]
    out[("slipstream", "cpu-f64-o6k8b16-compiled")] = dict(dq_ref=cpu["dq_ref"], warm_s=cpu.get("warm_s"))
    return out


ARCHIVED = {"ring_tunes": archived_ring_tunes}


def compare(tag: str, case: str, results: pathlib.Path) -> int:
    if case not in ARCHIVED:
        sys.exit(f"no archived evidence is wired for {case} yet (ring_tunes only)")
    subprocess.run([sys.executable, str(BENCH / ANALYSIS[case]), "--campaign", tag, "--case", case],
                   check=True, capture_output=True, env={**__import__("os").environ, "SSV_RESULTS": str(results), "PYTHONDONTWRITEBYTECODE": "1"})
    new = {(r["code"], r["config"]): r for r in csv.DictReader((results / tag / case / "summary.csv").open())}
    old = ARCHIVED[case]()
    env = json.loads((results / tag / "environment.json").read_text()) if (results / tag / "environment.json").exists() else {}
    print(f"{tag}: image {str(env.get('image_id'))[:19]}  slipstream {str(env.get('slipstream', {}).get('commit'))[:12]}  case {case}")
    print(f"{'code':10} {'config':26} {'archived dq':>12} {'new dq':>12} {'status':8} {'warm archived':>14} {'warm new':>9}")
    bad = 0
    for (code, cfg), r in sorted(new.items()):
        key = ALIASES.get((code, cfg), (code, cfg))
        o = old.get(key)
        if not r.get("dq_ref") or o is None:
            status = "NOREF" if o is None else "FAILED"
            bad += status == "FAILED"
            print(f"{code:10} {cfg:26} {'-':>12} {'-':>12} {status:8}")
            continue
        n, a = float(r["dq_ref"]), float(o["dq_ref"])
        ok = abs(n - a) <= max(REL * abs(a), ABS)
        status = "ok" if ok else ("XFAIL" if (case, code, cfg) in KNOWN else "MOVED")
        bad += status == "MOVED"
        w_old = f"{float(o['warm_s']):.3g}" if o.get("warm_s") not in (None, "") else "-"
        w_new = f"{float(r['warm_s']):.3g}" if r.get("warm_s") else "-"
        print(f"{code:10} {cfg:26} {a:12.3e} {n:12.3e} {status:8} {w_old:>14} {w_new:>9}" + (f"   ({KNOWN[(case, code, cfg)]})" if status == "XFAIL" else ""))
    print(f"tolerance: max({REL:g} * archived, {ABS:g}) in absolute tune error; times are not gated")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("tag")
    c.add_argument("--case", default="ring_tunes")
    c.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    sys.exit(compare(a.tag, a.case, pathlib.Path(a.results)))


if __name__ == "__main__":
    main()

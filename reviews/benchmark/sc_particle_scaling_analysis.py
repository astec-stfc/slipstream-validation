"""Particle-count scaling of the space-charge drift for every code that has a worker: call time of the 3 m drift against the number of macroparticles.

The bunch is the one of the paper's ladder (injector_scaling_control.py: 2 000, 200 000 and 1 000 000 macroparticles, fresh frozen sets; 20 000 is the original
sc_drift case).  Slipstream and ASTRA come from the paper's own evidence (``injector_scaling_control.json``, so the numbers are those of the paper); every other
code is read from its campaign.  A point is the median over processes of the warm call time (the cold time when a configuration had no warm repeats), and
``n_proc`` is the number of processes behind it (1 for the slowest points, as in the paper).

    python sc_particle_scaling_analysis.py            # writes reviews/claims/evidence/2026-11-10_opalx/sc_particle_scaling.json
"""
import json
import re
import statistics
from pathlib import Path

import common

EVIDENCE = common.EVIDENCE
OUT = EVIDENCE / "2026-11-10_opalx" / "sc_particle_scaling.json"
CAMPAIGNS = ["2026-10-26_R1", "2026-11-10_opalx", "2026-11-10_opalx_scaling", "2026-11-10_sc_ladder"]   # a later campaign overrides an earlier one for the same point
# (Cheetah's float32 GPU rung cuda-f32-g32s2 was run but is left out: it keeps the energy spread frozen, 19 % off ASTRA at 3 m, see codes/CODES.md)
# label -> (code, [config names]) for the codes read from campaigns; mesh class "32"/"64"/"128" (or "" when the code has one setting), device "gpu"/"cpu"
CURVES = {
    "OPAL-X GPU f64, 32^3": ("opalx", ["cuda-f64-open32h40"], "32", "gpu"),
    "OPAL-X GPU f64, 64^3": ("opalx", ["cuda-f64-open64h40"], "64", "gpu"),
    "OPAL-X GPU f64, 128^3": ("opalx", ["cuda-f64-open128h40"], "128", "gpu"),
    "Cheetah GPU f64, 32^3": ("cheetah", ["cuda-f64-g32s2"], "32", "gpu"),
    "Cheetah CPU f64, 32^3": ("cheetah", ["cpu-f64-g32s2"], "32", "cpu"),
    "Cheetah CPU f64, 64^3": ("cheetah", ["cpu-f64-g64s4"], "64", "cpu"),
    "Ocelot CPU f64, 32^3": ("ocelot", ["cpu-f64-g32s2"], "32", "cpu"),
    "Ocelot CPU f64, 64^3": ("ocelot", ["cpu-f64-g64s4"], "64", "cpu"),
    "ImpactX CPU f64, 32x32x40": ("impactx", ["cpu-f64-mg32x40s2"], "32", "cpu"),
    "ImpactX CPU f64, 64x64x80": ("impactx", ["cpu-f64-mg64x80s4"], "64", "cpu"),
    "GPT CPU f64, mesh solver": ("gpt", ["cpu-f64-mesh-acc5"], "", "cpu"),
}


def particles_of(case_dir: str):
    """Macroparticle number of a case folder: ``sc_drift`` is the 20 000 original, ``sc_drift_n<N>`` the ladder; seed-control copies (``..s1``) are skipped."""
    if case_dir == "sc_drift":
        return 20000
    m = re.fullmatch(r"sc_drift_n(\d+)", case_dir)
    return int(m.group(1)) if m else None


def collect(code, configs):
    """{N: (seconds, n_proc, config)} from every campaign (a later campaign overrides an earlier one for the same N)."""
    out = {}
    for tag in CAMPAIGNS:
        for case_dir in sorted((common.RESULTS / tag).glob("sc_drift*")):
            n = particles_of(case_dir.name)
            if n is None:
                continue
            for cfg in configs:
                times = []
                for f in sorted((case_dir / code / cfg).glob("p*/result.json")):
                    r = json.loads(f.read_text())
                    if r.get("status") != "ok":
                        continue
                    t = r["timing"]
                    times.append(t["warm_s"]["median"] if t.get("warm_s") else t["cold_s"])
                if times:
                    out[n] = (statistics.median(times), len(times), cfg)
    return out


def main():
    rows = []
    for label, (code, configs, mesh, device) in CURVES.items():
        for n, (t, k, cfg) in sorted(collect(code, configs).items()):
            rows.append(dict(label=label, code=code, config=cfg, mesh=mesh, device=device, n=n, warm_s=t, n_proc=k, us_per_particle=t / n * 1e6))
    # Slipstream (GPU f32, mass-fixed ASTRA-deck rungs) and ASTRA (3-D, 1.4 ps) from the paper's evidence
    ij = json.loads((EVIDENCE / "2026-10-26_R1" / "injector_scaling_control.json").read_text())["rows"]
    for code, name, dev, key in (("slipstream", "Slipstream GPU f32, {m}^3", "gpu", "slipstream{m}"), ("astra", "ASTRA CPU f64, {m}^3", "cpu", "astra{m}")):
        for m in ("32", "64", "128"):
            for r in ij:
                k = key.format(m=m)
                if k in r["warm_s"]:
                    t, n_proc = r["warm_s"][k], r["n_ok"][k]
                elif k in r["cold_s"]:
                    t, n_proc = r["cold_s"][k], r["n_ok"].get(k, 1)
                else:
                    continue
                rows.append(dict(label=name.format(m=m), code=code, config=("cuda-f32-3d%st1c10-r100-mfix" if code == "slipstream" else "cpu-f64-3d%sh1.4") % m, mesh=m,
                                 device=dev, n=r["n"], warm_s=t, n_proc=n_proc, us_per_particle=t / r["n"] * 1e6))
    for r in ij:   # Slipstream on the CPU (3-D 32^3, float64), measured up to 20 000 particles only
        if "slipstream_cpu32" in r["warm_s"] or "slipstream_cpu32" in r["cold_s"]:
            t = r["warm_s"].get("slipstream_cpu32") or r["cold_s"]["slipstream_cpu32"]
            rows.append(dict(label="Slipstream CPU f64, 32^3", code="slipstream", config="cpu-f64-3d32t1c1", mesh="32", device="cpu", n=r["n"], warm_s=t,
                             n_proc=r["n_ok"].get("slipstream_cpu32", 1), us_per_particle=t / r["n"] * 1e6))
    rows.sort(key=lambda r: (r["code"], r["label"], r["n"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(note="median over processes of the warm call time; Slipstream and ASTRA from injector_scaling_control.json, the others from their campaigns",
                                   campaigns=CAMPAIGNS, rows=rows), indent=1) + "\n")
    labels = sorted({r["label"] for r in rows})
    print(f"{len(rows)} points, {len(labels)} curves -> {OUT}")
    for lb in labels:
        pts = [(r["n"], round(r["warm_s"], 1), r["n_proc"]) for r in rows if r["label"] == lb]
        print(f"  {lb:28s}", pts)


if __name__ == "__main__":
    main()

"""Reproduce the bounded report campaign, sequentially (never race timing jobs).

python study.py --campaign 2026-10-26_R1
python study.py --campaign 2026-10-26_R1 --case ring_grad
Successful jobs resume; failed jobs are retained and retried by run.py.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import zipfile

import common
import run

HERE = pathlib.Path(__file__).resolve().parent
DESIGN = {
    "sc_drift": dict(codes="slipstream,astra,gpt,impactx,cheetah,ocelot", configs=",".join([
        "cpu-f64-3d32h1.4", "cpu-f64-n64l128h0.7", "cpu-f64-mesh-acc5", "cpu-f64-mesh-acc6",
        "cpu-f64-mg32x40s2", "cpu-f64-mg64x80s4", "cpu-f64-g32s2", "cpu-f64-g64s4",
        "cpu-f64-3d32t1c1", "cuda-f32-r32z128t2c5", "cuda-f32-r64z256t1c2"])),
    "linear_fodo": dict(codes="slipstream,helix", configs="cpu-f64-o4k4b8,cpu-f64-o6k8b16,cpu-f64-linear"),
    "ring_tunes": dict(codes="slipstream,elegant,xsuite,cheetah,jutrack,bmadx,scibmad", configs=",".join([
        "cuda-f32-o6q8s1b8", "cuda-f64-o6q8s1b8", "cuda-f64-o6k16b32",
        "cpu-f64-o6s8b16", "cpu-f64-o6s16b32", "cpu-f64-yosk8b16", "cpu-f64-yosk16b32",
        "cuda-f64-yosk8b16", "cpu-f64-n16b32", "cpu-f64-n32b64", "cpu-f64-q1s16", "cpu-f64-q1s64",
        "cpu-f64-lin1", "cpu-f64-dri4", "cpu-f64-dri16", "cpu-f64-o4k4", "cpu-f64-o6k32", "cpu-f64-std"])),
    "ring_grad": dict(codes="slipstream,elegant,xsuite,cheetah,jutrack,bmadx,scibmad", configs=",".join([
        "cpu-f64-o6q8s1b8", "cpu-f64-o6s16b32", "cpu-f64-yosk8b16", "cpu-f64-dri4",
        "cpu-f64-n16b32", "cpu-f64-q1s16", "cpu-f64-o4k4", "cpu-f64-o6k32"])),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--case", choices=list(DESIGN))
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    root = common.RESULTS / a.campaign
    root.mkdir(parents=True, exist_ok=True)
    design = dict(cases=DESIGN, procs=a.procs, repeats=a.repeats, threads=1, seed=20261026)
    frozen = root / "design.json"
    if frozen.exists() and common.load(frozen) != design:
        raise ValueError("Campaign design changed; use a new campaign name")
    frozen.write_text(json.dumps(design, indent=2) + "\n")
    # Preserve the exact implementation, not only a commit of a possibly dirty tree.
    archive = root / "source_snapshot.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for base in [HERE] + ([run.SLIPSTREAM / "src"] if run.SLIPSTREAM else []):
                for p in sorted(base.rglob("*")):
                    if p.is_file() and not any(s in p.parts for s in ("results", "__pycache__")) and p.suffix in (".py", ".json", ".toml", ".jl", ".sh", ".npz"):
                        z.write(p, p.relative_to(base.parents[1] if base == HERE else run.SLIPSTREAM.parent).as_posix())
    for case in ([a.case] if a.case else DESIGN):
        d = DESIGN[case]
        cmd = [sys.executable, "-u", str(HERE / "run.py"), "--case", case, "--campaign", a.campaign,
               "--codes", d["codes"], "--configs", d["configs"], "--procs", str(a.procs),
               "--repeats", str(a.repeats), "--threads", "1", "--seed", "20261026"]
        print("Starting", case, flush=True)
        subprocess.run(cmd, check=True)
        analysis = {"sc_drift": "analyse_sc.py", "linear_fodo": "analyse_transport.py",
                    "ring_tunes": "analyse.py", "ring_grad": "analyse_grad.py"}[case]
        subprocess.run([sys.executable, str(HERE / analysis), "--campaign", a.campaign, "--case", case], check=True)
        subprocess.run([sys.executable, str(HERE / "publish_study.py"), "--campaign", a.campaign], check=True)


if __name__ == "__main__":
    main()

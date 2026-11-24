"""Memory and cost of reverse-mode autograd through the injector with space charge (CLARA 5 pC deck, CUDA float32).

Reverse-mode differentiation stores the state of every step, so its memory grows with the number of steps (and of macroparticles and grid cells) unless the
track is checkpointed: ``checkpoint_every = k`` keeps the state of every k-th step and recomputes the others in the backward pass, for a memory of about
``n/k + k`` steps and one extra forward pass. This measures the peak GPU memory and the forward+backward time of the gradient of the emittance with
respect to the six knobs of ``injector_clara_ad --knobs all``, against the number of steps (the length of the track) without checkpointing and with
``k = sqrt(n)``, and against the number of macroparticles with checkpointing; an out-of-memory run is recorded as such.
A run without checkpointing that would not fit in this process's share of the card is skipped, never run: the driver would spill it into host memory
(the unguarded sweep of 3 November 2026 did, with peaks of 12 to 24 GiB on the 8 GiB card, and froze the PC). The slope of the stored graph is measured
on the runs that fit and extrapolated beyond them.

    python injector_clara_memory.py --image on --json ../../results/TAG/memory/injector_ad_memory.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import injector_clara_ad as b



def measure(inj, nominal, beam, n_steps, every, device, image_steps=0):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = {"n_steps": n_steps, "checkpoint_every": every}
    try:
        torch.cuda.synchronize()
        t0 = time.time()
        knobs = {k: torch.tensor(v, dtype=torch.float64, device=device, requires_grad=True) for k, v in nominal.items()}
        o = inj.track_differentiable(knobs, n_steps=n_steps, beam=beam, device=device, dtype=torch.float32, use_compile=False, checkpoint_every=every,
                                     sc_image_steps=image_steps)
        torch.autograd.grad(o["emit_x_mm_mrad"], list(knobs.values()))
        torch.cuda.synchronize()
        out.update(status="ok", seconds=time.time() - t0, peak_MiB=(torch.cuda.max_memory_allocated() - base) / 2**20)
    except (torch.OutOfMemoryError, torch.AcceleratorError) as e:  # the second: a raw cudaMalloc failure that the allocator's cap did not catch
        if not isinstance(e, torch.OutOfMemoryError) and "out of memory" not in str(e):
            raise
        out.update(status="out_of_memory", seconds=None, peak_MiB=(torch.cuda.max_memory_allocated() - base) / 2**20)
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deck", type=Path, default=None)
    p.add_argument("--dt-s", type=float, default=5e-13)
    p.add_argument("--particles", type=int, default=4096)
    p.add_argument("--stops", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5])
    p.add_argument("--particle-sweep", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    p.add_argument("--sweep-stop", type=float, default=0.5)
    p.add_argument("--image", choices=["off", "on"], default="off", help="cathode image in the space-charge solve (the nominal step count is held fixed)")
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--memory-fraction", type=float, default=0.7,
                   help="upper limit of this process's share of the GPU's memory (lower when other programs hold part of the card); a run that would exceed the "
                        "cap raises out-of-memory instead of spilling to host memory")
    p.add_argument("--resume", action="store_true", help="keep the entries already in --json and measure only the missing ones")
    a = p.parse_args()
    a.deck = a.deck or b.deck()
    dev = "cuda"
    # Windows' NVIDIA driver can spill GPU allocations into system memory instead of failing (CUDA - Sysmem Fallback Policy). A sweep that overran the card's
    # 8 GiB that way (peak 23 GiB, then a run that needed ~95 GiB) froze the whole machine on 3 November 2026, so the process is capped below the card's size.
    # Other programs hold part of the card (2.6 GiB on 21 November 2026, a desktop plus a browser): cap this process below what is free now, with a margin of 2 GiB
    # for its own CUDA context and the driver, so that the sum stays under the card's size and nothing can spill.
    free, total = torch.cuda.mem_get_info()
    if free < 3 * 2**30:
        raise SystemExit(f"only {free / 2**30:.1f} GiB of the GPU is free: close the programs that use it and start again")
    fraction = min(a.memory_fraction, (free - 2 * 2**30) / total)
    print(f"GPU: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free; this process is capped at {fraction * total / 2**30:.1f} GiB", flush=True)
    torch.cuda.set_per_process_memory_fraction(fraction)
    budget_MiB = fraction * total / 2**20
    args = argparse.Namespace(deck=a.deck, nz=512, dt_s=a.dt_s, seed=0, n_steps=0, z_stop=0.0, compile=False)
    inj = b._machine(args, a.image == "on")
    image_steps = b.nominal_image_steps(inj, args, a.particles)
    print(f"cathode image {a.image}: {image_steps} steps")
    nominal = b._nominal(inj, "all")
    res = {"steps": [], "particles": [], "device": torch.cuda.get_device_name(0), "dt_s": a.dt_s, "image": a.image, "image_steps": image_steps, "env": {"torch": torch.__version__}}
    if a.resume and a.json.exists():
        res = json.loads(a.json.read_text())

    def done(table, n_p, stop, mode):
        return any(r["n_particles"] == n_p and r["z_stop_m"] == stop and r["mode"] == mode for r in res[table])

    warm = inj.initial_beam(512, dev, torch.float32)
    measure(inj, nominal, warm, 200, 14, dev, image_steps)  # warm-up: CUDA context and first-use costs are not part of the measurement
    probe = measure(inj, nominal, warm, 200, 0, dev, image_steps)  # the stored-graph memory per step and particle, to refuse a run that cannot fit
    per_step_particle_MiB = probe["peak_MiB"] / (200 * 512)

    def run(table, n_p, stop, beam, n):
        for label, every in (("none", 0), ("sqrt", max(1, int(math.sqrt(n))))):
            if done(table, n_p, stop, label):
                continue
            predicted = per_step_particle_MiB * n * n_p   # the probe overestimates by ~1.7: it includes the start-up steps, with few emitted particles
            ok = [r for r in res[table] if r["mode"] == "none" and r["status"] == "ok" and r["n_particles"] == n_p]
            if table == "steps" and len(ok) >= 2:         # the stored graph grows linearly with the steps: extrapolate from the last two measured runs
                p0, p1 = ok[-2], ok[-1]
                predicted = 1.05 * (p1["peak_MiB"] + (p1["peak_MiB"] - p0["peak_MiB"]) / (p1["n_steps"] - p0["n_steps"]) * (n - p1["n_steps"]))
            if label == "none" and predicted > budget_MiB:
                r = {"n_particles": n_p, "z_stop_m": stop, "mode": label, "n_steps": n, "checkpoint_every": 0, "status": "skipped_exceeds_budget",
                     "seconds": None, "peak_MiB": None, "predicted_MiB": predicted}
            else:
                r = {"n_particles": n_p, "z_stop_m": stop, "mode": label, **measure(inj, nominal, beam, n, every, dev, image_steps)}
            res[table].append(r)
            print(r)
            a.json.write_text(json.dumps(res, indent=1))

    for stop in a.stops:
        args.z_stop = stop
        torch.manual_seed(0)
        beam = inj.initial_beam(a.particles, dev, torch.float32)
        run("steps", a.particles, stop, beam, b._steps_to(inj, beam, args))
    args.z_stop = a.sweep_stop
    for n_p in a.particle_sweep:
        torch.manual_seed(0)
        beam = inj.initial_beam(n_p, dev, torch.float32)
        run("particles", n_p, a.sweep_stop, beam, b._steps_to(inj, beam, args))


if __name__ == "__main__":
    main()

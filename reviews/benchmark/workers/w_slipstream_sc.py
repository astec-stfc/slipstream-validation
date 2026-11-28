"""Slipstream adapter for ``sc_drift``: the ``Injector`` (ASTRA-deck driven, Boris push + Poisson space charge) reads the same ASTRA deck and
macroparticle file that the ASTRA adapter writes, so both start from identical input.  Rungs vary the fixed space-charge mesh, the fixed time
step and the field-refresh interval (``subcycle_steps``); the last rung is the 3-D Cartesian solver.  Statistics are the Injector's own
trajectory records interpolated to the stations (station 0 is the common-function value of the input particles)."""
import dataclasses
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from astra_deck import write_deck  # noqa: E402
from common import lab_stats, launches, run_worker  # noqa: E402
from phase_space import save_snapshot  # noqa: E402

# (nr, nz, dt [ps], subcycle steps)
_RUNGS = [(16, 64, 4.0, 10), (32, 128, 2.0, 5), (64, 256, 1.0, 2), (64, 256, 0.5, 1)]
CONFIGS = [dict(name=f"cuda-f32-r{nr}z{nz}t{dt:g}c{c}", device="cuda", dtype="float32", nr=nr, nz=nz, dt_ps=dt, subcycle=c, geometry="cylindrical")
           for nr, nz, dt, c in _RUNGS[:3]]
CONFIGS += [dict(name=f"cpu-f64-r{nr}z{nz}t{dt:g}c{c}", device="cpu", dtype="float64", nr=nr, nz=nz, dt_ps=dt, subcycle=c, geometry="cylindrical")
            for nr, nz, dt, c in _RUNGS]
CONFIGS.append(dict(name="cpu-f64-3d32t1c1", device="cpu", dtype="float64", nr=0, nz=0, dt_ps=1.0, subcycle=1, geometry="3d"))
# Separate exploratory control; these names are not selected by the frozen R1 design.
CONFIGS += [dict(name=f"cuda-f{t[-2:]}-3d32t1c1", device="cuda", dtype=t, nr=0, nz=0,
                 dt_ps=1.0, subcycle=1, geometry="3d") for t in ("float32", "float64")]
_CART32 = next(c for c in CONFIGS if c["name"] == "cuda-f32-3d32t1c1")
CONFIGS[-1:-1] = [dict(_CART32, name=f"cuda-f32-3d32t1c{c}-r{r}", subcycle=c, record_every=r)
                  for c, r in ((1, 50), (1, 100), (5, 100), (10, 100))]
CONFIGS.insert(-1, dict(_CART32, name="cuda-f32-3d32t0.5c20-r200", dt_ps=0.5, subcycle=20, record_every=200))
for n in (16, 64):
    CONFIGS.insert(-1, dict(_CART32, name=f"cuda-f32-3d{n}t1c10-r100", n3d=n, subcycle=10, record_every=100))
# Deck momenta converted with the case mass (Slipstream's reader divides by the CODATA mass), so the native bunch has the case's beta*gamma.
CONFIGS.insert(-1, dict(_CART32, name="cuda-f32-3d32t1c10-r100-mfix", subcycle=10, record_every=100, mass_fix=True))
CONFIGS.insert(-1, dict(CONFIGS[next(i for i, c in enumerate(CONFIGS) if c["name"] == "cpu-f64-3d32t1c1")], name="cpu-f64-3d32t1c1-mfix", mass_fix=True))
# Convergence extension (mass-fixed deck): finer native meshes and 64^3 step / field-refresh refinements of the fast preset.
for n in (64, 96, 128):
    CONFIGS.insert(-1, dict(_CART32, name=f"cuda-f32-3d{n}t1c10-r100-mfix", n3d=n, subcycle=10, record_every=100, mass_fix=True))
CONFIGS.insert(-1, dict(_CART32, name="cuda-f32-3d64t0.5c20-r200-mfix", n3d=64, dt_ps=0.5, subcycle=20, record_every=200, mass_fix=True))
CONFIGS.insert(-1, dict(_CART32, name="cuda-f32-3d64t1c5-r50-mfix", n3d=64, subcycle=5, record_every=50, mass_fix=True))
for n, padding in ((64, .05), (64, .2), (81, .2)):
    CONFIGS.insert(-1, dict(_CART32, name=f"cuda-f32-3d{n}t1c10-r100-p{padding:g}",
                            n3d=n, subcycle=10, record_every=100, box_padding_3d=padding))


def setup():
    import slipstream.tracking.native.injector  # noqa: F401


def build(lat, case, cfg):
    from slipstream.tracking.native.injector import Injector

    d = pathlib.Path(tempfile.mkdtemp(prefix="bench_slipstream_sc_"))
    deck = write_deck(d, launches(case), case, **(dict(mc2_eV=case["mass_eV"]) if cfg.get("mass_fix") else {}))  # ASTRA mesh settings are not read for space charge
    inj = Injector.from_astra(deck)
    sc = dataclasses.replace(inj.space_charge, cathode_image=False, dt_adaptive=False, subcycle_steps=cfg["subcycle"], geometry=cfg["geometry"],
                             mesh="fixed", nr=cfg["nr"] or 64, nz=cfg["nz"] or 256, n_xyz=[cfg.get("n3d", 32)] * 3,
                             box_padding_3d=cfg.get("box_padding_3d", .05))
    inj.space_charge = sc
    inj.dt_s = cfg["dt_ps"] * 1e-12
    inj.z_stop_m = case["stations_m"][-1]
    return dict(inj=inj, case=case, lab=launches(case)), None


def track(model, init, turns, cfg):
    import torch

    inj, case = model["inj"], model["case"]
    res = inj.track(device=cfg["device"], dtype=getattr(torch, cfg["dtype"]), use_compile=cfg["device"] == "cuda",
                    record_every=cfg.get("record_every", 5), return_particles=True,
                    verbose=False, z_stop_m=case["stations_m"][-1] + 0.05)
    from slipstream.constants import C_LIGHT

    save_snapshot(*(res[k].double().cpu().numpy() for k in ("x", "y", "z")),
                  *(res[k].double().cpu().numpy() / C_LIGHT for k in ("ux", "uy", "uz")))   # u = gamma v
    tr = {k: np.asarray(v, dtype=float) for k, v in res["trajectory"].items()}
    st = np.array(case["stations_m"])
    f = lambda key, scale: np.interp(st, tr["z_m"], tr[key]) * scale  # noqa: E731
    out = np.column_stack([st, f("sigma_x_mm", 1e-3), f("sigma_y_mm", 1e-3), f("sigma_z_mm", 1e-3), f("emit_x_mm_mrad", 1e-6), f("emit_y_mm_mrad", 1e-6),
                           f("Ekin_MeV", 1e6), f("dE_rms_keV", 1e3)])
    s0 = lab_stats(*model["lab"].T, mass_eV=case["mass_eV"])  # station 0 is the input, by the common function
    out[0, 1:] = [s0[k] for k in ("sigma_x_m", "sigma_y_m", "sigma_z_m", "eps_nx_m", "eps_ny_m", "ekin_eV", "sigma_E_eV")]
    return out


def sync(dev):
    if dev == "cuda":
        import torch
        torch.cuda.synchronize()


def peak_mem(dev):
    if dev != "cuda":
        return {}
    import torch
    return dict(cuda_peak_allocated=int(torch.cuda.max_memory_allocated()), cuda_peak_reserved=int(torch.cuda.max_memory_reserved()))


def versions():
    import slipstream
    import torch
    return dict(torch=torch.__version__, cuda=torch.version.cuda, slipstream_file=slipstream.__file__)


if __name__ == "__main__":
    run_worker("slipstream", CONFIGS, build, track, sync, peak_mem, versions, setup)

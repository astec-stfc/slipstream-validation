"""One ImpactX run for ``sc_drift`` in its own process (AMReX can be initialised once per process).  usage: impactx_run.py <run.json>
run.json: dir, npz (lab-frame macroparticles), charge_C, kin_eV, ds, n_steps, nslice, n_cell, solver, mass_eV, max_level, prob_relative."""
import json
import os
import sys

import numpy as np

cfg = json.load(open(sys.argv[1]))
os.chdir(cfg["dir"])
from impactx import ImpactX, elements  # noqa: E402

z = np.load(cfg["npz"])
sim = ImpactX()
sim.max_level = cfg["max_level"]
sim.n_cell = cfg["n_cell"]
nb = max(cfg["n_cell"][0] // 2, 4)
sim.blocking_factor_x = [min(nb, 16)]
sim.blocking_factor_y = [min(nb, 16)]
sim.blocking_factor_z = [4]
sim.max_grid_size_x = [cfg["n_cell"][0]]
sim.max_grid_size_y = [cfg["n_cell"][1]]
sim.max_grid_size_z = [cfg["n_cell"][2]]
sim.particle_shape = 2
sim.space_charge = cfg["space_charge"]
if cfg["space_charge"] == "3D":
    sim.poisson_solver = cfg["solver"]
    sim.dynamic_size = True
    sim.prob_relative = cfg["prob_relative"]
sim.slice_step_diagnostics = True
sim.init_grids()
ref = sim.beam.ref
ref.set_species("electron").set_kin_energy_MeV(cfg["kin_eV"] * 1e-6)
bg0 = ref.pz
# lab-frame snapshot at one instant -> ImpactX converts to fixed s inside at_fixed_t(); momenta are relative to the reference pz
with sim.beam.at_fixed_t():
    sim.beam.add_n_particles(z["x"], z["y"], z["z"], z["bgx"] / bg0, z["bgy"] / bg0, (z["bgz"] - bg0) / bg0,
                             -1.0 / (cfg["mass_eV"] * 1e-6) / 1e6, bunch_charge=abs(cfg["charge_C"]))
sim.lattice.extend([elements.Drift(name=f"d{i}", ds=cfg["ds"], nslice=cfg["nslice"]) for i in range(cfg["n_steps"])])
sim.track_particles()
sim.finalize()

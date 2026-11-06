"""Recompute every number quoted in the Slipstream paper from the archived evidence in evidence/.

Each check recomputes one quoted value from a saved file and compares it with the value written in the text.  Needs numpy only,
no simulation code.  Writes results/claims.csv (one row per check); exit status 1 if any check fails.
Run: python reviews/claims/verify_claims.py   (or: pytest -m tier0)
"""
import csv
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "evidence" / "2026-10-26_R1"
OUT = HERE.parents[1] / "results"


def J(name):
    return json.loads((DATA / name).read_text())


corr, ring_rows = J("corrected_reference.json"), None
ring = {(r["code"], r["config"]): r for r in corr["ring_tunes"]}
xs = {r["config"]: r for r in corr["xsuite"]["rows"] if r["case"] == "ring_tunes"}
cpu = {r["case"]: r for r in corr["cpu"]["rows"]}
elem = {(r["code"], r["element"]): r["max_error"] for r in J("element_attribution.json")["rows"] if "max_error" in r}
sub = J("element_substitution_control.json")
scal = {(r["code"], r["config"], r["n"]): r for r in J("particle_scaling_control.json")["rows"] if r["n_ok"]}
ga, gb = J("gradient_acceptance.json")["rows"], J("gradient_acceptance_b.json")["rows"]
fd = {r["config"]: r for r in J("fd_ad_endpoint_control.json")["rows"]}
rich = J("richardson_reference.json")
ce = J("convergence_extension.json")
co = {(r["code"], r["config"]): r for r in J("charge_off_control.json")["rows"]}
inj = {r["n"]: r for r in J("injector_scaling_control.json")["rows"]}
seed = J("injector_scaling_control.json")["seed_control"]
mf = J("mass_fix_control.json")


def ga_row(rows, code, config, mode):
    return next(r for r in rows if r["code"] == code and r["config"] == config and r["mode"] == mode)


def ring_dq(code, config):
    return ring[(code, config)]["dq_ref"] if (code, config) in ring else xs[config]["dq_ref"]


checks = []   # (id, location, quoted, actual, rel_tol)


def chk(cid, loc, quoted, actual, tol=0.02):
    checks.append((cid, loc, quoted, float(actual), tol))


# ---- ring (Table II, Sec. IV A)
for cid, code, cfg, q in [("R1", "slipstream", "cuda-f32-o6k8b16", 1.87e-6), ("R2", "slipstream", "cuda-f64-o6k8b16", 1.63e-9), ("R3", "slipstream", "cuda-f64-o6k16b32", 3.59e-11),
                          ("R4", "elegant", "cpu-f64-o6s8b16", 3.82e-9), ("R5", "jutrack", "cpu-f64-n32b64", 4.17e-5), ("R6", "bmadx", "cpu-f64-q1s64", 1.33e-4),
                          ("R7", "cheetah", "cpu-f64-dri16", 1.04e-4), ("R8", "cheetah", "cpu-f64-lin1", 3.36e-3), ("R9", "xsuite", "cpu-f64-yosk8b16-exact", 6.86e-6),
                          ("R10", "xsuite", "cpu-f64-yosk16b32", 2.96e-4), ("R11", "xsuite", "cpu-f64-yosk16b32-exact", 6.03e-7)]:
    chk(cid, f"Table II / Sec. IV A: tune error {code} {cfg}", q, ring_dq(code, cfg), 0.02)
chk("R12", "Table II: Slipstream compiled CPU tune error", 1.63e-9, cpu["ring_tunes"]["dq_ref"], 0.02)
chk("R13", "Table II: warm time GPU f64 o6k8b16", 2.94, ring[("slipstream", "cuda-f64-o6k8b16")]["warm_s"], 0.02)
chk("R14", "Table II: warm time GPU f32 o6k8b16", 0.663, ring[("slipstream", "cuda-f32-o6k8b16")]["warm_s"], 0.02)
chk("R15", "Table II: first-turn max(x,y) GPU f64", 1.08e-11, max(ring[("slipstream", "cuda-f64-o6k8b16")]["onturn_x"], ring[("slipstream", "cuda-f64-o6k8b16")]["onturn_y"]), 0.02)
chk("R16", "Table II: first-turn max(px,py) GPU f32", 2.54e-8, max(ring[("slipstream", "cuda-f32-o6k8b16")]["onturn_px"], ring[("slipstream", "cuda-f32-o6k8b16")]["onturn_py"]), 0.02)
# ---- element-level (Sec. IV A)
chk("E1", "JuTrack bend deviation [m]", 9.2e-5, elem[("jutrack", "bend")]["x_m"], 0.02)
chk("E2", "Bmad-X sextupole deviation [m]", 1.0e-5, elem[("bmadx", "sext")]["x_m"], 0.03)
chk("E3", "Cheetah/Bmad-X quadrupole deviation [m]", 2.2e-7, elem[("cheetah", "quad")]["x_m"], 0.02)
chk("E4", "Cheetah sextupole deviation [m]", 2.0e-7, elem[("cheetah", "sext")]["x_m"], 0.03)
chk("E5", "Xsuite adaptive drift deviation [m]", 3.2e-6, elem[("xsuite", "drift")]["x_m"], 0.02)
chk("E6", "Slipstream worst element deviation <= 2e-13", 2e-13, max(max(v.values()) for (c, _), v in elem.items() if c == "slipstream"), 0.2)
chk("E7", "elegant quad/sext worst deviation (text: 1e-9)", 1e-9, max(max(elem[("elegant", e)].values()) for e in ("quad", "sext")), 0.25)
# ---- scaling
chk("P5", "ratio Xsuite/GPU f32 at 5000 (216)", 216, scal[("xsuite", "cpu-f64-yosk8b16-exact", 5000)]["warm_s"] / scal[("slipstream", "cuda-f32-o6k8b16", 5000)]["warm_s"], 0.03)
chk("P6", "ratio Xsuite/GPU f64 at 5000 (22)", 22, scal[("xsuite", "cpu-f64-yosk8b16-exact", 5000)]["warm_s"] / scal[("slipstream", "cuda-f64-o6k8b16", 5000)]["warm_s"], 0.03)
chk("P7", "per particle-turn GPU f32 at 50000 [us]", 0.11, scal[("slipstream", "cuda-f32-o6k8b16", 50000)]["per_particle_turn_us"], 0.03)
chk("P8", "per particle-turn GPU f64 at 50000 [us]", 1.7, scal[("slipstream", "cuda-f64-o6k8b16", 50000)]["per_particle_turn_us"], 0.03)
chk("P9", "per particle-turn elegant at 50000 [us]", 134, scal[("elegant", "cpu-f64-o6s8b16", 50000)]["per_particle_turn_us"], 0.02)
# ---- derivatives
chk("D1", "Slipstream e_abs launch 1 (FD) (text: 3e-7)", 3e-7, ga_row(ga, "slipstream", "cpu-f64-o6k8b16", "fd")["e_abs"], 0.05)
chk("D2", "Slipstream e_rel launch 1", 9.9e-7, ga_row(ga, "slipstream", "cpu-f64-o6k8b16", "fd")["e_rel"], 0.02)
chk("D3", "Slipstream e_abs launch 2 (AD)", 3.35e-7, ga_row(gb, "slipstream", "cpu-f64-o6k8b16-compiled", "ad")["e_abs"], 0.02)
chk("D4", "Slipstream e_rel launch 2 (AD)", 8.3e-7, ga_row(gb, "slipstream", "cpu-f64-o6k8b16-compiled", "ad")["e_rel"], 0.02)
chk("D5", "JuTrack e_abs launch 2", 1.4, ga_row(gb, "jutrack", "cpu-f64-n16b32", "fd")["e_abs"], 0.03)
chk("D6", "Xsuite exact e_abs launch 1", 1.2e-3, ga_row(ga, "xsuite", "cpu-f64-yosk8b16-exact", "fd")["e_abs"], 0.03)
ratios = [r["e_over_traj"] for r in ga if r["config"] != "cpu-f64-o6k8b16-compiled"]
gb_cmp = [r for r in gb if r["traj_gap"] > 0]   # the reference itself is not a comparison
c = fd["cpu-f64-o6k8b16-compiled"]
chk("F1", "compiled: AD cost in forward evaluations (9.3)", 9.3, c["ad_s"] / c["forward_s"], 0.02)
chk("F2", "compiled: AD time [s] (text: 1.9)", 1.9, c["ad_s"], 0.02)
chk("F4", "compiled: forward evaluation [s] (text: 0.2)", 0.2, c["forward_s"], 0.02)
lad = fd["cpu-f64-o6k8b16"]["single_step_fd_error_vs_best"]
chk("F6", "FD error at step 1e-3 (text: 0.09)", 0.09, lad["0.001"], 0.03)
chk("F7", "FD error at step 1e-4 (text: 1e-3)", 1e-3, lad["0.0001"], 0.06)
chk("F9", "AD error vs best FD (text: 4e-9)", 4e-9, lad["ad"], 0.03)
chk("F10", "AD speed-up vs extrapolated pair (2.2)", 2.2, 2 * c["fd_endpoint_s"] / c["ad_s"], 0.03)
chk("F11", "compiled CPU Jacobian AD vs own FD, Frobenius (text: 4e-9)", 4e-9, cpu["ring_grad"]["ad_vs_fd"][0], 0.03)
# ---- AD against FD and optimisation (29 October controls; Sec. V Derivatives)
import numpy as np

kl = J("fd_ad_knob_ladder.json")["rows"]
chk("D1", "AD Jacobian time, minimum over 1-40 knobs [s] (1.6)", 1.6, min(r["ad_s"] for r in kl), 0.02)
chk("D2", "AD Jacobian time, maximum over 1-40 knobs [s] (2.4)", 2.4, max(r["ad_s"] for r in kl), 0.02)
chk("D3", "AD cost in forward evaluations, median (8)", 8, float(np.median([r["ad_over_forward"] for r in kl])), 0.05)
chk("D4", "central difference, one step, 1 knob [s] (0.4)", 0.4, kl[0]["fd1_s"], 0.05)
chk("D5", "central difference, one step, 40 knobs [s] (20)", 20, kl[-1]["fd1_s"], 0.03)
chk("D6", "extrapolated pair, 1 knob [s] (0.8)", 0.8, kl[0]["fd2_s"], 0.03)
chk("D7", "extrapolated pair, 40 knobs [s] (34)", 34, kl[-1]["fd2_s"], 0.03)
chk("D8", "FD one step / AD at 40 knobs (8)", 8, kl[-1]["fd1_over_ad"], 0.03)
chk("D9", "FD pair / AD at 40 knobs (14)", 14, kl[-1]["fd2_over_ad"], 0.03)
chk("D10", "single step slower than AD from 5 knobs, not at 3 (break-even 3-5)", 1, float(next(r for r in kl if r["n_knobs"] == 3)["fd1_over_ad"] < 1 < next(r for r in kl if r["n_knobs"] == 5)["fd1_over_ad"]), 0)
chk("D11", "extrapolated pair slower than AD from 2-3 knobs (about 2)", 1, float(next(r for r in kl if r["n_knobs"] == 3)["fd2_over_ad"] > 1 and next(r for r in kl if r["n_knobs"] == 2)["fd2_over_ad"] < 1.1), 0)
chk("D12", "FD one step vs AD, worst e_abs (1.5e-6)", 1.5e-6, max(r["fd1_vs_ad_e_abs"] for r in kl), 0.05)
chk("D13", "FD pair vs AD, worst e_abs (3.4e-8)", 3.4e-8, max(r["fd2_vs_ad_e_abs"] for r in kl), 0.05)
al = J("fd_ad_knob_ladder.json")["alignment"]["rows"]
chk("D14", "quadrupole offsets, best e_abs at 0.1 um (4e-9)", 4e-9, min(r["e_abs_vs_ad"] for r in al), 0.1)
oc = J("optimisation_convergence_control.json")["rows"]
selc = lambda fam, n, m: [r for r in oc if r["family"] == fam and r["n_knobs"] == n and r["method"] == m]  # noqa: E731
tw = lambda fam, n, m: [r["wall_s"] for r in selc(fam, n, m)]  # noqa: E731
chk("O1", "24 knobs, AD time to 1e-9, min [s] (13)", 13, min(tw("A", 24, "lbfgs_ad")), 0.05)
chk("O2", "24 knobs, AD time to 1e-9, max [s] (14)", 14, max(tw("A", 24, "lbfgs_ad")), 0.05)
chk("O3", "24 knobs, FD L-BFGS-B time, min [s] (214)", 214, min(tw("A", 24, "lbfgs_fd")), 0.01)
chk("O4", "24 knobs, FD L-BFGS-B time, max [s] (251)", 251, max(tw("A", 24, "lbfgs_fd")), 0.01)
chk("O5", "24 knobs, Nelder-Mead time, min [s] (328)", 328, min(tw("A", 24, "neldermead")), 0.01)
chk("O6", "24 knobs, Nelder-Mead time, max [s] (387)", 387, max(tw("A", 24, "neldermead")), 0.01)
chk("O7", "24 knobs, Powell runs reaching 1e-9 (0 of 3)", 0, sum(r["reached"] for r in selc("A", 24, "powell")), 0)
bad = [r for r in oc if r["family"] == "B" and r["method"] == "lbfgs_ad"]
chk("O8", "4-12 knobs, AD time to 1e-9, min [s] (13)", 13, min(r["wall_s"] for r in bad), 0.05)
chk("O9", "4-12 knobs, AD time to 1e-9, max [s] (45)", 45, max(r["wall_s"] for r in bad), 0.02)
chk("O10", "4-12 knobs, AD runs reaching 1e-9 (12 of 12)", 12, sum(r["reached"] for r in bad), 0)
ratio = lambda m, fam, n: float(np.median([r["wall_s"] / next(a["wall_s"] for a in selc(fam, n, "lbfgs_ad") if a["seed"] == r["seed"]) for r in selc(fam, n, m)]))  # noqa: E731
rb = [ratio(m, "B", n) for m in ("lbfgs_fd", "neldermead") for n in (4, 6, 8, 12)]
chk("O11", "4-12 knobs, FD and Nelder-Mead over AD, min (3)", 3, min(rb), 0.15)
chk("O12", "4-12 knobs, FD and Nelder-Mead over AD, max (12)", 12, max(rb), 0.03)
chk("O13", "FD L-BFGS-B over AD at 4 knobs (3.4)", 3.4, ratio("lbfgs_fd", "B", 4), 0.03)
chk("O14", "FD L-BFGS-B over AD at 24 knobs (18)", 18, ratio("lbfgs_fd", "A", 24), 0.03)
chk("O15", "median FD L-BFGS-B time at 24 knobs [min] (3.7)", 3.7, float(np.median(tw("A", 24, "lbfgs_fd"))) / 60, 0.02)
chk("O16", "median Nelder-Mead time at 24 knobs [min] (6)", 6, float(np.median(tw("A", 24, "neldermead"))) / 60, 0.06)
ijf = J("injector_derivative_full.json")
ij = ijf["rows"]
best = sorted(r["best_rel"] for r in ij)
abs_err = sorted(r["best_abs"] for r in ij)
worst = max(ij, key=lambda r: r["best_rel"])
chk("I11", "injector autograd vs FD, full length, best step, minimum (7e-9)", 7e-9, best[0], 0.05)
chk("I12", "injector autograd vs FD, full length, best step, maximum (6e-6)", 6e-6, best[-1], 0.06)
chk("I14", "injector derivative check: six knobs, four objectives (24 pairs) over the whole 3.37 m, 1024 particles, CPU float64", 1,
    float(len(ij) == 24 and len({r["knob"] for r in ij}) == 6 and len({r["objective"] for r in ij}) == 4 and abs(ijf["design"]["z_stop_m"] - 3.37147) < 1e-5
          and ijf["design"]["particles"] == 1024 and ijf["design"]["dtype"] == "float64" and ijf["design"]["device"] == "cpu"), 0)
# sensitivity profile (Sec. IV, Fig. injad d, e) and the memory of the backward pass (Fig. injad c, f)
ap, mm = J("injector_ad_profile.json"), J("injector_ad_memory.json")
apn = ap["nominal"]
def apsens(row, obj, knob):
    return 100 * row["grad"][obj][knob] * (1.0 if knob.endswith("phase_deg") else 0.01 * abs(apn[knob])) / row["values"][obj]
late = [r for r in ap["rows"] if r["z_reached_m"] >= 1.3]
chk("AP1", "text: emittance response to the solenoid settles between -1 and -2.4% per per cent beyond 1.3 m", 1,
    float(all(-2.45 <= apsens(r, "emit_x_mm_mrad", "solenoids[0].b_max_T") <= -0.95 for r in late)), 0)
chk("AP2", "text: emittance response to the gun field between +0.5 and +1.4% beyond 1.3 m", 1,
    float(all(0.45 <= apsens(r, "emit_x_mm_mrad", "cavities[0].e_max_V_per_m") <= 1.5 for r in late)), 0)
chk("AP3", "text: linac phase and field change the emittance by < 0.02% per unit beyond 1.3 m", 1,
    float(all(abs(apsens(r, "emit_x_mm_mrad", k)) < 0.02 for r in late for k in ("cavities[1].phase_deg", "cavities[1].e_max_V_per_m"))), 0)
chk("AP4", "text: energy spread responds to the linac phase by about -6% per degree at the end of the line (and to the gun phase and field before the structure)", -6,
    apsens(ap["rows"][-1], "dE_rms_keV", "cavities[1].phase_deg"), 0.06)
fde = J("injector_fd_energy_spread.json")
fde_best = {}
for r in fde["rows"]:
    fde_best[r["knob"]] = min(fde_best.get(r["knob"], 1.0), r["rel"])
chk("AP6", "text: autograd of the energy spread agrees with central differences to better than 2e-6 for all six knobs at the end of the line", 1,
    float(len(fde_best) == 6 and max(fde_best.values()) < 2e-6 and abs(fde["z_stop_m"] - 3.37147) < 1e-5), 0)
chk("AP5", "text: before the structure the energy spread responds to the gun phase and field (> 4% per unit at 0.4 m)", 1,
    float(all(abs(apsens(next(r for r in ap["rows"] if abs(r["z_reached_m"] - 0.42) < 0.05), "dE_rms_keV", k)) > 4 for k in ("cavities[0].phase_deg", "cavities[0].e_max_V_per_m"))), 0)
none = sorted((r for r in mm["steps"] if r["mode"] == "none" and r["status"] == "ok"), key=lambda r: r["n_steps"])
slope = (none[-1]["peak_MiB"] - none[0]["peak_MiB"]) / (none[-1]["n_steps"] - none[0]["n_steps"])
sqrt_ = {r["n_steps"]: r["peak_MiB"] for r in mm["steps"] if r["mode"] == "sqrt" and r["status"] == "ok"}
chk("M1", "text: stored-graph memory 1.7 MiB per step for 4096 macroparticles", 1.7, slope, 0.03)
chk("M2", "text: the 8 GiB of the GPU is exceeded after about 4800 steps", 4800, mm["device_MiB"] / slope, 0.03)
chk("M3", "text: checkpointing needs 0.2 GiB at 10 000 steps", 0.2, sqrt_[max(sqrt_)] / 1024, 0.05)
pk = {r["n_particles"]: r["peak_MiB"] for r in mm["particles"] if r["mode"] == "sqrt" and r["status"] == "ok"}
chk("M4", "text: checkpointed memory 1.8 GiB at 65 536 macroparticles", 1.8, pk[65536] / 1024, 0.02)
chk("M5", "text: checkpointed memory grows roughly linearly with macroparticles (per-particle memory within 15% from 4096 to 65 536)", 1,
    float(0.85 <= (pk[65536] / 65536) / (pk[4096] / 4096) <= 1.15), 0)
# tolerance budget (Fig. injapps a, b)
tb = J("injector_tolerance.json")
tbr = {(r["output"], r["scale"]): r for r in tb["rows"]}
sc1 = {o: tbr[(o, 1.0)] for o in tb["design"]["outputs"]}
rel1 = [100 * r["jacobian_std"] / r["value"] for r in sc1.values()]
chk("TB1", "text: predicted rms jitters of the five outputs between 0.1 and 1.2%", 1, float(round(min(rel1), 1) == 0.1 and round(max(rel1), 1) == 1.2), 0)
for cid, out, k, q in (("TB2", "emit_x_mm_mrad", 5, 93), ("TB3", "sigma_x_mm", 1, 82), ("TB4", "sigma_z_mm", 0, 81), ("TB5", "Ekin_MeV", 3, 77), ("TB6", "dE_rms_keV", 2, 50)):
    chk(cid, f"text: knob {k} supplies {q}% of the variance of {out}", q, 100 * sc1[out]["variance_share"][k], 0.01)
chk("TB7", "text: Monte Carlo of 40 runs at 1, 5 and 20 times the tolerances agrees with the Jacobian within its 95% interval (all 15 cases)", 1,
    float(tb["design"]["scales"] == [1.0, 5.0, 20.0] and all(r["n"] == 40 for r in tb["rows"]) and len(tb["rows"]) == 15
          and all(r["mc_std_ci"][0] <= r["jacobian_std"] <= r["mc_std_ci"][1] for r in tb["rows"])), 0)
chk("TB8", "text: one-at-a-time differences over +-10 sigma give the same budget to 1%", 1,
    float(tb["design"]["oat_scale"] == 10.0 and all(abs(r["oat_linear_std"] / r["jacobian_std"] - 1) <= 0.01 for r in tb["rows"] if r["scale"] == 1.0)), 0)
chk("TB9", "text: the autograd Jacobian of the five outputs costs 24 forward runs", 24, tb["design"]["autograd_s"] / tb["design"]["forward_s_median"], 0.05)
chk("TB10", "text: Monte Carlo 120 runs, one-at-a-time differences 12 runs (6 knobs, plus and minus)", 1, float(tb["design"]["n_samples"] == 120 and len(tb["design"]["knobs"]) * 2 == 12), 0)
# five-knob tuning (Fig. injapps c, d)
op = J("injector_optimiser.json")
s1, s2 = op["starts"]["start1"]["methods"], op["starts"]["start2"]["methods"]
chk("TU1", "text: Bayesian optimization finishes lowest from both starts (0.376)", 0.376, max(s1["bayes"]["final_best_J"], s2["bayes"]["final_best_J"]), 0.003)
chk("TU2", "text: Bayesian optimization finishes lowest (below autograd and Nelder-Mead) from both starts", 1,
    float(all(s["bayes"]["final_best_J"] < min(s["lbfgs_ad"]["final_best_J"], s["neldermead"]["final_best_J"]) for s in (s1, s2))), 0)
chk("TU3", "text: Bayesian optimization within 10% of the best objective in 200 s (first start)", 200, s1["bayes"]["to_10pct"]["t_s"], 0.03)
chk("TU4", "text: Bayesian optimization within 10% of the best objective in 320 s (second start)", 320, s2["bayes"]["to_10pct"]["t_s"], 0.03)
chk("TU5", "text: first start, evaluations to come within 10%: autograd 7, Bayesian optimization 9, Nelder-Mead 27", 1,
    float((s1["lbfgs_ad"]["to_10pct"]["n_eval"], s1["bayes"]["to_10pct"]["n_eval"], s1["neldermead"]["to_10pct"]["n_eval"]) == (7, 9, 27)), 0)
tr2 = s2["lbfgs_ad"]["trace"]
chk("TU6", "text: autograd is still descending when the time is up from the second start (best objective falls > 5% over the last 400 s)", 1,
    float(min(r["J_best"] for r in tr2) < 0.95 * min(r["J_best"] for r in tr2 if r["t_s"] <= tr2[-1]["t_s"] - 400)), 0)
chk("TU7", "text: one autograd evaluation costs 6 forward runs", 6, (s1["lbfgs_ad"]["median_s_per_eval"] / s1["neldermead"]["median_s_per_eval"]
                                                                   + s2["lbfgs_ad"]["median_s_per_eval"] / s2["neldermead"]["median_s_per_eval"]) / 2, 0.05)
chk("TU8", "text: Nelder-Mead stops at 0.53 from the second start; each method has 40 minutes", 1,
    float(abs(s2["neldermead"]["final_best_J"] - 0.53) < 0.005 and op["design"]["budget_s"] == 2400.0), 0)

# ---- Sec. V further comparisons: CSR, IBS, booster (records collected by scripts/collect_further.py)
csrv = J("csr_validation.json")
fast_ratio = {(r["n"], r["bins"]): r["ratio"] for r in csrv["c1_analytic"]["rows"] if r["kernel"] == "fast"}
chk("F1", "CSR closed-form coefficient K (0.350)", 0.350, csrv["c1_analytic"]["K"], 0.002)
chk("F2", "steady kernel within 0.2% of the closed form from 400 to 1600 bins at 1e6 particles", 1,
    float(all(abs(fast_ratio[(1048576, b)] - 1) <= 0.002 for b in (400, 1600))), 0)
ent = {round(r["phi"], 4): r for r in csrv["c3_wake_check"]["entrance"]}
full, half = ent[0.1325], ent[0.066]
chk("F4", "entrance transient bunch-mean loss within 1% at 0.1325 and 0.066 rad", 1,
    float(all(abs(r["transient"]["mean_native"] / r["ref_mean_over_bunch"] - 1) <= 0.01 for r in (full, half))), 0)
dr = csrv["c3_wake_check"]["drift"]
chk("F6", "retarded drift kernel mean loss within 0.3% from 5 mm to 1 m", 1, float(all(abs(r["retarded"]["mean"] / r["ref_mean"] - 1) <= 0.003 for r in dr)), 0)
el = {round(r["z_m"], 3): r for r in csrv["c3_elegant_drift"]}
chk("F8", "elegant drift wake 3 to 6% above the retarded field near the exit (5 mm, 0.1 m)", 1,
    float(all(0.03 <= el[z]["elegant_mean"] / el[z]["reference_mean"] - 1 <= 0.06 for z in (0.005, 0.105))), 0)
chk("F9", "elegant drift wake above the retarded field at 0.9 m (17%)", 0.17, el[0.905]["elegant_mean"] / el[0.905]["reference_mean"] - 1, 0.05)

# single dipole and drift against the retarded-field ideal (c6_ladder, c6_slices, c6_ocelot_leadin)
lad, sl = csrv["c6_ladder"], csrv["c6_slices"]["dipole_loss_over_ideal"]
LM = lambda name, b=400: lad["models"][f"{name} | {b} bins | 1048576"]  # noqa: E731
ocl = csrv["c6_ocelot_leadin"]["rows"]["2.0"]
chk("L3", "Slipstream (transient, retarded drift) and Ocelot reproduce dipole and drift to 1% (4 values within 1.2%)", 1,
    float(all(abs(v - 1) <= 0.012 for v in (LM("native transient, retarded drift")["dipole_ratio"], LM("native transient, retarded drift")["drift_ratio"],
                                            ocl["dipole_ratio"], ocl["drift_ratio"]))), 0)
chk("L4", "steady-state kernel in the dipole overestimates by 44%", 0.44, LM("native steady, overtaking drift")["dipole_ratio"] - 1, 0.02)
chk("L5", "steady drift cut at the overtaking length underestimates by 39%", 0.39, 1 - LM("native steady, overtaking drift")["drift_ratio"], 0.02)
chk("L7", "elegant transient dipole 9% above the ideal with 8 slices", 0.09, LM("elegant transient, Stupakov drift")["dipole_ratio"] - 1, 0.05)
chk("L8", "elegant transient dipole within 1% of the ideal at 64 slices", 1, float(abs(sl["64"]["elegant"] - 1) <= 0.012), 0)
chk("L9", "Slipstream dipole converged to 1% from 8 slices", 1, float(all(abs(sl[k]["native"] / sl["64"]["native"] - 1) <= 0.01 for k in ("8", "16", "32"))), 0)

# first two dipoles at 5 pC against the multi-bend reference (c6_twobend)
tb = csrv["c6_twobend_5pC"]
SEG = ("dipole 1", "drift 1", "dipole 2", "drift 2")
tot = lambda k: sum(tb[k][s] for s in SEG) * 1e-3  # noqa: E731
chk("B4", "both 2% below the reference in total", 0.02, 1 - tot("ocelot") / tot("reference"), 0.15)
for who, key in (("Slipstream", "native transient, retarded drift"), ("Ocelot", "ocelot")):
    chk(f"B7-{who}", f"{who} 6% below the reference in the second dipole", 0.06, 1 - tb[key]["dipole 2"] / tb["reference"]["dipole 2"], 0.1)

# full compressor (final_full / final_bends / Ocelot)
FULL, BENDS, OC = csrv["chicane"]["1048576_final_full"], csrv["chicane"]["1048576_final_bends"], csrv["c5_ocelot_1048576_leadin"]
fs = ("0.8", "0.925", "1.0")
quoted = {  # Table (csr): loss [keV], emittance [um] for factors 0.8, 0.925, 1.0
    ("bends", "elegant_transient"): [(9.9, 0.4995), (165.3, 0.9416), (65.9, 0.8030)],
    ("bends", "native_fast_tr"): [(8.5, 0.4950), (153.7, 0.8858), (61.0, 0.7659)],
    ("full", "elegant_transient"): [(44.4, 0.5548), (373.7, 1.7442), (126.7, 1.1156)],
    ("full", "native_tr_ret"): [(44.1, 0.5442), (332.8, 1.6036), (115.2, 1.0283)],
    ("ocelot", "csr_b400"): [(43.3, 0.5417), (356.0, 1.6168), (126.5, 1.1162)],
}
for (kind, mode), vals in quoted.items():
    run = {"full": FULL, "bends": BENDS, "ocelot": OC}[kind]
    for f, (loss_q, eps_q) in zip(fs, vals):
        chk(f"T-{kind}-{mode}-{f}-loss", f"CSR table, {kind}, {mode}, factor {f}: loss [keV]", loss_q, run[f][mode]["loss_keV"], 0.006)
        chk(f"T-{kind}-{mode}-{f}-eps", f"CSR table, {kind}, {mode}, factor {f}: emittance [um]", eps_q, run[f][mode]["eps_x_um"], 0.0005)
chk("C1", "no-CSR emittance 0.483 um in the three codes, to 0.1%", 1,
    float(all(abs(FULL[f]["native_off"]["eps_x_um"] / FULL[f]["elegant_off"]["eps_x_um"] - 1) < 1e-3 and abs(OC[f]["off"]["eps_x_um"] / FULL[f]["elegant_off"]["eps_x_um"] - 1) < 1e-3
              and abs(FULL[f]["elegant_off"]["eps_x_um"] - 0.483) < 0.001 for f in fs)), 0)
dl = [BENDS[f]["native_fast_tr"]["loss_keV"] / BENDS[f]["elegant_transient"]["loss_keV"] - 1 for f in fs]
chk("C2", "dipoles only: Slipstream loss 7 to 14% below elegant", 1, float(all(-0.145 <= x <= -0.065 for x in dl)), 0)
S = {f: FULL[f]["native_tr_ret"] for f in fs}
E = {f: FULL[f]["elegant_transient"] for f in fs}
O = {f: OC[f]["csr_b400"] for f in fs}
chk("C3", "weakest strength: all three agree to 3% in loss", 1, float(max(abs(S["0.8"]["loss_keV"] / O["0.8"]["loss_keV"] - 1), abs(E["0.8"]["loss_keV"] / O["0.8"]["loss_keV"] - 1), abs(S["0.8"]["loss_keV"] / E["0.8"]["loss_keV"] - 1)) <= 0.03), 0)
chk("C9", "nominal: Ocelot and elegant coincide (loss to 0.5%)", 1, float(abs(E["1.0"]["loss_keV"] / O["1.0"]["loss_keV"] - 1) <= 0.005), 0)
chk("C10", "nominal: Slipstream loss 9% below Ocelot", -0.09, S["1.0"]["loss_keV"] / O["1.0"]["loss_keV"] - 1, 0.02)
spread = lambda key: max((max(v) - min(v)) / min(v) for v in ([x[f][key] for x in (S, E, O)] for f in fs))  # noqa: E731
c2 = lambda n, b: csrv["chicane"][f"{n}_c2_n{n}_b{b}"]  # noqa: E731
chg = lambda f, mode: c2(1048576, 1600)[f][mode]["loss_keV"] / c2(1048576, 400)[f][mode]["loss_keV"] - 1  # noqa: E731
chk("C14-el", "elegant loss rises by about 5% from 400 to 1600 bins (both strengths within 4 to 5.5%)", 1,
    float(all(0.04 <= chg(f, "elegant_transient") <= 0.055 for f in ("0.925", "1.0"))), 0)
chk("C14-nat", "Slipstream loss rises by 2 to 3% from 400 to 1600 bins", 1, float(all(0.02 <= chg(f, "native_tr_ret") <= 0.03 for f in ("0.925", "1.0"))), 0)
chk("C15", "Ocelot changes by less than 1% between 400 and 800 bins (loss)", 1,
    float(all(abs(OC[f]["csr_b800"]["loss_keV"] / OC[f]["csr_b400"]["loss_keV"] - 1) < 0.01 for f in fs)), 0)
dn = [abs(c2(1048576, 400)[f]["native_tr_ret"]["loss_keV"] / c2(65536, 400)[f]["native_tr_ret"]["loss_keV"] - 1) for f in ("0.925", "1.0")]
chk("C16", "Slipstream converged in particle number to 1% (65 536 to 1 048 576, loss)", 1, float(max(dn) <= 0.01), 0)

# run times (c4_timing, Ocelot timing, c4b kernels): Table (csrtime) with speed-up over elegant's transient model in parentheses
tim, ot = csrv["c4_timing"], {n: csrv[f"c5_ocelot_{n}_timing_leadin"]["0.925"] for n in ("100000", "1048576")}
TQ = {  # row -> {particles: (time [s], speed-up)}
    "elegant": {"1048576": (19.3, 1.0), "100000": (1.76, 1.0)},
    "ocelot": {"1048576": (43.3, 0.45), "100000": (6.1, 0.29)},
    ("native_cpu1", "transient_retarded"): {"1048576": (25.1, 0.77), "100000": (1.42, 1.2)},
    ("native_cpu", "transient_retarded"): {"1048576": (8.65, 2.2), "100000": (0.84, 2.1)},
    ("native_cuda32", "transient_retarded"): {"1048576": (1.61, 12), "100000": (0.54, 3.2)},
    ("native_cuda", "transient_retarded"): {"1048576": (2.51, 7.7), "100000": (0.59, 3.0)},
    ("native_cuda", "steady"): {"1048576": (1.16, 17), "100000": (0.15, 12)},
    ("native_cuda", "transient"): {"1048576": (1.16, 17), "100000": (0.18, 9.8)},
    ("native_cuda", "transient_stupakov"): {"1048576": (6.96, 2.8), "100000": (1.57, 1.1)},
}
for row, by_n in TQ.items():
    for n, (q_t, q_s) in by_n.items():
        base = tim[n]["elegant"]["transient"]["median_s"]
        t = (tim[n]["elegant"]["transient"]["median_s"] if row == "elegant" else ot[n]["csr_b400"]["wall_s"] if row == "ocelot" else tim[n][row[0]][row[1]]["median_s"])
        tag = row if isinstance(row, str) else "-".join(row)
        chk(f"TT-{tag}-{n}-time", f"run-time table: {tag}, {n} particles, time [s]", q_t, t, 0.012 + 0.006 / q_t)
        chk(f"TT-{tag}-{n}-speedup", f"run-time table: {tag}, {n} particles, speed-up over elegant", q_s, base / t, 0.04)
g = tim["1048576"]["native_cuda"]
chk("TT2", "Ocelot takes 2.2 times as long as elegant", 2.2, ot["1048576"]["csr_b400"]["wall_s"] / tim["1048576"]["elegant"]["transient"]["median_s"], 0.03)
chk("TT3", "retarded kernel takes a third of the closed-form drift time", 1 / 3, g["transient_retarded"]["median_s"] / g["transient_stupakov"]["median_s"], 0.15)
kb = csrv["c4b_kernels"]
chk("TT4", "Warp kernel 30 times faster than the PyTorch bisection", 30, kb["retarded_drift_call"]["torch_bisection"] / kb["retarded_drift_call"]["warp"], 0.05)
k64, k32 = kb["cuda_float64_1048576"], kb["cuda_float32_1048576"]
chk("TT11", "pairwise pass at 1e5, double precision, takes minutes (60 to 600 s)", 1, float(60 < tim["100000"]["native_cuda"]["pairwise"]["median_s"] < 600), 0)
ibs = J("ibs_elena_bridge.json")
chk("F24", "IBS element-averaged vs xsuite: within 5% in all three planes", 1, float(all(ibs["gap_element_averaged"][k] <= 0.05 for k in ("inv_Tx_s", "inv_Ty_s", "inv_Tp_s"))), 0)
chk("F27", "IBS single point vs xsuite: between 12 and 19%", 1, float(all(0.12 <= v <= 0.19 for v in ibs["gap_single_point"].values())), 0)
# ---- booster dynamics (D2B_NonlinearDynamics rerun of 31 October, clean run times from parallel_baseline.py)
bo = J("booster_nonlinear.json")
bs, bsv, bpb, bw = bo["summary"], bo["survival"], bo["parallel_baseline"], bo["fma_wall_s"]
et = bs["elegant_twiss"]
chk("BO1", "booster fractional tunes within 5e-5 of elegant's twiss output", 5e-5,
    max(abs(bs["Q0"][0] - et["nux"] % 1), abs(bs["Q0"][1] - et["nuy"] % 1)), 0.05)
chk("BO2", "booster chromaticities within 0.02% of elegant's", 1,
    float(all(abs(a / b - 1) <= 2e-4 for a, b in zip(bs["xi_native"], (et["dnux/dp"], et["dnuy/dp"])))), 0)
dlt = bo["elegant_tunes"]["deltas"]
dq = [max(abs(o["Qx"] - qx), abs(o["Qy"] - qy)) for o, qx, qy, d in zip(bo["optics"]["opts"], bo["elegant_tunes"]["Qx"], bo["elegant_tunes"]["Qy"], dlt) if abs(d) <= 0.02 + 1e-9]
chk("BO5", "DA area float32 vs elegant within 3% over the three offsets", 0.03, max(abs(r["gpu32"]["area_mm2"] / r["elegant"]["area_mm2"] - 1) for r in bsv.values()), 0.05)
agree = [1 - r[n]["n_differ"] / r["n_inside"] for r in bsv.values() for n in ("gpu32", "cpu64_split")]
chk("BO9", "float32 frequency map 3.4 decades noisier than double-single", 3.4, bs["fma"]["f32_median"] - bs["fma"]["ds_median"], 0.02)
sv, pbs, pbf = bpb["survival"], bpb["survival"], bpb["fma"]
chk("BO12", "DA grid on the GPU in float32 (9 s)", 9, min(sv["gpu32"]["walls"]), 0.02)
chk("BO14", "GPU float32 75 times faster than serial elegant", 75, sv["elegant_serial"]["wall"] / min(sv["gpu32"]["walls"]), 0.02)
chk("BO16", "GPU float32 11 times faster than Pelegant on 16 ranks", 11, sv["pelegant16"]["wall"] / min(sv["gpu32"]["walls"]), 0.02)
chk("BO22", "double-single FMA 2.3 times slower than Pelegant on 32 ranks", 2.3, bw["ds"] / pbf["pelegant32"]["wall"], 0.02)
chk("BO23", "float32 FMA 7.4 times faster than Pelegant on 32 ranks", 7.4, pbf["pelegant32"]["wall"] / bw["f32"], 0.02)

# ---- resistive-wall impedance against live IW2D (Stories/NEG_ImpedanceIW2D)
imp = J("impedance_iw2d.json")
ia = imp["arrays"]
f_imp = [f for f in ia["freq_Hz"]]
dev = lambda x, y: [abs(a / b - 1) for a, b in zip(x, y)]  # noqa: E731
fl = ia["flat"]
flat_re, flat_im = dev(fl["yokoya_re"], fl["iw2d_re"]), dev(fl["yokoya_im"], fl["iw2d_im"])
chk("IM1", "round longitudinal and dipolar impedances within a relative 3e-8 of IW2D", 1,
    float(all(imp[k][m] <= 3e-8 for k in ("round_long", "round_xdip") for m in ("re_max", "im_max"))), 0)
chk("IM2", "native evaluation of 91 frequencies: 7 ms", 7, 1e3 * imp["round_native_s"], 0.1)
chk("IM3", "IW2D round chamber run: 0.4 s", 0.4, imp["round_iw2d_s"], 0.1)
chk("IM4", "1 kHz to 1 THz", 1, float(imp["f_min_Hz"] == 1e3 and imp["f_max_Hz"] == 1e12), 0)
chk("IM5", "2.2 GeV", 2.2, imp["gamma"] * 0.51099895e-3, 0.001)
chk("IM7", "flat longitudinal: real and imaginary parts within 1% up to 5 GHz", 1, float(all(max(a, b) <= 0.01 for f, a, b in zip(f_imp, flat_re, flat_im) if f <= 5.1e9)), 0)
chk("IM8", "kb = 1 at 3.7 GHz for the 13 mm radius", 3.7, 299792458.0 / (2 * math.pi * imp["radius_m"]) * 1e-9, 0.01)
chk("IM9", "flat longitudinal: real part 4% off at 10 GHz", 0.04, flat_re[f_imp.index(min(f_imp, key=lambda f: abs(f - 1e10)))], 0.05)

# ---- space charge drift
fin = ce["native_vs_astra_finest"]
for mesh, (sz, em, sp) in {"16": (4.7, 11, 6.8), "32": (1.2, 3.3, 2.0), "64": (0.37, 1.2, 0.58), "96": (0.20, 0.75, 0.28), "128": (0.14, 0.51, 0.14)}.items():
    chk(f"M{mesh}s", f"Table III: size vs ASTRA128 at {mesh}^3 [%]", sz, 100 * fin[mesh]["size"], 0.06)
    chk(f"M{mesh}e", f"Table III: emittance x vs ASTRA128 at {mesh}^3 [%]", em, 100 * fin[mesh]["emit_x"], 0.05)
    chk(f"M{mesh}p", f"Table III: spread vs ASTRA128 at {mesh}^3 [%]", sp, 100 * fin[mesh]["spread"], 0.05)
chk("M3", "ASTRA 64->128 emittance change [%]", 0.9, 100 * ce["astra_successive"]["128_vs_64"]["emit_x"], 0.02)
chk("M4", "ASTRA 64->128 spread change [%]", 0.22, 100 * ce["astra_successive"]["128_vs_64"]["spread"], 0.02)
en = ce["mean_energy_gain_eV"]
for k, q in (("128", 31.0),):
    chk(f"N{k}a", f"ASTRA particle-file gain at {k}^3 [eV]", q, en["astra_particles"][k], 0.01)
for k, q in (("128", 30.5),):
    chk(f"N{k}n", f"Slipstream gain at {k}^3 [eV]", q, en["native"][k], 0.01)
sl = co[("slipstream", "cpu-f64-3d32t1c1")]
chk("C1", "charge-off: Slipstream CPU f64 spread gap at 20 pC [%]", 0.9, 100 * sl["q20_vs_astra_q20"]["spread"], 0.05)
chk("C2", "charge-off: Slipstream CPU f64 emittance gap at 20 pC [%]", 0.070, 100 * sl["q20_vs_astra_q20"]["emit"], 0.02)
chk("C3", "charge-off: Slipstream CPU f64 size gap at 20 pC [%]", 0.06, 100 * sl["q20_vs_astra_q20"]["size"], 0.05)
chk("C4", "charge-off: ImpactX spread gap at 20 pC [%] (text: 9)", 9, 100 * co[("impactx", "cpu-f64-mg32x40s2")]["q20_vs_astra_q20"]["spread"], 0.05)
for k, (lim, q) in {"astra": ("astra", None)}.items():
    pass
# ---- Richardson
ag, ag2 = rich["agreement"], rich["at_200k"]["agreement"]
chk("L2", "20k: emittance x limit gap [%]", 0.60, 100 * ag["eps_nx_m"]["limit_gap_rel_max"], 0.02)
chk("L3", "20k: emittance y limit gap [%] (text: 0.8)", 0.8, 100 * ag["eps_ny_m"]["limit_gap_rel_max"], 0.06)
chk("L5", "200k: size limit gap [%]", 0.04, 100 * ag2["sigma_x_m"]["limit_gap_rel_max"], 0.03)
chk("L6", "200k: emittance x limit gap [%]", 0.28, 100 * ag2["eps_nx_m"]["limit_gap_rel_max"], 0.02)
chk("L7", "200k: spread limit gap [%] (text: 0.04)", 0.04, 100 * ag2["sigma_E_eV"]["limit_gap_rel_max"], 0.05)
# ---- particle-count ladder
chk("I1", "GPU 64^3 call at 2000 particles [s] (text: 4-5)", 4.8, inj[2000]["warm_s"]["slipstream64"], 0.02)
chk("I2", "GPU 64^3 call at 200000 particles [s]", 4.6, inj[200000]["warm_s"]["slipstream64"], 0.02)
chk("I3", "GPU 64^3 call at 1e6 particles [s] (text: 13)", 13, inj[1000000]["warm_s"]["slipstream64"], 0.05)
a2 = inj[2000]["warm_s"].get("astra32") or inj[2000]["cold_s"]["astra32"]
a200 = inj[200000]["warm_s"].get("astra32") or inj[200000]["cold_s"]["astra32"]
chk("I4", "ASTRA 32^3 at 2000 particles [s]", 9, a2, 0.05)
chk("I5", "ASTRA 32^3 at 200000 particles [s]", 330, a200, 0.02)
g32_2 = inj[2000]["warm_s"].get("slipstream32") or inj[2000]["cold_s"]["slipstream32"]
g32_200 = inj[200000]["warm_s"].get("slipstream32") or inj[200000]["cold_s"]["slipstream32"]
chk("I6", "GPU/ASTRA time ratio at 2000 particles (3.0)", 3.0, a2 / g32_2, 0.03)
chk("I7", "GPU/ASTRA time ratio at 200000 particles (90)", 90, a200 / g32_200, 0.03)
chk("I8", "seed control: max emittance order at 200000 (1.94)", 1.94, max(v["order_emit_x"] for v in seed["200000"].values()), 0.01)
chk("I9", "seed control: min emittance order at 20000 (1.39)", 1.39, min(v["order_emit_x"] for v in seed["20000"].values()), 0.01)
chk("I10", "seed control: max emittance order at 2000 (0.42)", 0.42, max(v["order_emit_x"] for v in seed["2000"].values()), 0.02)

# ---- CLARA injector (Sec. V.E, Table VIII): converged ASTRA and Slipstream, from Stories/CLARA_InjectorConvergence/final_comparison.py
cl = J("clara_final.json")
CH = {"20": "20pC", "100": "100pC", "250": "250pC"}
pp = lambda q: max(0.06 / abs(q), 1e-9) if q else 0  # noqa: E731  one decimal in percent: tolerance 0.06 percentage points
# Table VIII (one row per code): eps_n,x [mm mrad], then the differences from converged ASTRA [%] in eps_x, sigma_x, sigma_z, sigma_E, eps_z; run time [min or s] in the last column
TV = {
    "20": {"ASTRA, baseline": (0.3108, -0.3, -0.1, 0.2, 0.6, 0.6, 8.7), "ASTRA, converged": (0.3117, None, None, None, None, None, 89.1),
           "Slipstream, default": (0.3122, 0.2, -4.1, -0.7, 2.1, 2.1, 14), "Slipstream, converged": (0.3121, 0.1, -4.5, -0.8, 1.9, 1.7, 82),
           "GPT": (0.3296, 5.7, 2.9, -2.9, 3.4, 5.3, 27)},
    "100": {"ASTRA, baseline": (0.4151, -2.9, 0.2, 0.5, 1.3, 4.8, 9.2), "ASTRA, converged": (0.4274, None, None, None, None, None, 91.3),
            "Slipstream, default": (0.4197, -1.8, -5.1, -0.9, 3.0, 0.9, 17), "Slipstream, converged": (0.4316, 1.0, -5.5, -1.0, 2.7, 0.6, 252),
            "GPT": (0.4434, 3.7, 5.6, -3.2, 4.9, 1.2, 36)},
    "250": {"ASTRA, baseline": (0.4830, -13.1, 0.1, 0.8, 2.8, 13.0, 10.5), "ASTRA, converged": (0.5560, None, None, None, None, None, 111),
            "Slipstream, default": (0.5452, -1.9, -4.1, -1.6, 4.3, -0.8, 22), "Slipstream, converged": (0.5325, -4.2, -3.7, -1.9, 3.4, -1.6, 132),
            "GPT": (0.5631, 1.3, 2.3, -4.8, 7.9, -0.1, 32)},
}
KEYS5 = ("eps_x", "sigma_x", "sigma_z", "dE", "eps_z")
for ch, name in CH.items():
    r = cl[name]
    rows = {"ASTRA, baseline": (r["astra"]["baseline"], r["astra_baseline_vs_converged_pct"]), "ASTRA, converged": (r["astra_converged"], None),
            "Slipstream, default": (r["native"]["default"], r["native_default_vs_astra_converged_pct"]),
            "Slipstream, converged": (r["native"]["converged"], r["native_converged_vs_astra_converged_pct"]), "GPT": (r["gpt"], r["gpt_vs_astra_converged_pct"])}
    for who, (end, d) in rows.items():
        q = TV[ch][who]
        chk(f"CL{ch}-{who}-eps", f"Table VIII: eps_n,x at {ch} pC, {who}", q[0], end["eps_x"], 0.0003)
        for key, qq in zip(KEYS5, q[1:6]):
            if qq is None:
                continue
            chk(f"CL{ch}-{who}-{key}", f"Table VIII: {who} vs converged ASTRA, {key} at {ch} pC [%]", qq, d[key], pp(qq))
    chk(f"CL{ch}-t-astra", f"Table VIII: baseline ASTRA run time on 16 threads at {ch} pC [min]", TV[ch]["ASTRA, baseline"][6], r["astra_baseline_parallel_wall_min"], 0.01)
    chk(f"CL{ch}-t-astra-serial", f"text: serial baseline ASTRA run time at {ch} pC [min]", {"20": 63, "100": 65, "250": 67}[ch], r["astra_baseline_wall_min"], 0.01)
    chk(f"CL{ch}-par-eq", f"text: parallel baseline ASTRA gives the serial run's emittance at {ch} pC [mm mrad]", r["astra"]["baseline"]["eps_x"], r["astra_baseline_parallel_eps_x"], 1e-4)
    chk(f"CL{ch}-t-slip", f"Table VIII: Slipstream default loop time at {ch} pC [s]", TV[ch]["Slipstream, default"][6], r["native"]["default"]["loop_s"], 0.04)
    chk(f"CL{ch}-t-slip-conv", f"Table VIII: Slipstream converged loop time at {ch} pC [s]", TV[ch]["Slipstream, converged"][6], r["native"]["converged"]["loop_s"], 0.04)
    chk(f"CL{ch}-t-gpt", f"Table VIII: GPT run time at {ch} pC [min]", TV[ch]["GPT"][6], r["gpt"]["wall_s"] / 60, 0.02)
    chk(f"CL{ch}-ship", f"text: baseline ASTRA below converged ASTRA at {ch} pC [%]", {"20": 0.3, "100": 2.9, "250": 13.0}[ch], -r["astra_baseline_vs_converged_pct"]["eps_x"], 0.1)
cl20, cl100, cl250 = (cl[v] for v in CH.values())
r20, r100, r250 = cl20, cl100, cl250
allr = (cl20, cl100, cl250)
rg = [x["gpt"]["wall_s"] / x["native"]["default"]["loop_s"] for x in allr]
rp = [60 * x["astra_baseline_parallel_wall_min"] / x["native"]["default"]["loop_s"] for x in allr]
chk("CL-x-astra-par", "text: baseline ASTRA on 16 threads 29 to 38 times longer than Slipstream's loop", 1, float(28.5 <= min(rp) <= 29.5 and 37.5 <= max(rp) <= 38.5), 0)
chk("CL-x-gpt", "text: GPT 90 to 130 times longer than Slipstream's loop", 1, float(88 <= min(rg) <= 92 and 128 <= max(rg) <= 134), 0)
# speed against accuracy
for ch, name in CH.items():
    chk(f"CL{ch}-t-astra-conv", f"Table VIII: converged ASTRA run time on 16 threads at {ch} pC [min]", TV[ch]["ASTRA, converged"][6], cl[name]["astra_converged_parallel_wall_min"], 0.01)
    chk(f"CL{ch}-par-eq-conv", f"text: parallel converged ASTRA gives the serial run's emittance at {ch} pC [mm mrad]", cl[name]["astra"]["hmfine"]["eps_x"], cl[name]["astra_converged_parallel_eps_x"], 1e-4)
rc = [x["astra_converged_parallel_wall_min"] / x["astra_baseline_parallel_wall_min"] for x in allr]
chk("CL-speed-astra", "text: baseline ASTRA 10 to 11 times faster than converged ASTRA (both on 16 threads)", 1, float(round(min(rc)) == 10 and round(max(rc)) == 11), 0)
rs = [x["native"]["converged"]["loop_s"] / x["native"]["default"]["loop_s"] for x in allr]
chk("CL-speed-slip", "text: Slipstream default 6 to 15 times faster than its converged settings", 1, float(round(min(rs)) == 6 and round(max(rs)) == 15), 0)
chk("CL-base-3pct", "text: baseline ASTRA within 3% of converged at 20 and 100 pC", 1, float(all(abs(x["astra_baseline_vs_converged_pct"]["eps_x"]) <= 3.0 for x in (cl20, cl100))), 0)
chk("CL-dflt-2pct", "text: Slipstream default within 2% of converged ASTRA at every charge", 1, float(all(abs(x["native_default_vs_astra_converged_pct"]["eps_x"]) <= 2.0 for x in allr)), 0)
chk("CL-ladder", "text: ASTRA emittance levels off near an emission-update interval of 22 fs (finer interval changes it by < 1.5%, 100 and 250 pC)", 1,
    float(all(abs(x["astra"]["hm001094"]["eps_x"] / x["astra"]["hm002188"]["eps_x"] - 1) < 0.015 for x in (cl100, cl250))), 0)
# converged codes (Sec. V.E): GPT and Slipstream against converged ASTRA
gv = {k: [x["gpt_vs_astra_converged_pct"][k] for x in allr] for k in ("eps_x", "sigma_x", "sigma_z", "dE", "Ekin")}
sv = {k: [x["native_converged_vs_astra_converged_pct"][k] for x in allr] for k in ("eps_x", "sigma_x", "sigma_z", "dE", "Ekin")}
chk("CL-conv-6pct", "text: converged codes agree on eps_x to within 6% at all three charges", 1, float(max(abs(v) for v in gv["eps_x"] + sv["eps_x"]) <= 6.0), 0)
chk("CL-gpt-eps", "abstract and text: GPT emittance 1 to 6% above converged ASTRA", 1, float(1.0 <= min(gv["eps_x"]) <= 1.5 and 5.5 <= max(gv["eps_x"]) <= 6.0), 0)
# structure wake (Sec. V.E): ASTRA 4.0 (Wk_N_bin 160) and Slipstream (128 bins) at 250 pC, each against its own no-wake run
wk = J("clara_wake.json")
wa, wn = wk["astra"]["160"], wk["native"]["128"]
dEkin_keV = [abs(1e3 * x["delta"]["Ekin_MeV"]) for x in (wa, wn)]
dsig_pct = [100 * abs(x["delta"]["dE_rms_keV"]) / wk[b]["dE_rms_keV"] for x, b in ((wa, "astra_no_wake"), (wn, "native_no_wake"))]
gaps = {k: 100 * (wn["delta"][k] / wa["delta"][k] - 1) for k in ("Ekin_MeV", "dE_rms_keV", "emit_z_kev_mm")}
kick_gap = [100 * (wk["per_kick_eV"]["native"][c] / wk["per_kick_eV"]["astra"][c] - 1) for c in wk["per_kick_eV"]["astra"]]
chk("WK-dEkin", "text: the wake lowers the final mean energy by about 40 keV in both codes", 40, np.mean(dEkin_keV), 0.1)
chk("WK-spread", "text: the wake lowers the energy spread by 12 to 13% in both codes", 1, float(all(round(v) in (12, 13) for v in dsig_pct)), 0)
chk("WK-epsx", "text: the wake leaves the transverse emittance unchanged (< 0.2%)", 1,
    float(max(abs(x["delta"]["emit_x_mm_mrad"]) / wk[b]["emit_x_mm_mrad"] for x, b in ((wa, "astra_no_wake"), (wn, "native_no_wake"))) < 0.002), 0)
chk("WK-end-6pct", "text: Slipstream reproduces ASTRA's change of the energy and the energy spread to 6%", 6, max(abs(gaps["Ekin_MeV"]), abs(gaps["dE_rms_keV"])), 0.05)
chk("WK-kick-1pct", "text: the energy loss of one kick agrees to 1%", 1, float(max(abs(v) for v in kick_gap) <= 1.0), 0)
chk("WK-testfile", "text: ASTRA's own wake potential reproduced to 0.01% of its peak", 0.01,
    100 * max(v["max_abs_diff_over_peak"] for v in wa["testfile_vs_native_kernel"].values()), 0.5)
chk("WK-time", "text: the wake adds no measurable time to the tracking loop (< 5%)", 1,
    float(abs(np.mean(wk["loop_s"]["wake"]) / np.mean(wk["loop_s"]["nowake"]) - 1) < 0.05), 0)
chk("CL-gpt-trend", "text: GPT excess falls with charge", 1, float(gv["eps_x"][0] > gv["eps_x"][1] > gv["eps_x"][2] > 0), 0)
chk("CL-slip-vs-gpt", "text: Slipstream closer to ASTRA than GPT at 20 and 100 pC, 4% below it at 250 pC", 1,
    float(all(abs(sv["eps_x"][i]) < abs(gv["eps_x"][i]) for i in (0, 1)) and round(sv["eps_x"][2]) == -4), 0)
chk("CL-size-sign", "text: rms size below ASTRA's in Slipstream and above it in GPT at every charge", 1, float(all(v < 0 for v in sv["sigma_x"]) and all(v > 0 for v in gv["sigma_x"])), 0)
chk("CL-gpt-vs-slip", "text: both codes give a shorter bunch and higher energy spread than ASTRA, GPT by more", 1,
    float(all(s < 0 and abs(s) < abs(g) and g < 0 for s, g in zip(sv["sigma_z"], gv["sigma_z"])) and all(0 < s < g for s, g in zip(sv["dE"], gv["dE"]))), 0)
chk("CL-ekin", "text: the three final energies agree to 0.3%", 1, float(max(abs(v) for v in gv["Ekin"] + sv["Ekin"]) <= 0.3), 0)
gs = cl["gpt_20pC_sensitivity"]
chk("CL-gpt-n2", "text: GPT 20 pC emittance falls by 1.8% from 65536 to 262144 particles", 1.8, 100 * (1 - gs["n262144"]["eps_x"] / gs["n65536"]["eps_x"]), 0.03)
chk("CL-gpt-mesh", "text: GPT 20 pC emittance falls by 0.2% when the mesh lines are doubled at 262144 particles", 0.2, 100 * (1 - gs["nfac2_n262144"]["eps_x"] / gs["n262144"]["eps_x"]), 0.15)
# Slipstream choices
nr = lambda x, a, b: 100 * (x["native_runs"][a]["eps_x"] / x["native_runs"][b]["eps_x"] - 1)  # noqa: E731
chk("CL-image-250pC", "text: removing the cathode image raises the emittance by up to 60% (250 pC)", 60, nr(cl250, "no_image", "ref"), 0.04)
chk("CL-radial", "text: emittance rises by 4% from 64 to 256 radial cells at 100 pC", 4, nr(r100, "refnr256_n1m", "refnr64_n1m"), 0.02)

# ---- additions to the Overleaf text of 5 November: Xsuite on the GPU (Fig. 3), the CSR entrance transient (Fig. 8)
xg = J("xsuite_gpu_scaling.json")
xr = {r["n"]: r for r in xg["rows"]}
xcpu = lambda n: scal[("xsuite", "cpu-f64-yosk8b16-exact", n)]["warm_s"]  # noqa: E731
ratio = [r["xsuite_gpu_f64"]["warm_s"] / r["slipstream_gpu_f64"]["warm_s"] for r in xg["rows"]]
chk("XG1", "text: Xsuite on the same GPU in float64 takes 1.6-1.8 times as long as Slipstream at every particle count", 1, float(round(min(ratio), 1) == 1.6 and round(max(ratio), 1) == 1.8), 0)
chk("XG2", "text: Xsuite GPU float64 is 13-14 times faster than its single-thread CPU calculation at 5000 and 50000 particles", 1,
    float(all(13 <= xcpu(n) / xr[n]["xsuite_gpu_f64"]["warm_s"] <= 14 for n in (5000, 50000))), 0)
chk("XG3", "text: Xsuite GPU float64 takes 2.8 us per particle-turn at 50000 particles", 2.8, xr[50000]["xsuite_gpu_f64"]["per_particle_turn_us"], 0.01)
chk("XG4", "text: Xsuite's GPU line build takes about 5 min per process", 300, sum(xg["xsuite_gpu_build"]["build_s"]) / len(xg["xsuite_gpu_build"]["build_s"]), 0.06)
chk("XG5", "text: float32 speed-up over single-thread Xsuite at 5000 particles is 222 (same-session GPU data)", 222, xcpu(5000) / xr[5000]["slipstream_gpu_f32"]["warm_s"], 0.01)
chk("XG6", "text: float64 speed-up over single-thread Xsuite at 5000 particles is 22 (same-session GPU data)", 22, xcpu(5000) / xr[5000]["slipstream_gpu_f64"]["warm_s"], 0.01)
chk("XG7", "text: the Xsuite GPU points are medians of two processes", 2, xg["design"]["xsuite_processes"], 0)
cw = J("csr_validation.json")["c3_wake_check"]
ent = cw["entrance"]
tr = [r["transient"]["mean_native"] / r["ref_mean_over_bunch"] for r in ent]
chk("CS1", "text: entrance-transient mean loss agrees to 1% at the last two sampled angles, 0.066 and 0.1325 rad", 1,
    float([r["phi"] for r in ent][-2:] == [0.066, 0.1325] and all(abs(v - 1) <= 0.01 for v in tr[-2:])), 0)
chk("CS2", "text: larger differences occur earlier (the transient kernel is 34% and 21% below the reference at 0.0166 and 0.033 rad and zero at the two smallest angles)", 1,
    float(tr[0] == 0 and tr[1] == 0 and 0.30 < tr[2] < 0.36 and 0.76 < tr[3] < 0.81), 0)
chk("CS3", "text: the retarded-drift mean loss agrees to 1% over the sampled downstream points", 1, float(all(abs(100 * (r["retarded"]["mean"] / r["ref_mean"] - 1)) <= 1 for r in cw["drift"])), 0)

# ---- beam evolution along the 100 pC line (Fig. clara_profile): refined ASTRA, refined Slipstream and GPT
import numpy as np  # noqa: E402

_st = HERE / "evidence" / "clara"
_fc = J("clara_final.json")
_ast = [np.loadtxt(_st / "astra" / f"100pC_{t}" / "injector400.Xemit.001") for t in _fc["100pC"]["astra_converged_runs"]]
_nat = json.loads((_st / "runs" / f"100pC_{_fc['native_converged_run']['100pC']}.json").read_text())["traj"]
_gp = J("clara_gpt_profile.json")
_za = _ast[0][:, 0]
_za = _za[(_za >= 0.25) & (_za <= 3.38)]
_gr = [r for r in _gp["rows"] if 0.25 <= r["z_m"] <= 3.4]
_zg = np.array([r["z_m"] for r in _gr])
TWS = 1.19357   # entrance of the travelling-wave structure


def _resid(code, key_native, key_gpt, col):
    if code == "native":
        ref = np.mean([np.interp(_za, a[:, 0], a[:, col]) for a in _ast], axis=0)
        return _za, 100 * (np.interp(_za, _nat["z_m"], _nat[key_native]) / ref - 1)
    ref = np.mean([np.interp(_zg, a[:, 0], a[:, col]) for a in _ast], axis=0)
    return _zg, 100 * (np.array([r[key_gpt] for r in _gr]) / ref - 1)


def _rng(code, key_native, key_gpt, col, lo, hi):
    z, r = _resid(code, key_native, key_gpt, col)
    m = (z >= lo) & (z < hi)
    return float(r[m].min()), float(r[m].max()), float(r[-1])


def _near(v, q, tol=0.05):
    return float(abs(v - q) <= tol)


ne = lambda lo, hi: _rng("native", "emit_can_x_mm_mrad", None, 5, lo, hi)  # noqa: E731
ns = lambda lo, hi: _rng("native", "sigma_x_mm", None, 3, lo, hi)  # noqa: E731
ge = lambda lo, hi: _rng("gpt", None, "emit_can_x_mm_mrad", 5, lo, hi)  # noqa: E731
gs_ = lambda lo, hi: _rng("gpt", None, "sigma_x_mm", 3, lo, hi)  # noqa: E731
chk("CP1", "text: before the structure Slipstream's emittance is 0.1 to 2.8% above the ASTRA mean", 1, _near(ne(0.25, TWS)[0], 0.1) * _near(ne(0.25, TWS)[1], 2.8), 0)
chk("CP2", "text: before the structure GPT's emittance is between -0.9 and +2.0% of the ASTRA mean", 1, _near(ge(0.25, TWS)[0], -0.9) * _near(ge(0.25, TWS)[1], 2.0), 0)
chk("CP3", "text: before the structure the beam sizes of Slipstream and GPT differ from ASTRA by less than 0.8% and 2%", 1,
    float(max(abs(v) for v in ns(0.25, TWS)[:2]) < 0.8 and max(abs(v) for v in gs_(0.25, TWS)[:2]) < 2.0), 0)
chk("CP4", "text: over the first 0.4 m of the structure Slipstream's emittance differs from ASTRA's by -0.5 to +6.6%", 1, _near(ne(TWS, 1.6)[0], -0.5) * _near(ne(TWS, 1.6)[1], 6.6), 0)
chk("CP5", "text: beyond 1.6 m Slipstream's emittance is 0.2 to 1.6% above ASTRA's and GPT's 3.2 to 5.0% above", 1,
    _near(ne(1.6, 3.4)[0], 0.2) * _near(ne(1.6, 3.4)[1], 1.6) * _near(ge(1.6, 3.4)[0], 3.2) * _near(ge(1.6, 3.4)[1], 5.0), 0)
chk("CP6", "text: at the end of the line the beam size is -5.5% (Slipstream) and +5.5% (GPT) from ASTRA's", 1, _near(ns(1.6, 3.4)[2], -5.5) * _near(gs_(1.6, 3.4)[2], 5.5), 0)
_zn = np.array(_nat["z_m"])
_mn = (_zn > 1.2) & (_zn < 1.7)
_ma = (_ast[0][:, 0] > 1.2) & (_ast[0][:, 0] < 1.7)
_mg = _zg > 1.6
chk("CP7", "text: the output spacing is 2 mm for Slipstream, 10 mm for ASTRA and 6 cm for GPT", 1,
    _near(1e3 * np.median(np.diff(_zn[_mn])), 2.0, 0.15) * _near(1e3 * np.median(np.diff(_ast[0][_ma, 0])), 10.0, 0.2) * _near(1e2 * np.median(np.diff(_zg[_mg])), 6.0, 0.1), 0)
chk("CP8", "text: the GPT profile run's last snapshot reproduces the earlier default-run end emittance to 0.1%", 0, abs(_gp["design"]["consistency_with_earlier_default_run"]["eps_x"]["pct"]), 0.1)

_zz = np.linspace(0.25, 3.37147, 1400)
_spread = max(float((100 * (lambda P: (P.max(0) - P.min(0)) / P.mean(0))(np.array([np.interp(_zz, a[:, 0], a[:, c]) for a in _ast]))).max()) for c in (3, 5))
chk("CP9", "text: the three refined ASTRA runs differ by less than 0.4% in emittance and beam size (too narrow to draw as a band)", 0, _spread, 0.4)

rows, bad = [], 0
for cid, loc, quoted, actual, tol in checks:
    rel = abs(actual - quoted) / abs(quoted) if quoted else abs(actual - quoted)
    ok = rel <= tol or (tol == 0 and actual == quoted)
    bad += not ok
    rows.append(dict(claim_id=cid, location=loc, quoted=f"{quoted:g}", recomputed=f"{actual:.6g}", rel_diff=f"{rel:.3g}", tolerance=f"{tol:g}", status="ok" if ok else "MISMATCH"))
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "claims.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    if r["status"] != "ok":
        print(r)
print(f"{len(rows) - bad}/{len(rows)} checks ok")
sys.exit(1 if bad else 0)

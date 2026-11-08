# Code Locations and syntax for Benchmarking Studies

26 October study update: the current bounded campaign is implemented in `../study.py`, with generated results in [the benchmark study](../BENCHMARK_STUDY.md). It adds Ocelot's `sc_drift` adapter and HELIX's separate proton `linear_fodo` control. `run.py` now overrides the sibling editable Slipstream install with this workspace's `slipstream/src`; actual imported paths, versions and repository state are recorded per campaign. The installation recipes below retain their 22 October provenance. Archived 22 October rehearsals are excluded from the new report's numerical results.

Checked 2026-10-22 on host DDAST0117 (Ryzen 9 7950X, RTX 3050 8 GB, Windows 11 + WSL2 Ubuntu-22.04). "Adapter" is the worker in
`workers/` that drives the code for a benchmark case; `bench.toml` is the only file that holds machine paths for the
runner. Status: **ready** = an adapter ran end to end on the case(s) named; **probed** = the code starts and the syntax below was run, no adapter
yet. Comparison areas, cases and the campaign order are in the last sections.

| Code | Where it runs | Version checked | Adapter | Status |
| --- | --- | --- | --- | --- |
| Slipstream | Windows conda `astec-accelerator-models` (`CXX=clang-cl` for compile) | 0.1.0, `<GitHub>/slipstream` | `w_slipstream.py`, `w_slipstream_sc.py` | ready: `ring_tunes`, `ring_grad`, `sc_drift` |
| elegant | Windows `C:\Program Files\APS\Elegant x64\elegant.exe` | 2026.3.0 | `w_elegant.py` | ready: `ring_tunes`, `ring_grad` (finite difference) |
| Xsuite | WSL conda `astec-accelerator-models` | xtrack 0.111.6, xfields 0.27.2, xobjects 0.6.10 | `w_xsuite.py` | ready: `ring_tunes`, `ring_grad` (finite difference) |
| Cheetah | Windows conda `astec-accelerator-models` | 0.8.4 | `w_cheetah.py`, `w_cheetah_sc.py` | ready: `ring_tunes`, `ring_grad`, `sc_drift` |
| JuTrack | Windows, Julia 1.10.4 | JuTrack 1.5.0, commit 5b5eebd | `w_jutrack.py` + `jutrack_server.jl` | ready: `ring_tunes`, `ring_grad` (finite difference) |
| Bmad-X | WSL conda `bmadx` (Python 3.14, torch 2.14.1) | checkout `<GitHub>/Bmad-X` | `w_bmadx.py` | ready: `ring_tunes`, `ring_grad` |
| SciBmad | Windows conda env `scibmad` (`conda install -c bmad-sim scibmad`, Julia 1.12.7 distribution) | 0.6.0+26.10.14: BeamTracking 0.8.7, Beamlines 0.11.0, GTPSA 1.5.6 | `w_scibmad.py` + `scibmad_server.jl` | ready: `ring_tunes`, `ring_grad`, `ring_grad_b` (7 November 2026; replaces Bmad-X in the paper) |
| ImpactX | WSL conda `impactx-cpu-mpich-dev` | 26.10 | `w_impactx.py` + `impactx_run.py` | ready: `sc_drift` (CPU module) |
| ASTRA | Windows `Astra.exe`, WSL `/opt/ASTRA` | 4.0 (DESY 2022) | `w_astra.py` + `astra_deck.py` | ready: `sc_drift` |
| GPT | Windows `gpt.exe` | Oct 14 2024 build | `w_gpt.py` | ready: `sc_drift` |
| Ocelot | Windows and WSL conda `astec-accelerator-models` | 26.06.1 (Windows), 25.06.0 (WSL) | `w_ocelot_sc.py` | ready: `sc_drift` on Windows |
| HELIX | Windows conda `helix` | linac_gen 1.20.0 | `w_helix.py` | `linear_fodo` adapter; independent linear-limit control, not a PIC comparison |

A second elegant exists at `<simcodes>/Elegant/nt/elegant.exe` (2025.3.0). The paper
cites 2026.3.0, so `bench.toml` points at the APS install. Never mix the two inside one campaign.

## SciBmad

Replaces Bmad-X in the paper (7 November 2026): Bmad-X has had no commit since 20 June 2024 and one release (v0.1.0), SciBmad 0.6.0 was released on 14 October 2026.
Install with `conda create -n scibmad -c bmad-sim -c conda-forge scibmad`; the launcher is `envs/scibmad/libexec/scibmad/bin/julia.exe` (Julia 1.12.7, SciBmad,
BeamTracking, Beamlines, GTPSA, ForwardDiff and JSON are in the default environment, so no `--project` is needed). `bench.toml` sets `BENCH_SCIBMAD_JULIA`.
Results: `results/2026-11-07_scibmad/{ring_tunes,ring_grad,ring_grad_b,element_probes}`; `corrected_reference.py`, `gradient_acceptance.py` and
`element_attribution.py` read them next to the 26-28 October campaigns; `scripts/make_ol_fig2_fig4_scibmad.py` redraws Figs. 2 and 4.

- Syntax: `Beamline([Drift(L=..), Quadrupole(Kn1=.., L=.., tracking_method=BeamTracking.Symplectic(order=6, n_steps=8, fringe_at=BeamTracking.Fringe.NoEnd)),
  SBend(L=.., angle=.., e1=0, e2=0, tracking_method=BeamTracking.Exact(fringe_at=BeamTracking.Fringe.NoEnd))]; species_ref=Species("electron"), pc_ref=p0c)`;
  `track(line, v0=N x 6 matrix, n_turns=T-1)` returns `res.v[particle, coordinate, turn+1]` (the initial state is slice 1) and `res.state` (0x1 = alive).
  Coordinates are Bmad's (x, px, y, py, z, pz) with `px = Px/P0` and `pz = delta`: already the BSF.
- Rungs (`w_scibmad.py`): `std` is the shipped `SciBmadStandard()`, which applies **hard-edge fringe fields**, so it differs from elegant by 4e-3 in tune (a model
  difference, not an error); `sym` rungs use `Symplectic(order, n_steps)` on quadrupoles and sextupoles with the fringes off and exact sector bends.
  Order 4 with 4 steps already gives 9e-9 in tune; orders 4 and 6 with more steps converge to 6e-11 to 8e-11 (the reference's own accuracy).
- Julia 1.10.4 trap: installing SciBmad from the General registry fails (`KeyError: key "DifferentiationInterface" not found`, a Pkg bug with package
  extensions); use the conda distribution or a newer Julia.
- Derivatives: forward-mode `ForwardDiff.jacobian` through `track`. Knob families take their strength from the beamline context through `DefExpr(c -> c.knobN)`;
  assigning a `Dual` to an existing element field fails (`Float64(::Dual)`), and `v0` must be a matrix of the same (Dual) type as the parameters.
- GPU (7 November 2026): the distribution has no CUDA package. `benchmarks/codes/scibmad_gpu` is a Julia project holding `CUDA.jl 6.4.1` (runtime 13.4 artifacts,
  RTX 3050, sm_86); the SciBmad packages load from the distribution as standard libraries. `w_scibmad.py` starts Julia with `--project=codes/scibmad_gpu`,
  `BENCH_SCIBMAD_CUDA=1` and `JULIA_CUDA_HARD_MEMORY_LIMIT=4GiB` for `cuda-*` configs (`cuda-f64-o6k8`, `cuda-f32-o6k8`); tracking is `track(line, v0=CuArray(...))`.
  float64 on the GPU matches the CPU to 3e-12 (tune error 6e-11 against elegant); float32 gives 8e-4 (fails the 1e-5 gate; first turn 4e-6).
  The first GPU call costs about 2 minutes of JIT (60 s on a warm project). Timing is launch-bound: 5 particles x 1024 turns takes 56 s (f64) or 15 s (f32) against 1 s
  on one CPU thread. `scibmad_scaling.py` runs the 64-turn ladder in one process per device (results/2026-11-07_scibmad_scaling): warm call [s] at 5, 50, 500, 5000, 50 000
  particles, order 6 with 8 steps: CPU f64 0.14, 0.76, 7.6, 80, 654 (200 to 250 us per particle-turn: about 5 times Xsuite or compiled Slipstream); GPU f64 3.2, 3.2, 9.1, 10, 101 (31 us);
  GPU f32 0.92, 0.89, 1.1, 2.6, 18 (5.7 us). The GPU overtakes its own CPU at about 600 particles in f64 and about 50 in f32.
  Accuracy-matched rung, order 4 with 4 steps (tune error 9e-9; the one in the paper's Fig. 3): CPU f64 0.062, 0.19, 1.4, 14, 138 s (43 us per particle-turn, the same as Xsuite and compiled Slipstream);
  GPU f64 0.94, 0.98, 2.6, 4.7, 32 s (9.9 us); GPU f32 0.29, 0.35, 0.47, 1.6, 14 s (4.4 us); crossovers at about 1000 (f64) and 100 (f32) particles; the GPU f32 tune error is 1.5e-4 (fails 1e-5).
  `scibmad_scaling_analysis.py` writes data/2026-10-26_R1/scibmad_scaling.json (read by `scripts/make_ol_fig3_xsuite_gpu.py`; the GPU ring_tunes runs are in results/2026-11-07_scibmad_gpu).
- Not benchmarked: space charge (planned upstream).

## Benchmark standard frame (BSF)

Every adapter converts explicitly to `(x, px, y, py, delta)` with `px = Px/P0` (canonical, no vector potential), `delta = (p-p0)/p0`
and, where a case needs it, late-positive `tau`. Nothing is assumed to match by name. The harness audits lattice inventories where adapters expose them,
and the ring analyser reports initial transverse-state and first-turn component residuals. The tune gate does not also gate those map residuals, and these
checks are not a complete six-dimensional coordinate round trip. Independent finite-energy conversion controls and exploratory element probes supplement them.

| Code | Native coordinates | Conversion to BSF |
| --- | --- | --- |
| Slipstream | `(x, px, y, py, tau, delta)` | none (`RingMap` is the BSF) |
| elegant | `(x, xp, y, yp, t, p=beta*gamma)` slopes | `slopes_to_canon`, `delta = p/(p0c/m) - 1` |
| Xsuite | `(x, px, y, py, zeta, delta)`, `px = Px/P0` | none |
| Cheetah | `(x, px, y, py, tau, p)` with `p = (E-E0)/(p0c)` | `delta_to_cheetah_p` / `cheetah_p_to_delta` (cancellation-free form) |
| JuTrack | `(x, px, y, py, z, delta)`, AT convention | none; `Beam(energy=)` is **kinetic**, pass `E0 - m` |
| Bmad-X | `(x, px, y, py, z, pz)`, `pz = delta`, `px = Px/P0` | none; `z` is negative for late particles |
| ASTRA | file columns `x y z px py pz clock q species flag`: `p` in eV/c, `clock` in ns, `q` in nC; first line is the absolute reference particle, `z` and `pz` of the others are relative to it | `astra_deck.write_deck`: `p = beta*gamma * 511003.0 eV/c` (ASTRA's electron mass) |
| GPT | `x y z Bx By Bz` (`B` = beta*gamma), `t`; reads the ASTRA file after `astra2gdf` | none; GPT's mean kinetic energy comes out 2.43 keV below the 5 MeV input, constant along the drift (origin not identified: not the 511003.0 versus CODATA mass, which would shift it by a few eV), so compare GPT's energy *changes*, not its absolute energy |
| ImpactX | at fixed s: `(x, y, t)`, `(px, py, pt)` with `t = c*time`, `pt = -gamma`, momenta relative to the reference `beta*gamma` | `with sim.beam.at_fixed_t():` takes lab-frame `x, y, z` and `(bg - bg0)/bg0`; outputs `sigma_z = beta_ref * sigma_t`, `dE = -dpt * (beta*gamma)_ref * mc2` |

## Slipstream

`<GitHub>/slipstream` (editable install `slipstream-accel`). Ring: `slipstream.tracking.native.ring.RingMap(specs,
energy_eV=, mc2_eV=, device=, dtype=, n_kicks=, n_kicks_bend=, integration_order=, compile=)`; `ring.track(state, turns, aperture=,
record_every=1)` records the state **after** each turn. Injector: `Injector.from_astra(deck).track(device=, dtype=)`. Set `CXX=clang-cl`
before any Windows `compile=True` run. Run space-charge-dominated cases in WSL at float32 (the project norm); ring cases may use either.

## ASTRA

`ASTRA` and `generator` are in Windows at `<simcodes>/ASTRA/nt` (`Astra.exe`,
`generator.exe`), and in WSL at `<home>/GitHub/simcodes/SimCodes/ASTRA/linux`; the WSL parallel wrapper is
`/opt/mpich-5.0.1/bin/mpiexec -np N /opt/ASTRA/astra_MPICH2.sh deck`.

- Syntax: run `Astra.exe deck.in` from the deck's own directory (outputs `deck.Xemit.001`, `.Yemit`, `.Zemit`, `.ref`, `.log`, `.track.001`).
  With no argument it prompts for a project name and blocks, so always pass the deck.
- `generator.exe gen.in` blocks forever unless the namelist has `Lprompt = False`; run it from a scratch directory (it writes `NORRAN`
  and the output in the cwd).
- Existing CLARA decks with cached ASTRA reference output: `slipstream_dev/examples/clara_linac/ASTRA/{5pC,20pC,100pC,250pC}/injector400.in`.
  Check the cached `.Xemit.001` before rerunning.
- Free-space deck used by `sc_drift` (`workers/astra_deck.py`): `&NEWRUN` with `Distribution='beam.ini'`, `Auto_phase=F`, `H_max=H_min=<ns>`; `&OUTPUT` with
  `Zemit=<n>` (equal z spacing between `ZSTART` and `ZSTOP`), `EmitS=T`, `RefS=T`; `&CHARGE` with `LSPCH=T`, `Lmirror=F` (no cathode). `Lmagnetized`, `Zstart` and `Zstop`
  belong to `&OUTPUT`, not `&NEWRUN`. Output tables: `.Xemit.001` / `.Yemit.001` (z, t, mean x, rms x [mm], rms x' [mrad], normalised emittance [pi mm mrad], correlation) and
  `.Zemit.001` (z, t, Ekin [MeV], rms z [mm], rms dE [keV], emittance, correlation).
- Space charge is the `&CHARGE` namelist: `LSPCH`, `LSPCH3D`, `nrad`, `nlong_in`, `Lmirror` (cathode image), `cell_var`, `min_grid`, `max_scale`.
  The ASTRA electron mass is 511003.0 eV, not CODATA. Time step is `h_min = h_max` in `&NEWRUN`.

## GPT (General Particle Tracer)

`C:\Program Files\General Particle Tracer\bin\gpt.exe` (Oct 14 2024 build) with `gdfa.exe`, `gdf2a.exe`, `gdfselect.exe`, `asci2gdf.exe`,
`astra2gdf.exe`.

- Syntax: `gpt.exe [-hHvs] [-j nCPU] [-o outfile.gdf] [-a outfile] input.in [var=value]`; the licence is passed as `GPTLICENSE=...`
  on the command line (a working line is in the local `examples/RUEDI/GPT/delta.bat`; do not copy the key into this repository).
- Run from a scratch directory with the long `<home>/...` path. The 8.3 alias (`JKJ62~1.CLR`) breaks PowerShell
  `Set-Location` and GPT then writes into the wrong cwd.
- Particles from ASTRA format: `astra2gdf.exe -o beam.gdf beam.ini`, then in the input `setfile("beam","beam.gdf"); settdist("beam","u",0,0);` (the manual's recipe: ASTRA writes both
  time and position, and the second line keeps the positions at one fixed time). Snapshots: `tout(t0, t1, dt)`. Reduce with
  `gdfa.exe -o a.gdf out.gdf time avgz stdx stdy stdz nemixrms nemiyrms avgG stdG` and print with `gdf2a.exe a.gdf` (whitespace table, first line the names). Space charge: `spacecharge3Dmesh();`
  (mesh solver) or `spacecharge3D();` (point-particle tree code with an opening angle, theta = 0 means all N^2 pairs; it stops with `Particle plummer radius must be positive` unless `setrmacrodist("beam","d",R);` is given, R <= 1e-15 m is the manual's safe value); `accuracy(n)` sets the integration accuracy. The licence is the user-level environment variable `GPTLICENSE`, passed as `GPTLICENSE=$value`
  on the command line; the adapter never writes it to a result file. Full `tout` output of a 20000-particle, 61-snapshot run is 185 MB, so the adapter keeps only the reduced table.
- Read GDF with `easygdf.load` (conda `astec-accelerator-models`); blocks named `position` hold one screen each.
- `qe` is negative. `Bmultipole` sextupole is `B_y = H (x^2 - y^2)`. `spacecharge3Dmesh` underestimates the space-charge expansion of a
  very short bunch by about 12 % in sigma_x against GPT's own point-particle `spacecharge3D`; ASTRA and the native injector agree with the
  point-particle result. Record which solver a reference used.

## Elegant

APS install `C:\Program Files\APS\Elegant x64\elegant.exe` (2026.3.0, used by the campaign). Needs `RPN_DEFNS=C:\Program Files\APS\Elegant x64\defns.rpn`
or it stops with `error: RPN_DEFNS environment variable undefined`. Pelegant: `C:/Program Files/Microsoft MPI/Bin/mpiexec.exe -n 16
Pelegant.exe run.ele`; it cannot read an ASCII `&sdds_beam` (use binary SDDS or `&bunched_beam`).

- Syntax: `elegant.exe run.ele` with `&run_setup`, `&run_control n_passes=`, `&sdds_beam input="beam.sdds", input_type="elegant"`, `&track`.
  Turn-by-turn data come from a `WATCH,MODE="coordinates",INTERVAL=1` element, read with `sdds2stream run.w1 -col=particleID,x,xp,y,yp,p`.
  Rows are in pass order and **not** in particle order within a pass; sort by `particleID`.
- Particle files: `plaindata2sdds beam.txt beam.sdds -inputMode=ascii -outputMode=binary -noRowCount -column=x,double,units=m ...`.
- Slicing knobs: `N_SLICES` and `INTEGRATION_ORDER` on `KQUAD`/`KSEXT`/`CSBEND`. Elegant's default slicing is not converged for this ring.
- `RFCA`/`RFCW` comparisons must set `N_KICKS=1` (the default uses a matrix method). Source trees for primary-source checks:
  `<neg_coating_rwi>/elegant/elegant/src\`.

## XSuite

WSL, conda env `astec-accelerator-models` (the separate `xsuite` env is 0.86 and CPU-only, so it is not used). Source checkouts in
`<home>/GitHub/xsuite` (`xfields`, `xobjects`, `xpart`, `xsuite`, `xtrack`); the env holds xtrack 0.111.6, xfields 0.27.2,
xobjects 0.6.10 and cupy 14.2.0.

- Ring syntax used: `xt.Line(elements=, element_names=)`, `line.particle_ref = xt.Particles(p0c=, mass0=)`, `line.build_tracker(_context=xo.ContextCpu(omp_num_threads=1))`
  or `xo.ContextCupy()`, `line.build_particles(x=,px=,y=,py=,delta=,zeta=)`, `line.track(p, num_turns=, turn_by_turn_monitor=True)`,
  `line.record_last_track`. The monitor holds the state at the **start** of turns 0..T-1.
- Elements: `Quadrupole(..., model="drift-kick-drift-exact", integrator="yoshida4", num_multipole_kicks=)`, `Sextupole(...)`,
  `Bend(..., model="bend-kick-bend", edge_entry_active=False, edge_exit_active=False)`.
- Traps: with a cupy context `record_last_track` arrays can already be host numpy, so convert with `a.get() if hasattr(a, "get") else np.asarray(a)`
  (the old `nparray_from_context_array` call failed with `AttributeError: 'numpy.ndarray' object has no attribute 'get'`).
  Kernel compilation on CUDA warns `nvrtc version < 12.9` and took about 400 s for a first process, so report it as cold cost.
- Finite differences must not rebuild the tracker: every `build_tracker` recompiled C kernels (a 40-point finite-difference Jacobian had not finished after 13 minutes). `fd_update` in
  `w_xsuite.py` sets `line.element_dict['<name>_<i>'].k1` / `.k2` in place instead (the harness's `fd_update` hook); the job then took 90 s.

## Cheetah

Windows and WSL conda `astec-accelerator-models`, cheetah 0.8.4. Adapter uses `cheetah.Segment` of `Drift`, `Quadrupole(num_steps=)`,
`Sextupole`, `Dipole` with `tracking_method` in `linear`, `second_order`, `drift_kick_drift`. Sextupoles only offer `linear` and
`second_order`. `ParticleBeam(particles[N,7], energy=)` with the last coordinate `p = (E-E0)/(p0c)` (see the BSF table).

- Cost: `drift_kick_drift` is a Python loop per element per step; measured on the 104-element ring about 0.07 s per turn on CPU at 4 steps
  (about 70 s for 1024 turns) and about 0.3 s per turn on CUDA at 4 steps. A job with `dri16` runs for many minutes, so give it a long `--timeout`.
- The linear one-turn path (`lin1`) is a merged matrix and takes about 1 s; it has no amplitude dependence, so it cannot pass a tune test
  of a sextupole ring.
- Autograd: with `tracking_method="second_order"` the Jacobian of the tracked state with respect to magnet strengths is **NaN from about 10 turns** on this ring (anomaly detection:
  `BmmBackward0` returned nan in `Element._track_second_order`'s `einsum`); a single quadrupole is fine except at exactly `k1 = 0` (NaN gradient). `drift_kick_drift` and `linear` give finite gradients
  for 39 turns, so the `ring_grad` Cheetah configuration is `cpu-f64-dri4`. The `second_order` NaN is unresolved (not checked against a newer Cheetah).
- Space charge (`w_cheetah_sc.py`): `Drift` / `SpaceChargeKick(effect_length, grid_shape, grid_extent_x/y/tau)` / `Drift`, a momentum-only integrated-Green-function kick. Particles are
  `ParticleBeam(particles[N,7], energy, particle_charges=)`. In float32 on CUDA the longitudinal kick vanishes (the mean energy stays exactly at its initial value), a float32 resolution
  limit of the `p` coordinate; use float64 for energy-spread comparisons.

## JuTrack

`<GitHub>/JuTrack.jl` (JuTrack 1.5.0, `Manifest.toml` in the checkout). Julia 1.10.4 at
`<programs>/Julia-1.10.4/bin/julia.exe`. An Accelerator-Toolbox-style code: 4th-order symplectic multipole pass
(`StrMPoleSymplectic4Pass!`), AT bend pass, automatic differentiation through its own `DTPSAD` type.

```julia
using Pkg
Pkg.activate("<GitHub>/JuTrack.jl")
Pkg.instantiate()
using JuTrack
```

- Ring syntax used: `DRIFT(name=, len=)`, `KQUAD(name=, len=, k1=, NumIntSteps=)`, `KSEXT(name=, len=, k2=, NumIntSteps=)`,
  `SBEND(name=, len=, angle=, e1=0, e2=0, NumIntSteps=, FringeBendEntrance=0, FringeBendExit=0)`, `Beam(r; energy=E_kinetic, mass=)`,
  `ringpass!(line, beam, nturns, true)` returns copies of `r` after each turn.
- The default `SBEND` fringes are on (`FringeBend* = 1`); switch them off to compare with `e1 = e2 = 0` sector bends.
- Python is not wired to Julia (`juliacall` is not installed), so `w_jutrack.py` keeps one Julia process alive and talks to it over stdin/stdout
  (`jutrack_server.jl`). `import_s` is the package load, `cold_s` includes Julia's JIT.
- Package noise on stdout is ignored; the server prefixes its replies with `@@`.

## Bmad-X

WSL conda env `bmadx` (Python 3.14.9, torch 2.14.1, pytao 1.3.3); source `<home>/GitHub/Bmad-X` (Windows `<GitHub>/Bmad-X`).
The torch path is `bmadx.bmad_torch.track_torch`: `TorchDrift(L)`, `TorchQuadrupole(L, K1, NUM_STEPS)`, `TorchSextupole(L, K2, NUM_STEPS)`,
`TorchSBend(L, P0C, G, FRINGE_AT=, FRINGE_TYPE=)`, `TorchLattice(elements)`, `Beam(data[N,6], p0c, mc2=)`.

- Model facts read from the source: drift and sector-bend body are exact; the quadrupole body is an exact linear map per step; the sextupole is
  **one explicit-Euler kick-drift per step** (first order, not symplectic), so the sextupole step count is the convergence knob and the cost is a Python
  loop over steps. It has no thick octupole, no solenoid and no space charge.
- Use `FRINGE_TYPE="none"` to match `e1 = e2 = 0` sector bends.

## ImpactX

WSL conda env `impactx-cpu-mpich-dev` (impactx 26.10, Python 3.12, cupy 14.2.0). Binaries `/opt/impactx/bin/impactx_cuda` (CUDA+OpenMP) and
`impactx_omp` (OpenMP only); the Python module `impactx` is in the env. Worked examples (with `README.rst`, `run_*.py` and `input_*.in`) are bundled at
`$CONDA_PREFIX/lib/python3.12/site-packages/impactx/examples/`: `expanding_beam`, `fodo_space_charge`, `space_charge_gaussian_kick`, `chicane`,
`compression`, `iota_lattice`, `symplectic_integration`, `initialize_from_array`, `epac2004_benchmarks`, `solenoid`, `rfcavity`.

- Python syntax (run in `workers/impactx_run.py`, one process per run because AMReX initialises once per process): `sim = ImpactX()`, set `sim.max_level`, `sim.n_cell`, `blocking_factor_*`
  and `max_grid_size_*` (`max_grid_size` must be divisible by the blocking factor, else `RuntimeError: max_grid_size not divisible by blocking_factor`), `sim.space_charge = "3D"`,
  `sim.poisson_solver = "multigrid"`, `sim.prob_relative`, `sim.slice_step_diagnostics = True`, then `sim.init_grids()`; **only after that** does `sim.beam` exist (touching it earlier
  segfaults: `Segmentation fault (core dumped)`). Reference particle: `sim.beam.ref.set_species("electron").set_kin_energy_MeV(E)`. Particles from arrays:
  `with sim.beam.at_fixed_t(): sim.beam.add_n_particles(dx, dy, dz, dpx, dpy, dpz, qm_eev, bunch_charge=Q)` with momenta relative to the reference `pz`.
  Lattice: `sim.lattice.extend([elements.Drift(name="d", ds=0.05, nslice=k), ...])`; `sim.track_particles()`; `sim.finalize()`.
- This Python build has **no FFT support**: `sim.poisson_solver = "fft"` fails at the first kick with `Must compile with FFT support to use the IGF solver!`. Use `multigrid` (mesh
  refinement through `max_level = 1`, `prob_relative = [3.0, 1.1]`) or `space_charge = "Gauss3D"`. The CUDA build is the input-file executable `/opt/impactx/bin/impactx_cuda`; whether it
  has FFT support has not been checked.
- Moments at every slice end are written to `diags/reduced_beam_characteristics.0.0` (header line then rows; columns include `s`, `sigma_x`, `sigma_y`, `sigma_t`, `sigma_pt`, `mean_pt`,
  `emittance_xn`, `emittance_yn`) and the reference to `diags/ref_particle.0.0`.

## Ocelot

Windows and WSL conda env `astec-accelerator-models` (Windows ocelot-collab 26.6.1, WSL 25.06.0 - different versions; a campaign must name which one it used).
Importing prints `module NUMBA is not installed` on Windows. Ocelot has tracking with `Navigator` plus `track`, second-order and kick transfer maps,
space charge (`SpaceCharge`) and CSR (`CSR`) physical processes. Use it only where an overlap exists (see the area matrix); it is not a ring-tune comparator here.

## HELIX

Windows conda env `<miniconda>/envs/helix` (python 3.12.15, torch 2.14.1+cu132, CUDA available); package `linac_gen` 1.20.0 from
`<GitHub>/HELIX`. A TraceWin-compatible hadron-linac envelope and multiparticle code with a 3-D PIC solver and a differentiable demo
(`examples/differentiable_demo.py`). Its scope is proton RFQ and linac work, so this campaign uses a separate proton linear FODO control. Its differentiable PIC capability remains unmeasured; it is not a ring or electron-injector reference.

## Comparison areas that make sense for Slipstream

Slipstream's claims are (1) a differentiable, GPU, PyTorch ring map for nonlinear studies, (2) single-pass transport and (3) an electron photoinjector with
space charge. Areas outside those (spin, beam-beam, electron cloud, hadron RFQ, a full MAD/Bmad language) are not benchmarked. Gate numbers follow
`../../PLAN.md` (Sec. 4).

| Area | Gate | Case id | Reference | Comparators | Observables | Status |
| --- | --- | --- | --- | --- | --- | --- |
| A1 map conventions, first turn | G1 | `ring_tunes` (first-turn residual) | elegant finest | Xsuite, Cheetah, JuTrack, Bmad-X | per-component residual of `(x,px,y,py)` after one turn | adapters ready |
| A2 amplitude-dependent tunes | G2 | `ring_tunes` | elegant finest; spread of all finest rungs | Xsuite, Cheetah, JuTrack, Bmad-X | absolute tune error, cheapest passing configuration | adapters ready |
| A3 off-momentum tunes, chromaticity, DA | G2 | `ring_da` | elegant | Xsuite, JuTrack | DA area and boundary, per-launch loss agreement | planned |
| A4 long-run precision | G3 | `ring_tunes` with `turns = 1e5` | CPU float64 Slipstream | Slipstream float32/double-single, Xsuite | tune and invariant drift | planned |
| A5 gradients | G11, G12 | `ring_grad` | Richardson-extrapolated central difference; elegant finest for cross-code | Slipstream, Cheetah, Bmad-X (autograd); elegant, Xsuite, JuTrack (finite difference only; JuTrack's DTPSAD path is not wired) | Jacobian error against own finite difference and against the reference, forward, finite-difference and AD time | adapters ready |
| B1 space-charge field | G4 | `sc_field` | analytic Gaussian (closed form) | Slipstream, Cheetah, ImpactX | field error vs grid | planned |
| B2 free-space space-charge expansion | G4, G6 | `sc_drift` | ASTRA finest; GPT as the second reference | Slipstream, Cheetah, ImpactX, GPT | `sigma_x(z)`, `sigma_z(z)`, `eps_n(z)`, mean energy, energy spread at 20 stations | adapters ready |
| B3 CLARA gun and booster, 5/20/100 pC | G5 | `clara_inj` | ASTRA (cached and live), GPT where a deck exists | Slipstream | energy, sizes, projected and slice emittance along the line | decks in repository |
| B4 differentiable PIC at fixed energy | G12 | `sc_grad` | finite difference | Slipstream, HELIX, Cheetah | gradient time to a declared accuracy | optional |
| C1 cavity and linac transport | G7 | `linac_rf` | elegant `RFCA` with `N_KICKS=1` | Slipstream, Ocelot, ImpactX | energy gain, phase slip, jitter response | planned |
| C2 chicane and CSR | G8 | `chicane_csr` | elegant (CSR off, then on) | Slipstream, ImpactX, Ocelot | covariance and emittance at bend boundaries | planned |

A1, A2, A5, B2 and the separate analytic proton FODO are runnable cases (`ring_tunes`, `ring_grad`, `sc_drift`); the others follow the same worker pattern. Case files are in `../cases/`; `make_cases.py` freezes the lattice and the 20000 macroparticles.

## Running a campaign

Interpreter: the Windows one in `bench.toml` (Python >= 3.11, because `run.py` reads `bench.toml` with `tomllib`). One command per case; the runner starts each
code's worker in its own environment, shuffles all `(code, config, process)` jobs with a fixed seed, writes
`results/<campaign>/<case>/<code>/<config>/p<k>/{job,result,wall}.json` plus `tbt.npz` (or `grad.npz`), skips jobs that already have an `ok` result (an interrupted campaign
resumes), and records git state, CPU and GPU in `results/<campaign>/environment.json`. `--configs` takes `finest`, `all`, or comma-separated globs; a code with no worker for the case is
reported as an explicit configuration error, and a job that exceeds its code's `timeout` in `bench.toml` is recorded as a failure.

```powershell
$py = "<conda>/envs/astec-accelerator-models/python.exe"
$env:CXX = "clang-cl"
& $py run.py --case ring_tunes --campaign R1 --codes slipstream,elegant,xsuite,cheetah,jutrack,bmadx --configs all --procs 3 --repeats 3 --threads 1
& $py run.py --case ring_grad  --campaign R1 --codes slipstream,elegant,xsuite,cheetah,jutrack,bmadx --configs "cpu-f64-o6k8b16,cpu-f64-o6s16b32,cpu-f64-yosk8b16,cpu-f64-n16b32,cpu-f64-q1s64,cpu-f64-dri4" --procs 3 --repeats 3
& $py run.py --case sc_drift   --campaign R1 --codes astra,gpt,impactx,cheetah,slipstream --configs all --procs 3 --repeats 3
& $py analyse.py      --campaign R1 --case ring_tunes
& $py analyse_grad.py --campaign R1 --case ring_grad
& $py analyse_sc.py   --campaign R1 --case sc_drift
```

Rules for the real campaign (the comparison protocol in `../../PLAN.md`, Sec. 5): do not start anything else on the host while it runs (the first `demo_R1` was started twice and its jobs shared the CPU); keep
`--procs` independent processes so that import, JIT and cache effects are in the numbers; do not tune a configuration after seeing the reference; keep failed and timed-out jobs in the tables.
Expect the slow rungs to dominate the wall time: Xsuite on CUDA spends about 400 s in `nvrtc` in its first process, Cheetah `drift_kick_drift` and Bmad-X sextupole steps are Python loops
(minutes per call), and Slipstream on CPU float64 takes minutes per 1024-turn call, so the CPU float64 rungs are the converged reference, not the cost comparator.

## Historical rehearsals

The excluded 22 October rehearsal notes are preserved in [REHEARSALS_2026-10-22.md](REHEARSALS_2026-10-22.md). Use the current [benchmark study](../BENCHMARK_STUDY.md) for measured results.

## Traps found while building the adapters

- Git Bash mangles `/mnt/...` arguments to `wsl.exe`; prefix with `MSYS_NO_PATHCONV=1`, and put multi-line shell in a `.sh` file instead of `bash -c`. `conda activate` inside a
  `| head` pipeline runs in a subshell and does nothing.
- Julia, Xsuite CUDA and torch.compile pay a large first-process cost; the harness reports `import_s`, `build_s`, `cold_s` and `warm_s` separately and the analysis must not mix them.
- A wall-clock timeout on the runner (`--timeout`, default 3600 s) turns a slow rung into a recorded failure; that is data, not noise.

## Step-model study: order, per-kind slicing and a matrix-kick quadrupole (7 November 2026)

Why: Fig. 2(a)'s SciBmad point (order 4, four steps, 9e-9 in 1.0 s on one CPU thread) is more accurate per second than any Slipstream GPU rung at five particles.

- **GPU timings come from Warp.** `RingMap` on CUDA uses the packed Warp kernel (`ring_warp.py`) when Warp is installed (1.17.0 here), so GPU time scales with stages per particle, not launches.
- **Order 8 does not move the trade-off** (`results/2026-11-07_slipstream_o8`): its Yoshida scheme is a triple jump of order 6 (21 kicks per step), and at equal cost order 4/6/8 lie on one cost-versus-error curve, CPU and GPU, float64 and float32. Float32 has a tune floor of about 1.5e-6 (gate 1e-5); double-single (hybrid) costs about the same as float64 here and floors near 1.5e-9.
- **Why SciBmad is ahead at order 4** (`results/2026-11-07_scibmad_attribution`, SciBmad variants `symy`/`dk`/`dky` in `scibmad_server.jl`): drift-kick quads with the Yoshida triple jump (`dky`) reproduce Slipstream's order 4, 4 steps exactly (7.46e-5); the optimized (Suzuki, 5-stage) composition alone gives 3.0e-6, the matrix-kick quad alone 9.3e-8, both 8.9e-9.
- **Matrix-kick prototype** (`prototypes/ring_warp_mkm.py`, a patched copy of the package kernel; `prototypes/mkm_prototype.py`; `results/2026-11-07_mkm_prototype`): exact paraxial quadrupole matrix over each Yoshida stage, bracketed by the exact-minus-paraxial drift over half the stage (BeamTracking's `mkm_quadrupole!`), path length into tau. One quad, one step: 1e-12 against a fine reference (drift-kick: 1e-7); ring order 4, 4 steps reproduces SciBmad `symy` (9.34e-8). A matrix-kick stage costs about 2.5 times a drift-kick stage in float64 (RTX 3050, FP64 at 1/64 rate), and float32 rounding in the matrix hurts (floor 3e-6 to 3e-5), so it is only level with well-sliced drift-kick.
- **Per-kind slicing is the lever that works today.** The sextupoles need one step; the quadrupoles carry the steps. Order 6, 8 quad steps, 1 sextupole step: same tune error as the uniform 8 steps (1.7e-9) in 2.07 s against 2.99 s (float64); order 6, 4 quad steps, 1 sextupole step (float32): 1.3e-6 in 0.32 s against 0.66 s. At 50 000 particles and 64 turns: float64 3.6 s against 5.3 s, float32 0.23 s against 0.50 s. `RingMap` has one `n_kicks` for every non-bend element; a per-kind (or per-element) step count is the missing option.
- Preview figures (not in Overleaf): `figures/ol_o8_preview/fig2_preview.pdf` (Fig. 2(a) plus a trade-off panel (b)) and `fig_step_schematic.pdf`; scripts `scripts/make_fig2b_tradeoff_preview.py`, `scripts/make_fig2_step_schematic_preview.py`.

## Reading the first results (dress rehearsals, not results)

These runs were made on 2026-10-22 only to prove the pipeline; they used one process and one repeat, and other work shared the host. Do not quote them.

- `ring_tunes` (campaign `R0`, 1024 turns, 5 amplitudes, reference `elegant/cpu-f64-o6s16b32`, tolerance 1e-5 in absolute tune): elegant is converged (`o6s8b16` differs from `o6s16b32` by 4e-9; the
  default-like `o2s4b8` is 2.6e-2 off). Slipstream agrees with it to 2.0e-6 (CUDA float64) and 3.8e-6 (CUDA float32) from the `o6k8b16` rung on, and its cheapest passing rung (`cuda-f32-o6k8b16`,
  0.66 s warm per call) costs about the same as elegant's (`o6s8b16`, 0.71 s) at this size. The other codes converge, if at all, to tunes that differ from elegant's: Xsuite 3.0e-4 (self-converged to
  6e-6), JuTrack 4.4e-5 (self-converged to 3e-7), Bmad-X 1.3e-4 (still moving by 2e-6 between its last two rungs), Cheetah `drift_kick_drift` 1.0e-4 (not self-converged). First-turn map residuals
  against elegant at the finest rungs are 2e-7 (Slipstream) to 2e-5 (JuTrack) in position and 1e-7 to 5e-6 in momentum, none near the proposed 1e-8. Whether the differences are model choices
  (JuTrack's AT-style bend, Bmad-X's Euler sextupole, Cheetah's `second_order` sextupole, Xsuite's bend model) or adapter errors has **not** been separated; do that with single-element probes before
  quoting any of these numbers.
- `ring_grad`: Slipstream's and Bmad-X's reverse-mode Jacobians agree with their own Richardson-extrapolated central differences to 4e-8 and 4e-9 (Frobenius norm). On CPU float64 with 5 knobs the
  AD Jacobian is not faster than finite differences for Slipstream (52 s against 43 s; its eager CPU path is dispatch-bound) and only 1.2x faster for Bmad-X; Cheetah's `drift_kick_drift` path (the one configuration with a working AD, agreeing with its own finite difference to 4e-9) takes 26 s against 22 s for finite differences. The `ring_grad` command lists the
  converged rung of each code explicitly, because the glob `cpu-f64-*` would also run Cheetah's NaN `second_order` path and its multi-hour `dri64`. The break-even needs more knobs or the GPU, which is a campaign question.
- `sc_drift` (campaign `R0`, 5 MeV, 20 pC, 20000 frozen macroparticles; sigma_x grows from 0.50 to about 0.85 mm and the normalised emittance from 0.30 to about 0.59 um over 3 m; reference
  `astra/cpu-f64-3d32h1.4`, the 3-D solver; tolerances 3 % size and length, 5 % emittance): ASTRA's own settings spread by 0.8-1.1 % in size and 2.5-3.1 % in emittance (its axisymmetric and 3-D solvers
  differ by 3 % in emittance). Slipstream's 3-D solver agrees with ASTRA's 3-D solver to 0.06 % in size and 0.07 % in emittance (float64, CPU), its axisymmetric solver is within 1.1 % and 2.8 % at the
  finest mesh. GPT's mesh solver is within 0.7 % and 2.5 %; its point-particle tree code (`spacecharge3D`, 600 s per call) is within 1.2 % in size, 5.2 % in emittance and 0.6 % in energy spread. ImpactX multigrid passes from 32x32x40 (0.4 % and 0.8 %; the 16x16x20 mesh is 4.6 % and 10 % off) and its `Gauss3D` kick fails the emittance (23 %).
  Cheetah passes from a 32^3 grid. Energy spread at the last station: GPT's mesh solver is 6.9 % below ASTRA's 3-D value (6198 against 6655 eV; the tree code is within 0.6 %, so the deficit belongs to the mesh solver, as the local notes found), ImpactX multigrid differs by 7-9 %,
  Slipstream's cylindrical solver by 5.8-8.4 % and its 3-D solver by 0.9 %, Cheetah in float64 by 0.2-0.8 %;
  Cheetah float32 on CUDA shows no energy gain (the longitudinal kick is below the float32 resolution of its `p` coordinate: the float32 force-deletion trap recorded in Slipstream's `CLAUDE.md`).
  Cost at the cheapest passing setting: Slipstream CUDA float32 1.3 s, Cheetah CUDA float32 1.7 s, GPT 2.8 s, ImpactX CPU 4.4 s, ASTRA 22 s (Slipstream and Cheetah on the RTX 3050; GPT, ImpactX and ASTRA on one CPU thread; 20000 particles, 20 stations, one process, one repeat).


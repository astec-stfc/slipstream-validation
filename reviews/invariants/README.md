# invariants review

Conditions that a lossless ring map must satisfy, checked on Slipstream alone: no other code and no archived number, so they cannot
agree with a reference that has the same error.

```bash
SSV_SLIPSTREAM=../slipstream pytest reviews/invariants -m "not slow"   # about 15 s on a CPU
SSV_SLIPSTREAM=../slipstream pytest reviews/invariants                 # with the 100 000 turn runs, a few minutes more
```

## Coordinates

The state is `(x, px, y, py, tau, delta)`. The partner of `tau` in the symplectic form is minus `E / (p0 c)`, so the Jacobian is
taken to those variables with `d(E / p0 c) / d(delta) = beta(delta)` before `M^T J M - J` is formed. In `(tau, delta)` as they are,
a correct drift at 5 MeV misses the condition by 2e-6.

## Gates

| Quantity | Elements | Limit | Kind |
| --- | --- | --- | --- |
| largest element of `M^T J M - J`, and `det M - 1` | drift, quadrupole, sextupole, octupole, skew and thin multipoles, misaligned quadrupole, sector bend (exact and split), RF folded to a drift, corrector at zero kick; 3 GeV and 5 MeV | 1e-12 | target |
| the same | the 104-element benchmark ring at the paper's setting | 1e-12 | target |
| the same, on the reference orbit | edge bend, combined-function bend, deflecting cavity | 1e-12 | target |
| the same, off axis | edge maps (elegant's slope-space map), deflecting cavity (symplectic with `beta0` in place of `beta(delta)`) | per row in `RATCHET` | ratchet |
| sum of `log det M` along an orbit; symplectic error at every turn | bends, sextupole, octupole; 2000 and 100 000 turns | 1e-14 per turn; 1e-12 | target |
| fitted drift of the Courant-Snyder invariant of the tracked orbit | linear lattice; 2000 and 100 000 turns | 1e-10 over the run | target |

## Known failures (strict `xfail`: a fix shows up as a failure of the mark)

Measured on 28 November 2026 at Slipstream `86f31e6`, CPU float64, one element of 1 m with 0.2 rad.

| Element | Symplectic error | What was checked |
| --- | --- | --- |
| bend with `k1 = 0.4` | 8.4e-5 at y = 1 mm | zero at y = 0; proportional to y; the same with 8, 32 and 128 slices and orders 4, 6 and 8; zero for a pure bend and for a straight quadrupole. The curl of the kick equals `-ds h Fx` |
| corrector with 1 mrad kick, 0.2 m / thin | 1e-4 / 1.5e-6 on the reference orbit | zero without a kick |

The determinant is 1 to round-off for the combined-function bend, so a volume check alone does not see it. The frozen benchmark
cases have no combined-function bend and no corrector. Lattices that have them, such as a booster with combined-function magnets,
carry the error in their amplitude-dependent behaviour; how large the effect is on dynamic aperture or frequency maps has not been
measured.

## Not covered

Maps with collective kicks, radiation damping or an aperture are not lossless Hamiltonian maps and are outside this review. RF cavities
in `NativeTracker` and the linac tracker are not covered yet.

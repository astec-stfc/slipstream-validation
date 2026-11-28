# edges review

Slipstream near the places where its models are weakest. Each reference is a closed form, a quadrature or dense sampling; no other
code is needed.

```bash
SSV_SLIPSTREAM=../slipstream pytest reviews/edges    # under a minute on a CPU
```

## Gates and what they found

Measured on 28 November 2026 at Slipstream `86f31e6`, CPU float64. A target is what the physics should give; a ratchet records
today's accuracy so that a loss of it is noticed.

| Regime | Reference | Result | Gate |
| --- | --- | --- | --- |
| electrons at rest on the cathode, transverse field on the plane | zero, by the image | round-off | target 1e-12 |
| the same, axial field at the centre, 3-D solver, slab 1 um to 1 mm | uniform cylinder with image | 2.3 % to 3.9 % low at 32 cells, about half that at 64 | ratchet 5 to 8 %; converges |
| the same, default cylindrical mesh, slab 10 um to 1 mm | the same | 0.3 % to 13 %, not monotonic | ratchet per thickness |
| the same, default cylindrical mesh, slab 1 um | the same | **85 % low** | strict `xfail` |
| the same with `sigma_z_floor_m` lowered to 0.5 um | the plane stays an equipotential | the adaptive mesh exceeds its cap and **drops the image without a message** | pinned (documented behaviour) |
| CSR of a bunch with a spike, grid kernel, 256 to 2048 bins | 1-D steady-state quadrature | rms 4.4 % at 256 bins, 0.09 % at 2048 | target: converges; ratchets |
| the same, pairwise kernel (`fast_collective_effects=False`, the `LinacTracker` default) | the same | **rms 24 %, largest 63 %** at 3000 particles, 22 % and 57 % at 10 000 | ratchet |
| the same random particles, grid kernel at 256 bins | the same | rms 5.5 %, largest 15 % at 3000; 3.8 % and 11 % at 10 000 | ratchet |
| steps that graze apertures, plugs, slits and scrapers, also placed ones | dense sampling of the step | no loss missed | target |
| a point exactly on an aperture edge | inside | inside | target |
| excursion between two turn boundaries | lost at the element | `track_loss_map` sees it, `track` does not (documented) | pinned |

The two kernels are compared above on equal numbers of particles, drawn at random (mean over five seeds; noise-free quantile samples
give the pairwise kernel the same error). The ranking is a statement about this spiked bunch, not about smooth ones.

Near the cathode the emission law of `PhotoCathodeModel` uses its own screening calculation, not this mesh; the mesh field is what
pushes the particles.

## Observed, not gated

- A bunch of zero length gives a finite CSR kick of about 1e198 in the grid kernel and zero in the pairwise kernel. The 1-D kick
  diverges there; an error or zero would be easier to notice.
- A particle with a NaN coordinate stays `alive` through an injector aperture (every comparison with NaN is false). The ring counts a
  non-finite particle as lost.

## Not covered

- **Where 1-D CSR itself breaks down** (transverse size comparable with `rho^(1/3) sigma_z^(2/3)`). Slipstream's model is 1-D, so the
  reference has to be a 3-D code; CSRTrack is in the simcodes image but has no worker here yet.
- GPU backends of these solvers (see `reviews/precision`).

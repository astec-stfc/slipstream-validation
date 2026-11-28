# precision review

Does every path of Slipstream give the same numbers? Eager float64 on the CPU is the reference; compiled, Warp, float32,
double-single and GPU paths run the same lattice and particles. A change to a formula in the torch maps does not reach the Warp or
double-single copies by itself, so each of them is a row here.

```bash
SSV_SLIPSTREAM=../slipstream pytest reviews/precision -m "not slow"   # about 15 s; GPU rows skip without a GPU
SSV_SLIPSTREAM=../slipstream pytest reviews/precision                 # adds the compiled double-single maps on the GPU
```

On Windows `torch.compile` needs `clang-cl`; the module sets `CXX` when it is not set. The Warp rows need `warp-lang`.

## Rows

| Path | Runs without a GPU | Coordinates |
| --- | --- | --- |
| ring: compiled float64, eager float32, double-single (full and hybrid) | yes | all six; `tau` is documented as not tracked in double-single |
| ring: Warp whole-turn kernel on Warp's CPU backend, float64 and float32 | yes | all six |
| ring: eager and default (Warp) on `cuda` or `rocm`, float64 and float32; compiled double-single (slow) | no | all six; double-single without `tau` |
| 3-D space-charge solve and grid CSR kick: float32, and both precisions on the GPU | float32 only | field, kick |

## Limits

Largest difference of a coordinate over 256 particles, divided by its largest reference value, after 20 turns of a 14-element cell.
`tau` starts at zero, so its error is that of its change.

| Precision | Limit | Kind | Measured 28 November 2026 (Ryzen 9 7950X, RTX 3050, `86f31e6`) |
| --- | --- | --- | --- |
| float64, any path | 1e-12 | target | 7e-15; Warp `tau` 5e-13 (CUDA), 2e-12 (CPU backend, one turn of a smaller cell) |
| float32, any path | 1.5e-5 | target | 5e-6 |
| double-single, full | 1e-11 | target (Slipstream's own self-check) | 2e-13 |
| double-single, hybrid | 2e-7 | ratchet | 9e-8 |
| collective kernels, float64 / float32 | 1e-12 / 1e-5 | target | 2e-14 / 2e-6 |

## Known failure

[astec-stfc/slipstream#62](https://github.com/astec-stfc/slipstream/issues/62): the Warp ring kernel loses the change of `tau` in
float32 (relative error 1.1e-4 against 4.7e-7 on the eager path). That is the default CUDA ring path. Strict `xfail` on the GPU and
on the CPU backend: a fix fails the mark, which is then removed.

## Not covered

- **ROCm has never run these rows**: no AMD card was available. torch presents ROCm as `torch.cuda`, so they should select it, and
  the report labels it `rocm`; Warp does not run on AMD cards, so there the default path is the compiled one. The limits may need a
  second look on a first run.
- The injector push, injector IBS and linac wake kernels also have Warp paths and are not compared here yet.
- CI has no GPU: the GPU rows run only where someone runs them.

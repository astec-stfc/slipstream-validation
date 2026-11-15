# Free-space space-charge expansion of a 5 MeV, 20 pC Gaussian electron bunch in a 3 m drift

Reference: `astra/cpu-f64-3d128h1.4`. Differences are the worst relative difference over stations 1.. (19 stations); tolerances: size and length 0.03, emittance 0.05. `prev` is the change within the same solver family, device and precision. Times are medians over independent processes. CSV energy columns separate the absolute mean-energy gap, entrance baseline offset and discrepancy in energy change, so a constant GPT offset is not read as an incorrect force.

| code | config | warm [s] | cold [s] | size vs ref | length vs ref | emit vs ref | spread vs ref | dEkin [eV] | emit vs prev | pass | warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| astra | cpu-f64-3d64h1.4 | 82.4 | 83.4 | 0.00258 | 1e-05 | 0.00995 | 0.00221 | 0 | - | True |  |
| astra | cpu-f64-3d128h1.4 | - | 540 | 0 | 0 | 0 | 0 | 0 | 0.01 | True |  |
| opalx | cpu-f64-open32h40-mono | 28.5 | 28.7 | 10.1 | 0.0612 | 13.9 | 2.1 | 60.3 | - | False |  |
| opalx | cpu-f64-open32h40 | 27.1 | 27.3 | 0.0083 | 0.000129 | 0.0251 | 0.00993 | 88.3 | 0.933 | True |  |
| opalx | cpu-f64-open64h40 | 259 | 258 | 0.00277 | 2.71e-05 | 0.0111 | 0.00278 | 87.9 | 0.0152 | True |  |
| opalx | cuda-f64-open32h40-mono | 7.84 | 7.85 | 10.1 | 0.0612 | 13.9 | 2.1 | 60.3 | - | False | cold and warm outputs differ by 1.863e-09 (non-deterministic or state-carrying); cold and warm outputs differ by 7.276e-12 (non-deterministic or state-carrying) |
| opalx | cuda-f64-open32h40 | 8.28 | 8.21 | 0.0083 | 0.000129 | 0.0251 | 0.00993 | 88.3 | 0.933 | True | cold and warm outputs differ by 3.638e-12 (non-deterministic or state-carrying); cold and warm outputs differ by 6.366e-12 (non-deterministic or state-carrying); cold and warm outputs differ by 9.313e-10 (non-deterministic or state-carrying) |
| opalx | cuda-f64-open64h40 | 28.5 | 28.5 | 0.00277 | 2.71e-05 | 0.0111 | 0.00278 | 87.9 | 0.0152 | True | cold and warm outputs differ by 9.313e-10 (non-deterministic or state-carrying) |
| opalx | cuda-f64-open128h40 | 173 | 173 | 0.00125 | 5.55e-05 | 0.00487 | 8.41e-05 | 87.8 | 0.00732 | True | cold and warm outputs differ by 1.863e-09 (non-deterministic or state-carrying); cold and warm outputs differ by 9.313e-10 (non-deterministic or state-carrying) |
| slipstream | cuda-f32-r32z128t2c5 | 2.49 | 4.25 | 0.00968 | 0.00213 | 0.0298 | 0.0747 | 52 | - | True | cold and warm outputs differ by 1.911e-04 (non-deterministic or state-carrying); cold and warm outputs differ by 3.735e-04 (non-deterministic or state-carrying); cold and warm outputs differ by 4.327e-04 (non-deterministic or state-carrying) |
| slipstream | cuda-f32-r64z256t1c2 | 9.06 | 10.6 | 0.00296 | 0.002 | 0.0169 | 0.0689 | 52.5 | 0.0143 | True | cold and warm outputs differ by 4.068e-05 (non-deterministic or state-carrying); cold and warm outputs differ by 4.397e-04 (non-deterministic or state-carrying) |
| slipstream | cuda-f32-3d32t1c10-r100 | 2.74 | 4.34 | 0.0124 | 0.000349 | 0.0338 | 0.0205 | 51.5 | - | True | cold and warm outputs differ by 4.068e-06 (non-deterministic or state-carrying); cold and warm outputs differ by 4.210e-04 (non-deterministic or state-carrying); cold and warm outputs differ by 6.071e-11 (non-deterministic or state-carrying) |
| slipstream | cuda-f32-3d64t1c10-r100 | 3.6 | 5.26 | 0.00368 | 3.56e-05 | 0.0125 | 0.00582 | 52.4 | 0.0224 | True | cold and warm outputs differ by 1.049e-03 (non-deterministic or state-carrying); cold and warm outputs differ by 2.421e-04 (non-deterministic or state-carrying); cold and warm outputs differ by 8.742e-04 (non-deterministic or state-carrying) |
| slipstream | cuda-f64-3d32t1c1 | 13.9 | 15.6 | 0.0125 | 0.000351 | 0.0338 | 0.0205 | 52.4 | - | True | cold and warm outputs differ by 1.084e-19 (non-deterministic or state-carrying) |

## Cheapest passing configuration per code (by warm time)

- astra: `cpu-f64-3d64h1.4` at 82.4 s (size 0.0026, emit 0.0099)
- opalx: `cuda-f64-open32h40` at 8.28 s (size 0.0083, emit 0.025)
- slipstream: `cuda-f32-r32z128t2c5` at 2.49 s (size 0.0097, emit 0.03)

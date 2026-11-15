# Slipstream benchmark study

Campaign `2026-10-26_R1`; provisional tables; campaign still running. 111/111 recorded jobs completed successfully.

Design: three independent processes and two warm calls per process, one CPU thread; sequential jobs in a seeded random order. Times include each adapter's diagnostics; executable adapters also include process and file I/O. GPU and CPU results remain labelled. Warm timing is conditional on an already initialised process. Cold, build, import, memory and process wall time remain in the raw results.

Frozen design and source snapshot: `benchmarks/results/2026-10-26_R1/`. Portable summary and per-job provenance: `data/2026-10-26_R1/evidence.json`. Reproduce with `python benchmarks/study.py --campaign <new-name>`. No rehearsal outputs are used.

References are continuity anchors, not truth: elegant's finest selected ring integrator and ASTRA's 3-D drift. Passing a tune test does not imply the stricter first-turn map gate passes. The drift pass gates size, bunch length and emittance only; energy-spread differences are reported separately. Jacobian norms combine components in their declared numerical SI units and are diagnostic, not unit-invariant physics metrics. This bounded study does not establish production-scale throughput, CSR fidelity or cathode-emission accuracy.

## sc_drift

| Code | Configuration | Warm s | Size relative error | Emittance relative error | Energy-spread relative error | Size/length/emittance pass |
| --- | --- | --- | --- | --- | --- | --- |
| astra | cpu-f64-n64l128h0.7 | 139 | 0.0121 | 0.0333 | 0.00852 | True |
| astra | cpu-f64-3d32h1.4 | 35.9 | 0 | 0 | 0 | True |
| cheetah | cpu-f64-g32s2 | 4.83 | 0.000984 | 0.017 | 0.00215 | True |
| cheetah | cpu-f64-g64s4 | 50.9 | 0.00861 | 0.0268 | 0.00802 | True |
| gpt | cpu-f64-mesh-acc5 | 2.91 | 0.00659 | 0.0245 | 0.0686 | True |
| gpt | cpu-f64-mesh-acc6 | 2.84 | 0.00659 | 0.0245 | 0.0686 | True |
| impactx | cpu-f64-mg32x40s2 | 4.47 | 0.00414 | 0.00781 | 0.0861 | True |
| impactx | cpu-f64-mg64x80s4 | 67.7 | 0.00857 | 0.0215 | 0.0724 | True |
| ocelot | cpu-f64-g32s2 | 3.8 | 0.00481 | 0.0142 | 0.0031 | True |
| ocelot | cpu-f64-g64s4 | 65.8 | 0.00949 | 0.0267 | 0.00965 | True |
| slipstream | cuda-f32-r32z128t2c5 | 2.88 | 0.00374 | 0.0146 | 0.0634 | True |
| slipstream | cuda-f32-r64z256t1c2 | 10.7 | 0.0106 | 0.0284 | 0.0576 | True |
| slipstream | cpu-f64-3d32t1c1 | 103 | 0.000577 | 0.000702 | 0.00859 | True |

## linear_fodo

| Code | Configuration | Position error m | Momentum error | Pass |
| --- | --- | --- | --- | --- |
| helix | cpu-f64-linear | 3.84e-22 | 5.47e-20 | True |
| helix | cpu-f64-linear | 3.84e-22 | 5.47e-20 | True |
| helix | cpu-f64-linear | 3.84e-22 | 5.47e-20 | True |
| slipstream | cpu-f64-o4k4b8 | 1.66e-11 | 4.07e-11 | False |
| slipstream | cpu-f64-o4k4b8 | 1.66e-11 | 4.07e-11 | False |
| slipstream | cpu-f64-o4k4b8 | 1.66e-11 | 4.07e-11 | False |
| slipstream | cpu-f64-o6k8b16 | 2.38e-16 | 4.13e-16 | True |
| slipstream | cpu-f64-o6k8b16 | 2.38e-16 | 4.13e-16 | True |
| slipstream | cpu-f64-o6k8b16 | 2.38e-16 | 4.13e-16 | True |

## ring_tunes (historical reference; see correction below)

| Code | Configuration | Warm s | Tune error | Change on refinement | First-turn x error m | First-turn px error | Tune pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bmadx | cpu-f64-q1s16 | 37.5 | 0.000131 | -- | 2.84e-06 | 1.58e-06 | False |
| bmadx | cpu-f64-q1s64 | 122 | 0.000133 | 9.15e-06 | 1.09e-06 | 1.97e-06 | False |
| cheetah | cpu-f64-lin1 | 1.35 | 0.00336 | -- | 0.000526 | 0.000203 | False |
| cheetah | cpu-f64-dri4 | 52.9 | 0.000104 | -- | 1.79e-07 | 1.75e-06 | False |
| cheetah | cpu-f64-dri16 | 138 | 0.000104 | 1.28e-11 | 1.79e-07 | 1.75e-06 | False |
| elegant | cpu-f64-o6s8b16 | 0.723 | 3.8e-09 | -- | 3.32e-10 | 2.55e-10 | True |
| elegant | cpu-f64-o6s16b32 | 1.3 | 0 | 3.8e-09 | 0 | 0 | True |
| jutrack | cpu-f64-n16b32 | 0.408 | 4.34e-05 | -- | 2.41e-05 | 8.16e-07 | False |
| jutrack | cpu-f64-n32b64 | 0.853 | 4.37e-05 | 3.13e-07 | 2.41e-05 | 8.22e-07 | False |
| slipstream | cuda-f32-o6q8s1b8 | 0.482 | 3.72e-06 | -- | 2.46e-07 | 1.42e-07 | True |
| slipstream | cuda-f64-o6q8s1b8 | 2.07 | 1.97e-06 | -- | 1.95e-07 | 1.11e-07 | True |
| slipstream | cuda-f64-o6k16b32 | 5.63 | 1.97e-06 | 1.7e-09 | 1.95e-07 | 1.11e-07 | True |
| xsuite | cuda-f64-yosk8b16 | 4.61 | 0.00029 | -- | 1.06e-06 | 4.62e-06 | False |
| xsuite | cpu-f64-yosk8b16 | 0.19 | 0.00029 | -- | 1.06e-06 | 4.62e-06 | False |
| xsuite | cpu-f64-yosk16b32 | 0.29 | 0.000297 | 6.26e-06 | 1e-06 | 4.73e-06 | False |

## ring_grad (historical reference; see correction below)

| Code | Configuration | FD s | AD s | AD/self FD norm | Jacobian/reference norm | Status |
| --- | --- | --- | --- | --- | --- | --- |
| bmadx | cpu-f64-q1s16 | 13.7 | 10.4 | 4.24e-09 | 0.00818 | ok |
| bmadx | cpu-f64-q1s16 | 13.9 | 10.3 | 4.24e-09 | 0.00818 | ok |
| bmadx | cpu-f64-q1s16 | 13.7 | 10.5 | 4.24e-09 | 0.00818 | ok |
| cheetah | cpu-f64-dri4 | 20.7 | 24.4 | 4.17e-09 | 0.00642 | ok |
| cheetah | cpu-f64-dri4 | 20.6 | 24.7 | 4.17e-09 | 0.00642 | ok |
| cheetah | cpu-f64-dri4 | 21.4 | 24.8 | 4.17e-09 | 0.00642 | ok |
| elegant | cpu-f64-o6s16b32 | 1.1 | -- | -- | 0 | ok |
| elegant | cpu-f64-o6s16b32 | 0.968 | -- | -- | 0 | ok |
| elegant | cpu-f64-o6s16b32 | 0.68 | -- | -- | 0 | ok |
| jutrack | cpu-f64-n16b32 | 0.136 | -- | -- | 0.000599 | ok |
| jutrack | cpu-f64-n16b32 | 0.207 | -- | -- | 0.000599 | ok |
| jutrack | cpu-f64-n16b32 | 0.192 | -- | -- | 0.000599 | ok |
| slipstream | cpu-f64-o6q8s1b8 | 44.9 | 78.7 | 3.87e-09 | 0.00022 | ok |
| slipstream | cpu-f64-o6q8s1b8 | 45 | 80.9 | 3.87e-09 | 0.00022 | ok |
| slipstream | cpu-f64-o6q8s1b8 | 44.9 | 79.1 | 3.87e-09 | 0.00022 | ok |
| xsuite | cpu-f64-yosk8b16 | 0.0375 | -- | -- | 0.0161 | ok |
| xsuite | cpu-f64-yosk8b16 | 0.0377 | -- | -- | 0.0161 | ok |
| xsuite | cpu-f64-yosk8b16 | 0.0375 | -- | -- | 0.0161 | ok |

## Corrected momentum reference

Nine new elegant reference jobs use the same normalised reference momentum as the input particles. The remaining rows reuse the original saved arrays and timings; no other codes were retimed. The original ring tables above and the original Xsuite/CPU control values below are historical. Full per-component and gradient results: `data/2026-10-26_R1/corrected_reference.json`. Independent bend attribution: `data/2026-10-26_R1/bend_attribution.json`.

| Code | Configuration | Warm s | Tune error | x error m | px error | y error m | py error | Tune pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bmadx | cpu-f64-q1s16 | 37.5 | 0.00013 | 2.64e-06 | 1.69e-06 | 3.88e-07 | 3.87e-09 | False |
| bmadx | cpu-f64-q1s64 | 122 | 0.000133 | 8.99e-07 | 2.08e-06 | 2.84e-07 | 8.1e-10 | False |
| cheetah | cpu-f64-lin1 | 1.35 | 0.00336 | 0.000526 | 0.000203 | 0.00021 | 2.12e-05 | False |
| cheetah | cpu-f64-dri4 | 52.9 | 0.000104 | 1.5e-07 | 1.86e-06 | 1.05e-07 | 9.47e-10 | False |
| cheetah | cpu-f64-dri16 | 138 | 0.000104 | 1.5e-07 | 1.86e-06 | 1.05e-07 | 9.47e-10 | False |
| jutrack | cpu-f64-n16b32 | 0.408 | 4.14e-05 | 2.39e-05 | 7.05e-07 | 2.15e-06 | 3.27e-07 | False |
| jutrack | cpu-f64-n32b64 | 0.853 | 4.17e-05 | 2.39e-05 | 7.1e-07 | 2.15e-06 | 3.27e-07 | False |
| slipstream | cuda-f32-o6q8s1b8 | 0.482 | 1.75e-06 | 5.04e-08 | 3.03e-08 | 6.78e-09 | 2.1e-10 | True |
| slipstream | cuda-f64-o6q8s1b8 | 2.07 | 1.66e-09 | 8.25e-12 | 2.68e-11 | 1.1e-11 | 4.87e-13 | True |
| slipstream | cuda-f64-o6k16b32 | 5.63 | 3.59e-11 | 5.04e-12 | 3.57e-12 | 3.67e-13 | 6.82e-15 | True |
| xsuite | cuda-f64-yosk8b16 | 4.61 | 0.00029 | 8.61e-07 | 4.73e-06 | 5.83e-07 | 2.76e-09 | False |
| xsuite | cpu-f64-yosk8b16 | 0.19 | 0.00029 | 8.61e-07 | 4.73e-06 | 5.83e-07 | 2.76e-09 | False |
| xsuite | cpu-f64-yosk16b32 | 0.29 | 0.000296 | 8.08e-07 | 4.84e-06 | 6.26e-07 | 3.91e-09 | False |
| elegant | cpu-f64-o6s8b16 | 0.701 | 3.82e-09 | 3.32e-10 | 2.55e-10 | 3.43e-11 | 9.23e-13 | True |
| elegant | cpu-f64-o6s16b32 | 1.3 | 0 | 0 | 0 | 0 | 0 | True |
| scibmad | cpu-f64-o4k16 | 1.65 | 5.98e-11 | 5.53e-12 | 3.51e-12 | 4.98e-13 | 1.46e-14 | True |
| scibmad | cpu-f64-o4k4 | 0.898 | 8.86e-09 | 7.48e-11 | 1.31e-10 | 2.48e-11 | 4.09e-13 | True |
| scibmad | cpu-f64-o6k32 | 4.77 | 8.08e-11 | 5.26e-12 | 4.04e-12 | 5.44e-13 | 1.47e-14 | True |
| scibmad | cpu-f64-o6k8 | 1.77 | 7.62e-11 | 5.26e-12 | 4.04e-12 | 5.44e-13 | 1.47e-14 | True |
| scibmad | cpu-f64-std | 0.745 | 0.00396 | 4.37e-05 | 5.5e-05 | 1.95e-05 | 4.82e-07 | False |

## Exploratory Xsuite model correction (historical reference)

Design and full per-process evidence: `data/2026-10-26_R1/xsuite_model_control.json`.

```json
{
  "design": {
    "classification": "exploratory adapter model correction; original R1 preserved",
    "cases": {
      "ring_tunes": [
        "cpu-f64-yosk8b16-exact",
        "cpu-f64-yosk16b32-exact",
        "cuda-f64-yosk8b16-exact"
      ],
      "ring_grad": [
        "cpu-f64-yosk8b16-exact"
      ]
    },
    "reference_campaign": "2026-10-26_R1",
    "procs": 3,
    "repeats": 2,
    "threads": 1,
    "change": "Force Drift exact and Sextupole drift-kick-drift-exact; other settings and error gates unchanged",
    "source_evidence": "installed xtrack beam_elements/elements_src/drift.h: adaptive resolves to expanded unless XTRACK_USE_EXACT_DRIFTS is defined"
  },
  "rows": [
    {
      "case": "ring_tunes",
      "config": "cpu-f64-yosk8b16-exact",
      "n_ok": 3,
      "warm_s": 0.1993928279262036,
      "dq_ref": 8.126936212715652e-06,
      "map_components": [
        2.5330365278040945e-07,
        2.3373989689514513e-07,
        6.287918608504587e-08,
        2.3964652172414613e-09
      ],
      "pass": true
    },
    {
      "case": "ring_tunes",
      "config": "cpu-f64-yosk16b32-exact",
      "n_ok": 3,
      "warm_s": 0.30423454544506967,
      "dq_ref": 2.5098554224112224e-06,
      "map_components": [
        2.0031820365667974e-07,
        1.2213635936604225e-07,
        2.04835833547733e-08,
        5.097684300505931e-10
      ],
      "pass": true
    },
    {
      "case": "ring_tunes",
      "config": "cuda-f64-yosk8b16-exact",
      "n_ok": 3,
      "warm_s": 4.760091964970343,
      "dq_ref": 8.126953352949329e-06,
      "map_components": [
        2.5330365235887164e-07,
        2.3373989674465787e-07,
        6.287918607886592e-08,
        2.396465216997516e-09
      ],
      "pass": true
    },
    {
      "case": "ring_grad",
      "config": "cpu-f64-yosk8b16-exact",
      "n_ok": 3,
      "fd_s": 0.03891112335259095,
      "rel_fro": 0.001541805350991313,
      "rel_col": 0.01862575121660883
    }
  ]
}
```

## Exploratory native GPU Cartesian control

Design and full per-process evidence: `data/2026-10-26_R1/gpu_control.json`.

```json
{
  "design": {
    "classification": "exploratory follow-up; primary R1 design unchanged",
    "reference_campaign": "2026-10-26_R1",
    "question": "3-D Cartesian parity and cost at fixed 32^3 mesh, 1 ps, refresh every step",
    "configs": [
      "cuda-f32-3d32t1c1",
      "cuda-f64-3d32t1c1"
    ],
    "procs": 3,
    "repeats": 2,
    "gates": "primary size/length/emittance gates unchanged; spread and CPU64 parity reported separately"
  },
  "rows": [
    {
      "config": "cuda-f32-3d32t1c1",
      "n_ok": 3,
      "warm_s": 13.396613349905238,
      "versus_astra": {
        "size": 0.0005756357804811283,
        "length": 0.00015022649749941088,
        "emit": 0.0007031000093390867,
        "spread": 0.008587570503679533
      },
      "versus_cpu64": {
        "size": 1.447346681672919e-06,
        "length": 2.1125751023822338e-06,
        "emit": 3.2053283412401616e-06,
        "spread": 1.536287987706979e-06
      }
    },
    {
      "config": "cuda-f64-3d32t1c1",
      "n_ok": 3,
      "warm_s": 14.572817950043827,
      "versus_astra": {
        "size": 0.0005770173981923539,
        "length": 0.00015154573529373917,
        "emit": 0.0007024691985382957,
        "spread": 0.008588077463232152
      },
      "versus_cpu64": {
        "size": 5.874568880615462e-15,
        "length": 5.6168002975834225e-15,
        "emit": 2.3442947030049752e-14,
        "spread": 4.60991892304051e-15
      }
    }
  ]
}
```

## Implemented opt-in CPU compilation (historical reference)

Design and full per-process evidence: `data/2026-10-26_R1/cpu_control.json`.

```json
{
  "design": {
    "classification": "post-baseline performance implementation control",
    "hypothesis": "Existing regional CPU compilation reduces eager dispatch while preserving maps/gradients",
    "config": "cpu-f64-o6q8s1b8-compiled",
    "procs": 3,
    "repeats": 2,
    "threads": 1,
    "change": "compile_cpu=True only; native parser guard is unused by this JSON-to-ElementSpec adapter",
    "gates": "Unchanged ring tune gate; full component and AD/self-FD residuals retained"
  },
  "rows": [
    {
      "case": "ring_tunes",
      "n_ok": 3,
      "warm_s": 4.569227950007189,
      "dq_ref": 1.972518031273518e-06
    },
    {
      "case": "ring_grad",
      "n_ok": 3,
      "fd_s": 1.8277730000118027,
      "ad_s": 1.5232469500042498,
      "ad_vs_fd": [
        3.8734412279856156e-09,
        3.5960313886449006e-08
      ],
      "versus_eager_ad": [
        9.906855493614458e-15,
        4.0443493478652807e-14
      ]
    }
  ]
}
```

## Injector diagnostic and refresh presets

Design and full per-process evidence: `data/2026-10-26_R1/injector_cost_control.json`.

```json
{
  "design": {
    "classification": "post-baseline workload-specific performance control",
    "configs": [
      "cuda-f32-3d32t1c1-r50",
      "cuda-f32-3d32t1c1-r100",
      "cuda-f32-3d32t1c5-r100",
      "cuda-f32-3d32t1c10-r100"
    ],
    "fixed": "CUDA32 Cartesian 32^3, 1 ps push, common 20000 particles and 20 output stations",
    "procs": 3,
    "repeats": 2,
    "threads": 1,
    "diagnostics_gate": "Worst size/length/emittance/spread change <=1e-4 relative to record_every=5",
    "refresh_gate": "Worst observable change <=1e-3 relative to c1-r100",
    "physics_gate": "Original 3% size/length and 5% emittance; additional <=1% energy-spread error vs ASTRA32",
    "scope": "Free-space 20 pC drift only; no gun/RF preset or converged-reference claim"
  },
  "rows": [
    {
      "config": "cuda-f32-3d32t1c1-r50",
      "n_ok": 3,
      "warm_s": 12.120112650096416,
      "versus_astra": {
        "size": 0.0005755436688531362,
        "length": 0.0001501801463804065,
        "emit": 0.0007081185932371745,
        "spread": 0.008587539986960196
      },
      "versus_record5": {
        "size": 4.379098455053784e-06,
        "length": 2.2252273373718062e-06,
        "emit": 1.4900876174334789e-05,
        "spread": 2.547526656921801e-06
      },
      "versus_c1r100": {
        "size": 1.6012115833165517e-05,
        "length": 2.3598468975048228e-06,
        "emit": 4.6484676156252696e-05,
        "spread": 6.285873950047208e-06
      },
      "physics_pass": true,
      "parity_pass": true,
      "pass": true
    },
    {
      "config": "cuda-f32-3d32t1c1-r100",
      "n_ok": 3,
      "warm_s": 11.763477050233632,
      "versus_astra": {
        "size": 0.0005691101524720224,
        "length": 0.00015012048523524796,
        "emit": 0.0007283243686494972,
        "spread": 0.008587449033550151
      },
      "versus_record5": {
        "size": 1.731170307718258e-05,
        "length": 1.7050356863793556e-06,
        "emit": 6.077459834719096e-05,
        "spread": 6.366321982449603e-06
      },
      "versus_c1r100": {
        "size": 5.4087576540559875e-08,
        "length": 0.0,
        "emit": 7.28570010320144e-07,
        "spread": 0.0
      },
      "physics_pass": true,
      "parity_pass": true,
      "pass": true
    },
    {
      "config": "cuda-f32-3d32t1c5-r100",
      "n_ok": 3,
      "warm_s": 4.03195254993625,
      "versus_astra": {
        "size": 0.0005384617963242489,
        "length": 0.0001500620378344957,
        "emit": 0.0006949845241417233,
        "spread": 0.00857405700616925
      },
      "versus_record5": {
        "size": 3.737559359544575e-05,
        "length": 1.5722096832443453e-06,
        "emit": 5.28169913197543e-05,
        "spread": 1.363055082650932e-05
      },
      "versus_c1r100": {
        "size": 3.074470530357534e-05,
        "length": 2.5378244933596247e-07,
        "emit": 5.9075971456757295e-05,
        "spread": 1.350802687321691e-05
      },
      "physics_pass": true,
      "parity_pass": true,
      "pass": true
    },
    {
      "config": "cuda-f32-3d32t1c10-r100",
      "n_ok": 3,
      "warm_s": 2.9617873998358846,
      "versus_astra": {
        "size": 0.0004996854525019101,
        "length": 0.00014982639685373204,
        "emit": 0.0006549459708500702,
        "spread": 0.008556320600062471
      },
      "versus_record5": {
        "size": 7.602983539213835e-05,
        "length": 9.915648709762438e-07,
        "emit": 0.00012345012015012267,
        "spread": 3.1520588896528686e-05
      },
      "versus_c1r100": {
        "size": 6.949999504217182e-05,
        "length": 7.134720318983874e-07,
        "emit": 0.00012970954235512732,
        "spread": 3.139806275130796e-05
      },
      "physics_pass": true,
      "parity_pass": true,
      "pass": true
    }
  ]
}
```

## Ring precision control

Full evidence: `data/2026-10-26_R1/precision_control.json`.

```json
{
  "design": {
    "date": "2026-11-07",
    "campaign": "2026-10-28_precision",
    "note": "Campaign name is a label; the jobs ran with slipstream 1.1.0 on 7-8 November 2026 at the paper setting o6q8s1 (the folder keeps its v1.0 name). Baseline arrays are the f32 and f64 rungs of 2026-10-26_R1 at the same setting. The full-mode rung ran once, one process and no warm repeats: its accuracy and cold time are reported, no warm timing is claimed (the v1.0 attempt was interrupted after more than six minutes of cold setup).",
    "procs": 3,
    "repeats": 2,
    "threads": 1,
    "target": "unchanged tune and first-turn component gates"
  },
  "rows": [
    {
      "config": "cuda-f32-o6q8s1b8",
      "n_started": 3,
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 0.48150474997237325,
      "cold_s": 0.6092552999616601,
      "dq_ref": 1.7510046218527364e-06,
      "map_components": [
        5.036055627030689e-08,
        3.031111754561028e-08,
        6.779400905942718e-09,
        2.1037659824916718e-10
      ],
      "both_gates": false
    },
    {
      "config": "cuda-f64-o6q8s1b8",
      "n_started": 3,
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 2.067158849997213,
      "cold_s": 2.1875875999685377,
      "dq_ref": 1.6649291667558686e-09,
      "map_components": [
        8.248745436700844e-12,
        2.6836561185100694e-11,
        1.0957069073672804e-11,
        4.870192519667443e-13
      ],
      "both_gates": true
    },
    {
      "config": "cuda-f32ds-hybrid-o6q8s1b8",
      "n_started": 3,
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 2.129218000016408,
      "cold_s": 26.659622600011062,
      "dq_ref": 2.8219342140811676e-09,
      "map_components": [
        1.3155973012379807e-10,
        5.316879358660542e-11,
        3.021250009069623e-11,
        2.5989468145940983e-12
      ],
      "both_gates": true
    },
    {
      "config": "cuda-f32ds-full-o6q8s1b8",
      "n_started": 1,
      "n_recorded": 1,
      "n_ok": 1,
      "warm_s": null,
      "cold_s": 206.52060799999163,
      "dq_ref": 1.6921402057334944e-09,
      "map_components": [
        8.247919708326279e-12,
        2.6836160897658612e-11,
        1.0957113905432636e-11,
        4.870260553353767e-13
      ],
      "both_gates": null
    }
  ]
}
```

## ASTRA 3-D refinement control

Full evidence: `data/2026-10-26_R1/astra_refinement.json`.

```json
{
  "design": {
    "date": "2026-10-27",
    "procs": 3,
    "repeats": 2,
    "threads": 1,
    "note": "Vary one of 3-D grid and time step independently; no convergence claim from two grid points."
  },
  "rows": [
    {
      "config": "cpu-f64-3d32h1.4",
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 35.94715579994954,
      "versus_baseline": {
        "size": 0.0,
        "length": 0.0,
        "emit": 0.0,
        "spread": 0.0
      },
      "native_vs_this": {
        "cpu64": {
          "size": 0.0005770173981950425,
          "length": 0.00015154573528812322,
          "emit": 0.0007024691985410272,
          "spread": 0.008588077463230238
        },
        "gpu_fast": {
          "size": 0.0004996854525019101,
          "length": 0.00014982639685373204,
          "emit": 0.0006549459708500702,
          "spread": 0.008556320600062471
        }
      },
      "native_artifact_sha256": {
        "cpu64": [
          "78020a1c39b9525e368ab17fc6b5e17ea1331cb81e1e74aa80dcc26bdb1fafb6",
          "78020a1c39b9525e368ab17fc6b5e17ea1331cb81e1e74aa80dcc26bdb1fafb6",
          "78020a1c39b9525e368ab17fc6b5e17ea1331cb81e1e74aa80dcc26bdb1fafb6"
        ],
        "gpu_fast": [
          "4ce4f017d9ccff6be7735a546e4aacf5b2ecb568949e7156c481b07b35ad6058",
          "4ff907ad9f2ed8e300c6b4812431b6d12d5ca2e5a5db576227609631e1995bc0",
          "df3e8b870e80ada2a2bdad3b66837ea48add9c51f1bd4b71a8c9fefefe4d48e1"
        ]
      }
    },
    {
      "config": "cpu-f64-3d64h1.4",
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 82.37846575002186,
      "versus_baseline": {
        "size": 0.009435299119057093,
        "length": 0.0001997203914518638,
        "emit": 0.025577119509704017,
        "spread": 0.009932381667918776
      },
      "native_vs_this": {
        "cpu64": {
          "size": 0.009918730329710058,
          "length": 0.0003506988981731335,
          "emit": 0.024527217037807067,
          "spread": 0.01833831597771149
        },
        "gpu_fast": {
          "size": 0.009842121213939467,
          "length": 0.00034897990220232855,
          "emit": 0.024484546392233252,
          "spread": 0.018306871433755666
        }
      },
      "native_artifact_sha256": {
        "cpu64": [
          "78020a1c39b9525e368ab17fc6b5e17ea1331cb81e1e74aa80dcc26bdb1fafb6",
          "78020a1c39b9525e368ab17fc6b5e17ea1331cb81e1e74aa80dcc26bdb1fafb6",
          "78020a1c39b9525e368ab17fc6b5e17ea1331cb81e1e74aa80dcc26bdb1fafb6"
        ],
        "gpu_fast": [
          "4ce4f017d9ccff6be7735a546e4aacf5b2ecb568949e7156c481b07b35ad6058",
          "4ff907ad9f2ed8e300c6b4812431b6d12d5ca2e5a5db576227609631e1995bc0",
          "df3e8b870e80ada2a2bdad3b66837ea48add9c51f1bd4b71a8c9fefefe4d48e1"
        ]
      }
    },
    {
      "config": "cpu-f64-3d32h0.7",
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 74.52112515014596,
      "versus_baseline": {
        "size": 0.0002128766734470727,
        "length": 5.972495464346356e-07,
        "emit": 0.001179797699394538,
        "spread": 9.0157776108244e-05
      },
      "native_vs_this": {
        "cpu64": {
          "size": 0.0003645349536277548,
          "length": 0.00015154573528812322,
          "emit": 0.0016640612586776664,
          "spread": 0.008498685909743458
        },
        "gpu_fast": {
          "size": 0.0002871865667669186,
          "length": 0.00014982639685373204,
          "emit": 0.001787391992063092,
          "spread": 0.008466926183189376
        }
      },
      "native_artifact_sha256": {
        "cpu64": [
          "78020a1c39b9525e368ab17fc6b5e17ea1331cb81e1e74aa80dcc26bdb1fafb6",
          "78020a1c39b9525e368ab17fc6b5e17ea1331cb81e1e74aa80dcc26bdb1fafb6",
          "78020a1c39b9525e368ab17fc6b5e17ea1331cb81e1e74aa80dcc26bdb1fafb6"
        ],
        "gpu_fast": [
          "4ce4f017d9ccff6be7735a546e4aacf5b2ecb568949e7156c481b07b35ad6058",
          "4ff907ad9f2ed8e300c6b4812431b6d12d5ca2e5a5db576227609631e1995bc0",
          "df3e8b870e80ada2a2bdad3b66837ea48add9c51f1bd4b71a8c9fefefe4d48e1"
        ]
      }
    }
  ]
}
```

## Injector push-step refinement

Full evidence: `data/2026-10-26_R1/injector_step_control.json`.

```json
{
  "design": {
    "date": "2026-10-27",
    "procs": 3,
    "repeats": 2,
    "threads": 1,
    "fixed": "Cartesian 32^3, 10 ps field refresh, 100 ps recording interval",
    "gate": "max size/length/emittance/spread change <=1e-3; original ASTRA gates plus <=1% spread"
  },
  "rows": [
    {
      "config": "cuda-f32-3d32t1c10-r100",
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 2.9617873998358846,
      "versus_base": {
        "size": 1.1373878520784693e-07,
        "length": 0.0,
        "emit": 1.271749049021026e-06,
        "spread": 0.0
      },
      "versus_astra": {
        "size": 0.0004996854525019101,
        "length": 0.00014982639685373204,
        "emit": 0.0006549459708500702,
        "spread": 0.008556320600062471
      },
      "versus_astra64": {
        "size": 0.009842121213939467,
        "length": 0.00034897990220232855,
        "emit": 0.024484546392233252,
        "spread": 0.018306871433755666
      },
      "pass": true
    },
    {
      "config": "cuda-f32-3d32t0.5c20-r200",
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 5.679671350168064,
      "versus_base": {
        "size": 1.4732672345597471e-05,
        "length": 1.0052529234070884e-06,
        "emit": 9.041638425439573e-06,
        "spread": 1.6661253708919208e-06
      },
      "versus_astra": {
        "size": 0.0005133066539224981,
        "length": 0.00014982363956938887,
        "emit": 0.0006506758086477604,
        "spread": 0.00855480854783272
      },
      "versus_astra64": {
        "size": 0.00985561509654143,
        "length": 0.0003489771454671905,
        "emit": 0.02448250774491246,
        "spread": 0.018305374252105492
      },
      "pass": true
    }
  ]
}
```

## Native mesh sensitivity

Full evidence: `data/2026-10-26_R1/native_mesh_control.json`.

```json
{
  "design": {
    "procs": 3,
    "repeats": 2,
    "threads": 1,
    "fixed": "1 ps push, 10 ps refresh, 100 ps records, auto-fitted box with 5% particle-span margin",
    "note": "Two ASTRA meshes are peer references, not converged truth. No mesh is selected from observed errors."
  },
  "rows": [
    {
      "config": "cuda-f32-3d16t1c10-r100",
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 3.086588499834761,
      "versus_base": {
        "size": 0.03469045649323264,
        "length": 0.0011359922064873504,
        "emit": 0.07952535841112039,
        "spread": 0.04831994302299775,
        "emit_x": 0.07926467536108596,
        "emit_y": 0.07952535841112039
      },
      "versus_astra32": {
        "size": 0.03517277418659788,
        "length": 0.0012856484017219306,
        "emit": 0.07950893350795403,
        "spread": 0.0564628226991787,
        "emit_x": 0.07877079347977219,
        "emit_y": 0.07950893350795403
      },
      "versus_astra64": {
        "size": 0.044191116899304814,
        "length": 0.0014845756702405563,
        "emit": 0.10191010333855723,
        "spread": 0.06574222747214499,
        "emit_x": 0.10174555477540453,
        "emit_y": 0.10191010333855723
      }
    },
    {
      "config": "cuda-f32-3d32t1c10-r100",
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 2.9617873998358846,
      "versus_base": {
        "size": 1.1373878520784693e-07,
        "length": 0.0,
        "emit": 1.271749049021026e-06,
        "spread": 0.0,
        "emit_x": 1.271749049021026e-06,
        "emit_y": 1.1387739420849173e-06
      },
      "versus_astra32": {
        "size": 0.0004996854525019101,
        "length": 0.00014982639685373204,
        "emit": 0.0006549459708500702,
        "spread": 0.008556320600062471,
        "emit_x": 0.0006549459708500702,
        "emit_y": 0.0003091782819013589
      },
      "versus_astra64": {
        "size": 0.009842121213939467,
        "length": 0.00034897990220232855,
        "emit": 0.024484546392233252,
        "spread": 0.018306871433755666,
        "emit_x": 0.024484546392233252,
        "emit_y": 0.024410811208858314
      }
    },
    {
      "config": "cuda-f32-3d64t1c10-r100",
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 4.00102284993045,
      "versus_base": {
        "size": 0.008821012476022936,
        "length": 0.0003135387415084675,
        "emit": 0.02239746765641602,
        "spread": 0.014969740692824434,
        "emit_x": 0.021738464700051516,
        "emit_y": 0.02239746765641602
      },
      "versus_astra32": {
        "size": 0.008316954242030746,
        "length": 0.00017521818780951974,
        "emit": 0.02230971521039005,
        "spread": 0.006285334192094356,
        "emit_x": 0.022184109718836057,
        "emit_y": 0.02230971521039005
      },
      "versus_astra64": {
        "size": 0.0011078915884973868,
        "length": 3.5550579413209276e-05,
        "emit": 0.00339464095277921,
        "spread": 0.0036111798591914298,
        "emit_x": 0.00339464095277921,
        "emit_y": 0.002638046380481945
      }
    }
  ]
}
```

## Static Gaussian box and cell-size control

Full evidence: `data/2026-10-26_R1/space_charge_box_probe.json`.

```json
{
  "design": "Static spherical Gaussian grid density; analytic free-space radial electric field. No particles, time stepping, Lorentz transform or fitted injector box.",
  "rows": [
    {
      "n": 17,
      "half_box_sigma": 4,
      "cell_sigma": 0.5,
      "charge_fraction": 0.9999472473963081,
      "field": [
        1009.3713789093543,
        1654.517807624046,
        1620.5391241748675,
        980.3674486480238
      ],
      "relative_error": [
        0.09017187993821185,
        0.07375222332264209,
        0.023422953718217454,
        0.01134860368553059
      ]
    },
    {
      "n": 25,
      "half_box_sigma": 6,
      "cell_sigma": 0.5,
      "charge_fraction": 0.9999999991709833,
      "field": [
        1009.3712974348357,
        1654.5173954412594,
        1620.5361450469109,
        980.3559726756363
      ],
      "relative_error": [
        0.09017195337778983,
        0.07375245407470055,
        0.023424749014570923,
        0.011336765054345888
      ]
    },
    {
      "n": 33,
      "half_box_sigma": 4,
      "cell_sigma": 0.25,
      "charge_fraction": 0.9998938438660998,
      "field": [
        1082.5493856304522,
        1751.3532260188142,
        1649.8775005148295,
        972.3784920551512
      ],
      "relative_error": [
        0.024210619617094142,
        0.01954090533110031,
        0.005742920893726522,
        0.003107183485126841
      ]
    }
  ]
}
```

## Native dynamic box-padding control

Full evidence: `data/2026-10-26_R1/native_box_control.json`.

```json
{
  "design": {
    "procs": 3,
    "repeats": 2,
    "threads": 1,
    "fixed": "1 ps push, 10 ps refresh, 100 ps records",
    "default_parity": "max relative observable change <=1e-5 against saved 64^3",
    "box": "5% and 20% padding at 64^3, plus 20% at 81^3 for approximately matched initial cell size; dynamic refitting and cell-size buckets may change actual boxes.",
    "execution": "81^3/p0 was a pilot followed by the three-process continuation; all jobs ran sequentially.",
    "references": "ASTRA32 and ASTRA64 are peers, not converged truth"
  },
  "rows": [
    {
      "config": "cuda-f32-3d64t1c10-r100-p0.05",
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 4.303483949974179,
      "cuda_peak_bytes": 63361536,
      "versus_saved64": {
        "size": 1.0640755495322216e-07,
        "length": 1.0496957457235601e-07,
        "emit": 8.865700476485364e-07,
        "spread": 5.53423519202229e-08,
        "emit_x": 8.865700476485364e-07,
        "emit_y": 5.274192930673574e-07
      },
      "versus_astra32": {
        "size": 0.008316954242030746,
        "length": 0.00017521818780951974,
        "emit": 0.022309445617456517,
        "spread": 0.006285334192094356,
        "emit_x": 0.022184109718836057,
        "emit_y": 0.022309445617456517
      },
      "versus_astra64": {
        "size": 0.0011078915884973868,
        "length": 3.5550579413209276e-05,
        "emit": 0.00339464095277921,
        "spread": 0.003611215885403912,
        "emit_x": 0.00339464095277921,
        "emit_y": 0.002638046380481945
      },
      "versus_new_default": {
        "size": 8.609644746486759e-08,
        "length": 1.0496958559096876e-07,
        "emit": 8.691703730321572e-07,
        "spread": 5.53423519202229e-08,
        "emit_x": 8.691703730321572e-07,
        "emit_y": 4.089900477733309e-07
      },
      "default_parity_pass": true
    },
    {
      "config": "cuda-f32-3d64t1c10-r100-p0.2",
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 3.7871402001474053,
      "cuda_peak_bytes": 63361536,
      "versus_saved64": {
        "size": 0.0017528879724958404,
        "length": 6.67270800171746e-05,
        "emit": 0.004700483562319253,
        "spread": 0.0031215443648610485,
        "emit_x": 0.004617870760992865,
        "emit_y": 0.004700483562319253
      },
      "versus_astra32": {
        "size": 0.006549487580476252,
        "length": 0.0001291027829386795,
        "emit": 0.017504634791378097,
        "spread": 0.0031441698777047067,
        "emit_x": 0.017463795606213924,
        "emit_y": 0.017504634791378097
      },
      "versus_astra64": {
        "size": 0.002858837551152921,
        "length": 0.0001022752872440267,
        "emit": 0.007939410910444368,
        "spread": 0.00672145176591252,
        "emit_x": 0.007939410910444368,
        "emit_y": 0.0072856425225198175
      },
      "versus_new_default": {
        "size": 0.0017528879724958404,
        "length": 6.67270800171746e-05,
        "emit": 0.004699958621875039,
        "spread": 0.0031215083209436196,
        "emit_x": 0.004617870760992865,
        "emit_y": 0.004699958621875039
      }
    },
    {
      "config": "cuda-f32-3d81t1c10-r100-p0.2",
      "n_recorded": 3,
      "n_ok": 3,
      "warm_s": 15.032719949958846,
      "cuda_peak_bytes": 493444608,
      "versus_saved64": {
        "size": 2.7571140923782163e-05,
        "length": 1.6099607613989715e-06,
        "emit": 0.0001255600810052099,
        "spread": 5.0231241412597656e-05,
        "emit_x": 0.0001255600810052099,
        "emit_y": 8.261790511791086e-05
      },
      "versus_astra32": {
        "size": 0.0082891537931895,
        "length": 0.00017535629422259192,
        "emit": 0.022252273119271616,
        "spread": 0.006234787230542597,
        "emit_x": 0.022060836157590434,
        "emit_y": 0.022252273119271616
      },
      "versus_astra64": {
        "size": 0.0011354321835860541,
        "length": 3.684991971566739e-05,
        "emit": 0.00351970906439017,
        "spread": 0.003661229706556736,
        "emit_x": 0.00351970906439017,
        "emit_y": 0.002695804786596116
      },
      "versus_new_default": {
        "size": 2.7571140923782163e-05,
        "length": 1.5049913448049167e-06,
        "emit": 0.00012644653894914438,
        "spread": 5.019508644636814e-05,
        "emit_x": 0.00012644653894914438,
        "emit_y": 8.22089488600385e-05
      }
    }
  ]
}
```


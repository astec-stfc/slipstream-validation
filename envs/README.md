# Environments

One conda `environment.yml` per code family. They are the install route for a Linux machine without Docker, for a WSL
distribution, and the recipe the Docker image uses for the codes that the simcodes base image does not carry, so a code is
described in one place.

| File | Code(s) | Platform key | In the image |
| --- | --- | --- | --- |
| `core.yml` | Slipstream's dependencies, Cheetah, Ocelot, Xsuite (CPU), pytest | `conda_env = "ssv-core"` | base image (simcodes) already has them |
| `xsuite-gpu.yml` | Xsuite with CuPy, for the GPU rows | `conda_env = "ssv-xsuite-gpu"` | no (needs `--gpus`) |
| `bmadx.yml` | Bmad-X (torch) | `conda_env = "ssv-bmadx"` | `/opt/envs/bmadx` (checked: reproduces the archived tune error 1.304e-4) |
| `impactx.yml` | ImpactX (CPU, OpenMP), a released version | `conda_env = "ssv-impactx"` | no: conda-forge's 26.10 lacks `beam.at_fixed_t()`, which the `sc_drift` worker needs (the paper's run is a source build at `e4279a4b`); archived on Linux until a release has it |
| `scibmad.yml` | SciBmad (Julia 1.12 from the `bmad-sim` channel, 1.2 GB) | `conda_env = "ssv-scibmad"` | `/opt/envs/scibmad` (checked: reproduces the archive) |
| `julia.yml` | the Python side of JuTrack; `setup/julia.sh` installs the official Julia, `setup/jutrack.sh` the package | `conda_env = "ssv-julia"` | no: Enzyme does not precompile on Linux (Julia 1.10.4 and 1.10.10, WSL and Docker); JuTrack runs on Windows with Julia 1.10.4 and is archived on Linux |
| `helix.yml` | HELIX (`linac_gen`) | `conda_env = "ssv-helix"` | `/opt/envs/helix` (checked: reproduces the archived FODO error) |

Not conda: **ASTRA** and **elegant** (compiled binaries; the simcodes image carries them, or install simcodes natively and point
`BENCH_ASTRA_EXE` and `BENCH_ELEGANT_EXE` at them) and **GPT** (licensed; install it yourself, see `bench.toml`).

```bash
bash setup/install_envs.sh                 # all of them into $CONDA_BASE/envs (default ~/miniforge3)
bash setup/install_envs.sh bmadx scibmad   # some
```

The versions that produced the archived numbers are written next to each pin. A newer Cheetah, Xsuite or ImpactX is a finding,
not a failure: every result records the version it came from. Slipstream itself is never installed from here; the checkout under
test is put on `PYTHONPATH` by the runner (`--slipstream`).

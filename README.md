# slipstream-validation

Cross-code validation, benchmarks and archived evidence for [Slipstream](https://github.com/astec-stfc/slipstream).

The unit tests in the Slipstream repository check the package on its own and need no other code. This repository holds
everything that needs more: the other codes (ASTRA, elegant, Xsuite, Ocelot, Cheetah, GPT), the frozen benchmark cases, the
CLARA injector comparisons and the saved results that every number in the paper is recomputed from. Every check is a **gate**:
a quantity, a reference and a tolerance. A gate that fails means a bug, or a documented exception written next to it.

It follows [simba-integration-tests](https://github.com/astec-stfc/simba-integration-tests): a review is a folder, the codes run
in one Docker image, a campaign is kept under a tag, and `check` compares a campaign with another.

## Status

First scaffold (6 November 2026). Tier 0 runs: the 401 quoted-number checks of the paper and the framework tests, with numpy
only. The benchmark harness is ported to the image and all four cases are compared with the archive (25 jobs on this PC: all agree, one documented exception). The rest is planned in [DESIGN.md](DESIGN.md), which also says what a second PC can and cannot reproduce.

## Quick start (any PC, no simulation code)

```bash
git clone https://github.com/astec-stfc/slipstream-validation
cd slipstream-validation
pip install numpy pytest
pytest -m tier0
```

This recomputes every number quoted in the paper from `reviews/claims/evidence/` and writes `results/claims.csv`.

## Running the codes (needs Docker; all four cases reproduced on 6 November 2026)

```bash
docker pull ghcr.io/astec-stfc/simcodes-docker:master
docker build -t slipstream-validation:local -f docker/Dockerfile docker/
python -m slipstream_validation.runner checkout ../slipstream HEAD --name head
python -m slipstream_validation.runner run benchmark --execute --tag pc2 --slipstream checkouts/slipstream-head          # all four cases, about an hour on one CPU thread; add --case NAME for one
python -m slipstream_validation.check compare pc2
```

`bench.toml` says how each code is reached; copy it to `bench.local.toml` (not committed) to change it for one machine.

## Reviews

| Review | What it checks | Needs | State |
| --- | --- | --- | --- |
| [claims](reviews/claims/) | every number quoted in the paper follows from the archived evidence | numpy | runs (tier 0) |
| [benchmark](reviews/benchmark/) | the four frozen cases (ring tunes, five-knob Jacobian, space-charge drift, linear FODO) across the codes | the image | all four cases run and are compared with the archive by `check compare`; one documented exception (the archived elegant `ring_grad` reference) |
| injector | CLARA 20, 100 and 250 pC against refined ASTRA and GPT | the image, GPT archived | planned |

## Licence

Apache 2.0, see [LICENSE](LICENSE).

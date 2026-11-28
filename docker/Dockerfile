# The validation image: the simcodes image (ASTRA, elegant, Pelegant, CSRTrack, OPAL, Genesis, and in its Python Ocelot,
# Xsuite, Cheetah, MAD-X, Bmad/Tao, Wake-T, torch with CUDA 13, numpy, scipy, h5py, pandas, PyYAML) plus pytest and soliday.sdds,
# plus one conda environment per extra code, built from envs/*.yml with micromamba: Bmad-X, SciBmad (which brings its own Julia 1.12) and
# HELIX.  One image, one image id in every campaign; the environments do not touch the base Python.
#
# Slipstream is not in it, so that one image serves every commit: the checkout under test is what the
# runner mounts (/slipstream, PYTHONPATH=/slipstream/src).  GPT is not in it either (licence): its result is archived evidence
# until you install it, see bench.toml and DESIGN.md.  ImpactX and JuTrack are not in it either: see DESIGN.md section 12.
#
#   docker build -t slipstream-validation:local -f docker/Dockerfile .          (context: the repository root; .dockerignore keeps it small)
#   docker run --rm -v "$PWD/../slipstream:/slipstream:ro" -e PYTHONPATH=/slipstream/src -w /slipstream \
#       --entrypoint python slipstream-validation:local -m pytest tests/test_package_smoke.py -q
#
# `--target core` stops before the extra environments (the simcodes codes only); the nightly workflow builds that.
#
# The base is pinned by digest, because `:master` moves: this is the image of the campaigns from 6 November 2026 on.  To move it,
# change the digest, rebuild, run a campaign and `check compare` it.
#
# Add --gpus all to run on an NVIDIA card (needs the NVIDIA container toolkit; Docker Desktop on Windows passes the card
# through WSL 2).  Rebuild only when a code or a dependency changes; the extra environments are the last layers, so a change to
# one yml rebuilds them and not the base.
ARG SIMCODES_IMAGE=ghcr.io/astec-stfc/simcodes-docker@sha256:b4ed6130ec5dd9fe80eab0e5dc4477cea02af401ac4f76fae9fd186025cdbf8a
FROM ${SIMCODES_IMAGE} AS core

# soliday.sdds is Slipstream's [sdds] extra, installed by its CI too
RUN pip install --no-cache-dir pytest==9.2.0 soliday.sdds==5.7.6
ENV MPLBACKEND=Agg
# elegant, the SDDS tools its worker calls, and ASTRA; the benchmark workers read these two variables
ENV PATH=/simcodes/SDDS/bin/Linux-x86_64:/simcodes/elegant/bin/Linux-x86_64:$PATH
ENV BENCH_ELEGANT_EXE=elegant BENCH_ASTRA_EXE=/simcodes/Astra-serial/Astra
ENTRYPOINT ["/bin/bash"]
WORKDIR /work


FROM core AS full
# ---- the rest of the stack: micromamba, then one layer per environment (each yml is copied just before it is used, so changing one
# file rebuilds that environment and the ones after it, not the ones before)
COPY --from=mambaorg/micromamba:2.0.5 /bin/micromamba /usr/local/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/mamba
COPY envs/bmadx.yml /tmp/envs/
RUN micromamba create -y -q -p /opt/envs/bmadx -f /tmp/envs/bmadx.yml && micromamba clean -a -y
COPY envs/helix.yml /tmp/envs/
RUN micromamba create -y -q -p /opt/envs/helix -f /tmp/envs/helix.yml && micromamba clean -a -y
COPY envs/scibmad.yml /tmp/envs/
RUN micromamba create -y -q -p /opt/envs/scibmad -f /tmp/envs/scibmad.yml && micromamba clean -a -y

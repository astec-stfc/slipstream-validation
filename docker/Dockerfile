# The validation image: the simcodes image (ASTRA, elegant, Pelegant, CSRTrack, OPAL, Genesis, and in its Python Ocelot,
# Xsuite, Cheetah, MAD-X, Bmad/Tao, Wake-T, torch with CUDA 13, numpy, scipy, h5py, pandas, PyYAML) plus pytest and soliday.sdds.
#
# Slipstream is not in it: its repository is private, so a build cannot clone it, and the checkout under test is what the
# runner mounts (/slipstream, PYTHONPATH=/slipstream/src).  GPT, ImpactX, JuTrack and HELIX are not in it either (licence,
# conda build, Julia); their results are archived evidence, see DESIGN.md.
#
#   docker build -t slipstream-validation:local -f docker/Dockerfile docker/
#   docker run --rm -v "$PWD/../slipstream:/slipstream:ro" -e PYTHONPATH=/slipstream/src -w /slipstream \
#       --entrypoint python slipstream-validation:local -m pytest tests/test_package_smoke.py -q
#
# Add --gpus all to run on an NVIDIA card (needs the NVIDIA container toolkit; Docker Desktop on Windows passes the card
# through WSL 2).  Rebuild only when a code or a dependency changes.
ARG SIMCODES_IMAGE=ghcr.io/astec-stfc/simcodes-docker:master
FROM ${SIMCODES_IMAGE}

# soliday.sdds is Slipstream's [sdds] extra, installed by its CI too
RUN pip install --no-cache-dir pytest soliday.sdds
ENV MPLBACKEND=Agg
# elegant, the SDDS tools its worker calls, and ASTRA; the benchmark workers read these two variables
ENV PATH=/simcodes/SDDS/bin/Linux-x86_64:/simcodes/elegant/bin/Linux-x86_64:$PATH
ENV BENCH_ELEGANT_EXE=elegant BENCH_ASTRA_EXE=/simcodes/Astra-serial/Astra
ENTRYPOINT ["/bin/bash"]
WORKDIR /work

#!/usr/bin/env bash
# Install the official Julia 1.10.4 build (the version of the paper) into DEST.  The official build, not conda-forge's: Enzyme, which
# JuTrack depends on, does not precompile with conda-forge's Julia (`Missing Julia object for global '*Core.Intrinsics.mul_float'`).
#   [JULIA_VERSION=1.10.4] bash setup/julia.sh [DEST]          DEST defaults to $HOME/julia-$JULIA_VERSION
set -euo pipefail
ver="${JULIA_VERSION:-1.10.4}"
dest="${1:-$HOME/julia-$ver}"
if [ -x "$dest/bin/julia" ]; then echo "julia exists: $dest/bin/julia"; exit 0; fi
mkdir -p "$dest"
curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/${ver%.*}/julia-$ver-linux-x86_64.tar.gz" | tar -xz -C "$dest" --strip-components=1
"$dest/bin/julia" --version

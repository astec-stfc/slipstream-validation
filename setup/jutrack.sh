#!/usr/bin/env bash
# Clone JuTrack.jl at the commit of the paper and instantiate its Manifest with the Julia of setup/julia.sh.
#   bash setup/jutrack.sh [DEST]          DEST defaults to $HOME/JuTrack.jl (/opt/JuTrack.jl in the image)
#   BENCH_JULIA=/path/to/julia            default $HOME/julia-1.10.4/bin/julia
set -euo pipefail
dest="${1:-$HOME/JuTrack.jl}"
julia="${BENCH_JULIA:-$HOME/julia-1.10.4/bin/julia}"
[ -d "$dest/.git" ] || git clone https://github.com/MSU-Beam-Dynamics/JuTrack.jl "$dest"
git -C "$dest" checkout --quiet 5b5eebd
"$julia" --project="$dest" -e 'using Pkg; Pkg.instantiate(); using JuTrack; println("JuTrack ", pkgversion(JuTrack))'
echo "set BENCH_JULIA=$julia and BENCH_JUTRACK=$dest"

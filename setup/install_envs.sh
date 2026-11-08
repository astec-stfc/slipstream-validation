#!/usr/bin/env bash
# Create the conda environments of envs/ (all, or the names given: bmadx scibmad ...), the official Julia 1.10.4 and JuTrack.
#   bash setup/install_envs.sh [name ...]       CONDA_BASE (default ~/miniforge3) is the conda installation
# Prefers mamba when the base has it.  An existing environment is left alone (remove it first to rebuild).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
base="${CONDA_BASE:-$HOME/miniforge3}"
mgr="$base/bin/conda"; [ -x "$base/bin/mamba" ] && mgr="$base/bin/mamba"
names=("$@"); [ ${#names[@]} -eq 0 ] && names=(core xsuite-gpu bmadx impactx scibmad julia helix)
for n in "${names[@]}"; do
  env="ssv-$n"
  if [ -d "$base/envs/$env" ]; then echo "== $env exists"; continue; fi
  echo "== creating $env from envs/$n.yml"
  "$mgr" env create -y -f "$here/envs/$n.yml"
done
case " ${names[*]} " in *" julia "*)
  bash "$here/setup/julia.sh"
  bash "$here/setup/jutrack.sh" ;;
esac
echo "done; set the platform of each code in bench.local.toml (conda_env = \"ssv-NAME\"), see bench.toml"

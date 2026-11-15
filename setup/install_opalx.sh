#!/usr/bin/env bash
# OPAL-X 1.0.0 (IPPL + Kokkos): a CUDA build for the NVIDIA GPU of this host, an OpenMP build, and the official CPU image.
#   bash setup/install_opalx.sh [cuda] [cpu] [image]       default: cuda image
# Run it on Linux or inside WSL (it is the platform of the `opalx` worker); nothing is installed outside $OPALX_HOME.
#   OPALX_HOME  default ~/opalx            OPALX_TAG  default OPALX-1.0.0 (the manual's release tag)
#   CUDA_HOME   default /usr/local/cuda    GPU_ARCH   default AMPERE86 (RTX 3050); Kokkos names: AMPERE80 A100, ADA89 RTX 40xx, HOPPER90 ...
#   JOBS        default 8 (a CUDA build needs about 3 GB of RAM per job)
# Needs: git, cmake >= 3.24, gcc/g++/gfortran 13, an MPI (OpenMPI 4 or 5), and for CUDA the toolkit (12.9 recommended by the manual; 13.4 works on
# an RTX 3050).  Everything else (Kokkos 5.2, IPPL, heFFTe, HDF5, H5hut, LAPACK) is fetched by CMake and built; about 25 minutes with 8 jobs.
# Afterwards put the printed variables into the [codes.opalx] block of bench.local.toml.
set -euo pipefail
home="${OPALX_HOME:-$HOME/opalx}"
tag="${OPALX_TAG:-OPALX-1.0.0}"
jobs="${JOBS:-8}"
arch="${GPU_ARCH:-AMPERE86}"
todo=("$@"); [ ${#todo[@]} -eq 0 ] && todo=(cuda image)
mkdir -p "$home"
[ -d "$home/OPALX" ] || git clone --branch "$tag" --depth 1 https://github.com/OPALX-project/OPALX.git "$home/OPALX"
git -C "$home/OPALX" describe --tags --exact-match
export CC="${CC:-gcc-13}" CXX="${CXX:-g++-13}" FC="${FC:-gfortran-13}"
export OMPI_CC="$CC" OMPI_CXX="$CXX" OMPI_FC="$FC"     # the MPI wrappers must use the same compiler as the dependencies
for what in "${todo[@]}"; do
  case "$what" in
    cuda)
      export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
      export PATH="$CUDA_HOME/bin:$PATH" CUDACXX="$CUDA_HOME/bin/nvcc"
      cmake -S "$home/OPALX" -B "$home/OPALX/build_cuda" -DCMAKE_BUILD_TYPE=Release -DPLATFORMS=CUDA -DARCH="$arch"
      nice -n 10 cmake --build "$home/OPALX/build_cuda" -j "$jobs"
      echo "BENCH_OPALX_CUDA_EXE = $home/OPALX/build_cuda/src/opalx" ;;
    cpu)
      cmake -S "$home/OPALX" -B "$home/OPALX/build_openmp" -DCMAKE_BUILD_TYPE=Release -DPLATFORMS=OPENMP
      nice -n 10 cmake --build "$home/OPALX/build_openmp" -j "$jobs"
      echo "BENCH_OPALX_CPU_EXE = $home/OPALX/build_openmp/src/opalx" ;;
    image)
      docker pull ghcr.io/opalx-project/opalx:cpu
      echo "BENCH_OPALX_CPU_IMAGE = ghcr.io/opalx-project/opalx:cpu   (default; used when BENCH_OPALX_CPU_EXE is unset)" ;;
    *) echo "unknown target $what (cuda, cpu, image)"; exit 1 ;;
  esac
done

using Pkg
Pkg.activate(@__DIR__)
Pkg.add("CUDA"; io=devnull)
Pkg.precompile()
using CUDA, SciBmad
println("CUDA.jl ", pkgversion(CUDA), " functional=", CUDA.functional())
CUDA.versioninfo()

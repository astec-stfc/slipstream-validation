using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
using SciBmad
println("OK SciBmad loaded, julia ", VERSION)
Pkg.status()

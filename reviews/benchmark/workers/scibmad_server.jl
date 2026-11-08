# Persistent SciBmad process driven by w_scibmad.py over stdin/stdout.  Replies are prefixed "@@" so package noise is ignorable.
#   BUILD   <lattice.json> <method> <order> <n_steps> <E0_eV> <mass_eV> [device dtype]  ->  @@OK {inventory}   (device cpu|cuda, dtype float64|float32)
#   TRACK   <init.bin> <out.bin> <turns>                                              ->  @@OK lost=<n>
#   ADBUILD <lattice.json> <knobs.json> <method> <order> <n_steps> <E0_eV> <mass_eV>  ->  @@OK
#   ADJAC   <init.bin> <out.bin> <turns>                                              ->  @@OK     (writes y[4] then J[4,K], float64, column-major)
# method: "std" = SciBmadStandard as shipped (default fringes, exact maps where solvable, else Yoshida order 4 with one step);
#         "sym" = BeamTracking.Symplectic(order, n_steps) on quadrupoles and sextupoles with the hard-edge fringes off, exact sector bends.
# init.bin: N x 6 float64 (x px y py z pz) row-major; px = Px/P0 (canonical) and pz = delta, SciBmad's own coordinates.
# out.bin: N x T x 4 float64, state at the START of turns 0..T-1 (the initial state included).
using JSON
t0 = time()
using SciBmad, ForwardDiff
const USE_CUDA = get(ENV, "BENCH_SCIBMAD_CUDA", "0") == "1"   # set with --project=codes/scibmad_gpu (CUDA.jl lives there, not in the distribution)
USE_CUDA && @eval using CUDA
import SciBmad.BeamTracking as BT
println("@@READY import_s=", time() - t0, " julia=", VERSION, " threads=", Threads.nthreads(),
        " scibmad=", pkgversion(SciBmad), " beamtracking=", pkgversion(BT), " beamlines=", pkgversion(SciBmad.Beamlines),
        " gtpsa=", pkgversion(SciBmad.GTPSA)); flush(stdout)

p0c_eV(E0, m) = sqrt(E0^2 - m^2)

# Knob families (AD) take their strength from the beamline context through a deferred expression, so that the context can hold ForwardDiff.Dual values.
mkexpr(sym) = DefExpr(c -> getproperty(c, sym))

function build_line(lat, method, order, nsteps, E0, mass, famsyms = Dict{String,Tuple{Symbol,String}}())
    els = Any[]
    fam = Dict{String,Vector{Any}}()
    inv = Dict("n" => 0, "length" => 0.0, "angle" => 0.0, "k1L_abs" => 0.0, "k2L_abs" => 0.0)
    nofringe = BT.Fringe.NoEnd
    # attribution variants (7 Nov 2026): "symy" = Symplectic with the plain Yoshida triple-jump instead of the optimized (Suzuki) composition;
    # "dk" / "dky" = DriftKick (quadrupole as drift-kick, like elegant's KQUAD and Slipstream) with the optimized / Yoshida composition
    qmethod = method == "std" ? SciBmadStandard() :
              method == "sym"  ? BT.Symplectic(order = order, n_steps = nsteps, fringe_at = nofringe) :
              method == "symy" ? BT.Symplectic(order = order, n_steps = nsteps, fringe_at = nofringe, use_optimized_schemes = false) :
              method == "dk"   ? BT.DriftKick(order = order, n_steps = nsteps, fringe_at = nofringe) :
              method == "dky"  ? BT.DriftKick(order = order, n_steps = nsteps, fringe_at = nofringe, use_optimized_schemes = false) :
              error("unknown method $method")
    bmethod = method == "std" ? SciBmadStandard() : BT.Exact(fringe_at = nofringe)   # sextupoles also use qmethod; with DriftKick they are unchanged (dkd either way)
    for e in lat["elements"]
        L = Float64(e["length"]); kind = e["kind"]; nm = String(e["name"])
        if kind == "drift"
            ele = Drift(name = nm, L = L)
        elseif kind == "quad"
            ele = Quadrupole(name = nm, L = L, Kn1 = haskey(famsyms, nm) ? mkexpr(famsyms[nm][1]) : Float64(e["k1"]), tracking_method = qmethod); inv["k1L_abs"] += abs(e["k1"]) * L
        elseif kind == "sext"
            ele = Sextupole(name = nm, L = L, Kn2 = haskey(famsyms, nm) ? mkexpr(famsyms[nm][1]) : Float64(e["k2"]), tracking_method = qmethod); inv["k2L_abs"] += abs(e["k2"]) * L
        elseif kind == "bend"
            ele = SBend(name = nm, L = L, angle = Float64(e["angle"]), e1 = 0.0, e2 = 0.0, tracking_method = bmethod); inv["angle"] += e["angle"]
        else
            error("unknown kind $kind")
        end
        inv["length"] += L
        push!(els, ele)
        push!(get!(fam, nm, Any[]), ele)
    end
    inv["n"] = length(els)
    line = Beamline(els; species_ref = Species("electron"), pc_ref = p0c_eV(E0, mass))
    for (nm, (sym, kind)) in famsyms
        e0 = first(filter(e -> e["name"] == nm, lat["elements"]))
        setproperty!(line.context, sym, Float64(e0[kind]))
    end
    return line, fam, inv
end

readinit(path) = (N = filesize(path) ÷ (6 * 8); raw = Array{Float64}(undef, 6, N); read!(path, raw); Matrix{Float64}(permutedims(raw)))

line, adline, adfam, adknobs, adsyms, adtheta0 = nothing, nothing, nothing, nothing, nothing, nothing
dev, T = "cpu", Float64
for cmd in eachline(stdin)
    a = split(cmd)
    try
        if a[1] == "BUILD"
            lat = JSON.parsefile(a[2])
            global line, _, inv = build_line(lat, a[3], parse(Int, a[4]), parse(Int, a[5]), parse(Float64, a[6]), parse(Float64, a[7]))
            global dev = length(a) >= 9 ? a[8] : "cpu"
            global T = length(a) >= 9 && a[9] == "float32" ? Float32 : Float64
            dev == "cuda" && !USE_CUDA && error("device cuda needs BENCH_SCIBMAD_CUDA=1 and the codes/scibmad_gpu project")
            println("@@OK ", JSON.json(inv))
        elseif a[1] == "TRACK"
            n = parse(Int, a[4]); r = readinit(a[2]); N = size(r, 1)
            if dev == "cuda"
                res = track(line, v0 = CuArray(convert(Matrix{T}, r)), n_turns = n - 1)
                CUDA.synchronize()
                v = Array(res.v)                                       # device -> host copy, as the other adapters' calls include
                alive = Array(res.state)
            else
                res = track(line, v0 = convert(Matrix{T}, r), n_turns = n - 1)
                v = res.v                                              # N x 6 x T (the initial state is slice 1)
                alive = res.state
            end
            out = Array{Float64}(undef, 4, n, N)
            for i in 1:N, t in 1:n, c in 1:4
                out[c, t, i] = v[i, c, t]
            end
            lost = count(!=(0x1), alive[:, end])
            write(a[3], out)
            println("@@OK lost=", lost)
        elseif a[1] == "ADBUILD"
            lat = JSON.parsefile(a[2]); knobs = JSON.parsefile(a[3])
            famsyms = Dict{String,Tuple{Symbol,String}}(kn["family"] => (Symbol("knob", j), kn["kind"]) for (j, kn) in enumerate(knobs))
            global adline, adfam, _ = build_line(lat, a[4], parse(Int, a[5]), parse(Int, a[6]), parse(Float64, a[7]), parse(Float64, a[8]), famsyms)
            global adknobs = knobs
            global adsyms = [famsyms[kn["family"]][1] for kn in knobs]
            global adtheta0 = [Float64(first(filter(e -> e["name"] == kn["family"], lat["elements"]))[kn["kind"]]) for kn in knobs]
            println("@@OK")
        elseif a[1] == "ADJAC"
            n = parse(Int, a[4]); r = readinit(a[2])
            function f(θ)
                for (j, sym) in enumerate(adsyms)
                    setproperty!(adline.context, sym, θ[j])
                end
                res = track(adline, v0 = convert(Matrix{eltype(θ)}, r), n_turns = n - 1)   # the coordinates must carry the same (Dual) type as the parameters
                return [res.v[1, 1, end], res.v[1, 2, end], res.v[1, 3, end], res.v[1, 4, end]]
            end
            J = ForwardDiff.jacobian(f, adtheta0)
            y = f(adtheta0)                                            # primal run at the nominal strengths (also restores the context)
            write(a[3], vcat(Float64.(y), vec(Float64.(J))))
            println("@@OK")
        else
            error("bad command $cmd")
        end
    catch err
        println("@@ERR ", sprint(showerror, err, catch_backtrace()))
    end
    flush(stdout)
end

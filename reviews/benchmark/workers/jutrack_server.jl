# Persistent JuTrack process driven by w_jutrack.py over stdin/stdout.  Replies are prefixed "@@" so package noise is ignorable.
#   BUILD <lattice.json> <n_steps_quad_sext> <n_steps_bend> <E0_eV> <mass_eV>   ->  @@OK {inventory}
#   TRACK <init.bin> <out.bin> <turns>                                           ->  @@OK
# init.bin: N x 6 float64 (x px y py z delta) row-major.  out.bin: N x T x 4 float64, state at the START of turns 0..T-1.
using JSON
t0 = time()
using JuTrack
println("@@READY import_s=", time() - t0, " julia=", VERSION, " threads=", Threads.nthreads()); flush(stdout)

line, E0, mass = nothing, 0.0, 0.0
for cmd in eachline(stdin)
    a = split(cmd)
    try
        if a[1] == "BUILD"
            lat = JSON.parsefile(a[2]); k = parse(Int, a[3]); kb = parse(Int, a[4])
            global E0 = parse(Float64, a[5]); global mass = parse(Float64, a[6])
            els = AbstractElement{Float64}[]
            inv = Dict("n" => 0, "length" => 0.0, "angle" => 0.0, "k1L_abs" => 0.0, "k2L_abs" => 0.0)
            for e in lat["elements"]
                L = Float64(e["length"]); kind = e["kind"]; nm = String(e["name"])
                if kind == "drift"
                    push!(els, DRIFT(name = nm, len = L))
                elseif kind == "quad"
                    push!(els, KQUAD(name = nm, len = L, k1 = Float64(e["k1"]), NumIntSteps = k)); inv["k1L_abs"] += abs(e["k1"]) * L
                elseif kind == "sext"
                    push!(els, KSEXT(name = nm, len = L, k2 = Float64(e["k2"]), NumIntSteps = k)); inv["k2L_abs"] += abs(e["k2"]) * L
                elseif kind == "bend"
                    push!(els, SBEND(name = nm, len = L, angle = Float64(e["angle"]), e1 = 0.0, e2 = 0.0, NumIntSteps = kb,
                                     FringeBendEntrance = 0, FringeBendExit = 0)); inv["angle"] += e["angle"]
                else
                    error("unknown kind $kind")
                end
                inv["length"] += L
            end
            global line = els; inv["n"] = length(els)
            println("@@OK ", JSON.json(inv))
        elseif a[1] == "TRACK"
            n = parse(Int, a[4]); N = filesize(a[2]) ÷ (6 * 8)
            raw = Array{Float64}(undef, 6, N); read!(a[2], raw)
            r = Matrix{Float64}(permutedims(raw))                     # N x 6
            beam = Beam(r; energy = E0 - mass, mass = mass)           # JuTrack's `energy` is kinetic: gamma = (energy + mass) / mass
            saved = ringpass!(line, beam, n, true)                    # copies of r after turns 1..T
            out = Array{Float64}(undef, 4, n, N)
            out[:, 1, :] = raw[1:4, :]
            for t in 2:n, c in 1:4, i in 1:N
                out[c, t, i] = saved[t-1][i, c]
            end
            lost = count(!=(0), beam.lost_flag)
            write(a[3], out)
            println("@@OK lost=", lost)
        else
            error("bad command $cmd")
        end
    catch err
        println("@@ERR ", sprint(showerror, err, catch_backtrace())); 
    end
    flush(stdout)
end

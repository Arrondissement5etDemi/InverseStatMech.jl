include("Particle.jl")
include("helper_funcs.jl")
include("Box.jl")

function reverse_mc!(b::Box, g2_targ::Function;
        sweeps::Integer=100, displace::Real=0.1, t_i::Real=1, t_f::Real=0.001, cooling_rate::Real=0.98)
    d = b.d
    bin_size = 0.02
    range = 5 #d/2
    old_rdf = b.rdf(bin_size, range)
    pretty_print(old_rdf)
    new_rdf = deepcopy(old_rdf)
    old_e = g2_dist(b, old_rdf, g2_targ)
    print(string(old_e)*"\n")
    new_e = old_e
    temperature = t_i
    while temperature >= t_f
        print("t ="*string(temperature)*"\n")
        for i = 1:sweeps
            n_success_move = 0
            for j = 1:b.n
                pj = b.partiArr[j]
                old_coord = deepcopy(pj.x)
                old_rdf_j = b.rdf_j(j, bin_size, range)
                b.move(j, displace)
                new_rdf_j = b.rdf_j(j, bin_size, range)
                new_rdf = deepcopy(old_rdf)
                new_rdf[:, 2] += (-old_rdf_j + new_rdf_j)
                new_e = g2_dist(b, new_rdf, g2_targ)
                if !boltzmann_accept(old_e, new_e, temperature)
                    pj.x = old_coord
                else
                    old_e = new_e
                    old_rdf = deepcopy(new_rdf)
                    n_success_move += 1
                end
            end
        end
        println(string(old_e)*" "*string(displace)*"\n")
        pretty_print(new_rdf)
        #reduce the temperature now
        temperature *= cooling_rate
        #print the box
        print(b.to_string())
        flush(stdout)
    end
    println(b.to_string())
    return nothing
end

function append_to_file(filename::String, whatever::String) 
    open(filename, "a") do f
        write(f, whatever)
    end
    return nothing
end

function g2_dist(b::Box, rdf, g2_targ::Function, bin_size::Real=0.02)
        result = 0
        for i = 1:size(rdf)[1]
            r = rdf[i, 1]
            gr = rdf[i, 2]
            result += surface_area_sph(b.dim, r)*bin_size*(g2_targ(r) - gr)^2
        end
    return result
end

function parse_g2()
    f = open("Targ.g2", "r")
    nlines = 374
    g2_data = zeros(nlines, 2)
    for i = 1:nlines
        line = readline(f)
        g2_data[i, :] = reshape(parse.(Float64, split(line)), 1, 2)
    end
    close(f)
    function g2(x)
        bin = min(max(Int(fld(x - 0.06, 0.08)) + 1, 1), nlines)
        return g2_data[bin, 2]
    end
    return g2
end

function main3D()
    dim = 3
    n = 5000
    ρ = 0.114
    b = #Box(dim, n, (n/ρ)^(1/dim))
    box_from_file(dim, n, (n/ρ)^(1/dim), "configN5000.snap")
    targ_g2 = parse_g2()
    reverse_mc!(b, targ_g2, t_i = 0.11, sweeps = 50)
end

main3D()


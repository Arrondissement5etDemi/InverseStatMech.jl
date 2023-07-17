include("box.jl")

"""
    reverse_mc(dim, n, ρ, g2_targ; initial_box = missing, bin_size = 0.05, range = 5, sweeps = 100, displace = 0.1, t_i = 1, t_f = 0.001, cooling_rate = 0.98)

Reverse Monte Carlo algorithm to generate equilibrium classical configurations that mimic the pair statistics of free and interacting Fermi gases.

# Arguments

- `dim::Int`: Dimensionality of the system.
- `n::Int`: Number of particles.
- `ρ::Float64`: Number density of the particles.
- `g2_targ::Function`: Target pair correlation function as a function `g2_targ(r)`.

# Keyword arguments

- `initial_box::Box` (optional): Initial configuration box. Default is `missing` which generates a random box.
- `bin_size::Float64` (optional): Bin size for computing the pair correlation function. Default is 0.05.
- `range::Float64` (optional): Range for the interaction potential. Default is 5.
- `sweeps::Int` (optional): Number of Monte Carlo sweeps at each temprature. Default is 100.
- `displace::Float64` (optional): Maximum displacement for particle moves. Default is 0.1.
- `t_i::Float64` (optional): Initial temperature. Default is 1.
- `t_f::Float64` (optional): Final temperature. Default is 0.001.
- `cooling_rate::Float64` (optional): Cooling rate for temperature reduction. Default is 0.98.

# Returns
- `b::Box`: Generated equilibrium classical configuration box. Use `b.particles'` to get the particle positions.

# Description

The `reverse_mc` function uses the Reverse Monte Carlo algorithm to generate equilibrium classical configurations that closely match the pair statistics (pair correlation function) of free and interacting Fermi gases. It iteratively adjusts the particle positions to match the desired pair correlation function, starting from an initial configuration or a random box.

The algorithm performs Monte Carlo sweeps where each particle is randomly moved within a specified displacement limit. The acceptance or rejection of each move is determined based on the energy change (given by the difference in the pair correlation function) and the current temperature. The temperature is gradually reduced during the sweeps using a cooling rate.

The `g2_targ` argument is the target pair correlation function, specified as a Julia function `g2_targ(r)` that takes the distance `r` as input and returns the desired value of the pair correlation function.

The generated equilibrium classical configuration is returned as a `Box` object containing the particle positions.

# Example

    box = Spec2Struc.reverse_mc(2, 100, 0.5, r -> 1 - exp(π*r^2))

"""
function reverse_mc(dim, n, ρ, g2_targ; 
        initial_box = missing, bin_size = 0.05, range = 5, sweeps = 100, displace = 0.1, t_i = 1, t_f = 0.001, cooling_rate = 0.98, test = false)
    if ismissing(initial_box)
        b = random_box(dim, n, ρ, bin_size, range)
    else
        b = initial_box
    end
    l = b.l
    old_rdf = b.compute_g2()
    println("Initial g2:")
    pretty_print(old_rdf)
    new_rdf = deepcopy(old_rdf)
    old_e = g2_dist(b, old_rdf, g2_targ)
    initial_e = old_e
    test_passed = false
    println("Initial energy: " * string(old_e))
    new_e = old_e
    temperature = t_i
    while temperature >= t_f
        println("Temperature ="*string(temperature)*"\n")
        for i = 1:sweeps
            for j = 1:b.n()
                pj = b.particles[:, j]
                old_coord = deepcopy(pj)
                old_rdf_j = b.rdf_j(j)
                b.move!(j, displace)
                new_rdf_j = b.rdf_j(j)
                new_rdf = deepcopy(old_rdf)
                new_rdf[:, 2] += (-old_rdf_j[:, 2] + new_rdf_j[:, 2])
                new_e = g2_dist(b, new_rdf, g2_targ)
                if !boltzmann_accept(old_e, new_e, temperature)
                    b.particles[:, j] = old_coord[:]
                else
                    old_e = new_e
                    old_rdf = deepcopy(new_rdf)
                end
            end
        end
        if old_e < initial_e
            test_passed = true
        end
        if test_passed && test
            break
        end
        println("New energy: " * string(old_e))
        println("New g2: ")
        pretty_print(new_rdf)
        #reduce the temperature
        temperature *= cooling_rate
        #print the box
        println("Configuration:")
        pretty_print(b.particles')
        flush(stdout)
    end
    pretty_print(b.particles')
    if test 
        return test_passed
    else
        return b
    end
end

function g2_dist(b, rdf, g2_targ)
        result = 0
        for i = 1:size(rdf, 1)
            r = rdf[i, 1]
            gr = rdf[i, 2]
            result += surface_area_sph(b.dim(), r)*b.bin_size*(g2_targ(r) - gr)^2
        end
    return result
end


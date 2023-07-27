using ForwardDiff, Interpolations
"""
    iterative_boltzmann(pot::Function, dim::Int, ρ::Float64, targ_g2::Function, α = 1; n = 500, bin_size = 0.05, r_range = 10)::Function

Iteratively updates the pair potential `pot` using the Boltzmann inversion method to match the target pair correlation function `targ_g2`.

# Arguments

- `pot::Function`: The initial pair potential function to be optimized.
- `dim::Int`: The dimensionality of the system.
- `ρ::Float64`: The number density of particles in the system.
- `targ_g2::Function`: The target pair correlation function `g_2(r)`.
- `α::Float64`: The update parameter for the potential. (Optional, default: 1)

# Keyword Arguments (All are optional)

- `n`: Number of configurations for each box in the simulation. (default: 500)
- `bin_size`: Bin size for the histograms of pair correlation functions. (default: 0.05)
- `r_range`: Maximum distance to compute the pair correlation function. (default: 10)
- `n_threads`: Number of threads to use for parallel computation. (default: 15)
- `configs_per_thread`: Number of configurations to generate for each thread. (default: 10)
- `displace`: Kick size in the metropolis Monte Carlo simulation. Default value is 0.2.
- `Ψ_tol`: Tolerance for stopping criterion. (default: 0.005)
- `show_pb`: Whether to show the progress bar during the simulation. (default: true)
- `test`: Whether to run the function in test mode. (default: false)

# Returns

- If `test=true`, returns a boolean indicating whether the optimization is successful.
- Otherwise, returns the optimized pair potential function.

# Example
    optimized_potential = iterative_boltzmann(r -> 0, 2, 1.0, r -> 1 - exp(-π*r^2))
"""
function iterative_boltzmann(pot, dim, ρ, targ_g2, α = 1; 
        n = 500, bin_size = 0.05, r_range = 10, 
        n_threads = 15, configs_per_thread = 10, displace = 0.2, Ψ_tol = 0.005, show_pb = true, test = false)
    f_g2(b) = b.compute_g2()
    threadarr = missing
    current_Ψ = 100
    round = 1
    max_round = ifelse(test, 10, 100)
    while current_Ψ > Ψ_tol && round <= max_round
        println("round " * string(round))
        #do the simulation
        boxarr, threadarr = simu_boxes((r, params) -> pot(r), [0], dim, n, ρ, bin_size, r_range, missing, threadarr; 
            n_threads = n_threads, configs_per_thread = configs_per_thread, show_pb = show_pb)
        n_configs = length(boxarr)
        weights_uniform = ones(n_configs)/n_configs
        #compute direct space pair statistics
        g2_arr = ThreadsX.map(f_g2, boxarr)
        g2 = reweighed_f(weights_uniform, g2_arr)
        r_vec = g2[:, 1]
        println("\ng2:")
        Base.print_matrix(stdout, g2[1:Int(min(length(r_vec), 100)), :])
        #prepare target g2 data and potential data 
        targ_g2_data = targ_g2.(r_vec)
        pot_data = pot.(r_vec)
        println("\nPotential:")
        Base.print_matrix(stdout, hcat(r_vec, pot_data))
        #compute the error metric
        current_Ψ = bin_size*sum([(g2[i, 2] - targ_g2_data[i])^2*surface_area_sph(dim, r_vec[i]) for i in eachindex(r_vec)]*ρ)
        println("\nΨ: " * string(current_Ψ))
        flush(stdout)
        #update the potential
        pot_data_new = zeros(0, 2)
        for i in eachindex(r_vec)
            if targ_g2_data[i]*g2[i] ≠ 0
                pot_data_new = vcat(pot_data_new, [r_vec[i], pot_data[i] - α*log(targ_g2_data[i]/g2[i, 2])]')
            end
        end
        pot_interp = Interpolations.linear_interpolation(pot_data_new[:, 1], pot_data_new[:, 2], extrapolation_bc = Line())
        pot = r -> ifelse(r < r_range, pot_interp(r), pot_interp(r_range))
        #start next round
        round += 1
    end
    if test
        return current_Ψ < Ψ_tol
    else
        return pot
    end
end


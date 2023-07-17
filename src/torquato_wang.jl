using ForwardDiff
include("box.jl")

"""
    optim_parametrized_pot(my_params, pot, dim, ρ, targ_g2, targ_s; 
        large_r_grid = missing, n::Int = 600, bin_size::Float64 = 0.05, r_range::Float64 = 10, k_range::Float64 = 10,
        g2_weight_range::Float64 = 2, s_weight_range::Float64 = 4, 
        n_threads::Int = 15, configs_per_box::Int = 10, Ψ_tol::Float64 = 0.005, show_pb::Bool = true, test::Bool = false)

Using the Torquato-Wang algorithm to perform iterative optimization of potential parameters to match target pair correlation function and structure factor.

## Arguments
- `my_params`: Vector of initial potential parameters.
- `pot`: Potential function that calculates the interaction potential between particles.
- `dim::Int`: Dimension of the system.
- `ρ::Float64`: Density of the system.
- `targ_g2::Function`: Function representing the target pair correlation function. Accepts a distance value `r` and returns the target g2 value at that distance.
- `targ_s::Function`: Function representing the target structure factor. Accepts a wave vector `k` and returns the target S value at that wave vector.

## Keyword Arguments
- `large_r_grid::Missing`: Large-r grid for computation of long-ranged potentials. Default value is `missing`.
- `n::Int`: Number of boxes for simulation. Default value is `600`.
- `bin_size::Float64`: Size of the bin for pair correlation function and structure factor calculations. Default value is `0.05`.
- `r_range::Float64`: Range of r values for pair correlation function calculation. Default value is `10`.
- `k_range::Float64`: Range of k values for structure factor calculation. Default value is `10`.
- `g2_weight_range::Float64`: Weight range for pair correlation function in the objective function. Default value is `2`.
- `s_weight_range::Float64`: Weight range for structure factor in the objective function. Default value is `4`.
- `n_threads::Int`: Number of threads for parallel computation. Default value is `15`.
- `configs_per_box::Int`: Number of configurations per box for simulation. Default value is `10`.
- `Ψ_tol::Float64`: Tolerance for convergence of the objective function. Default value is `0.005`.
- `show_pb::Bool`: Boolean indicating whether to display a progress bar during simulation. Default value is `true`.
- `test::Bool`: Boolean flag to indicate whether this is a test run and return a boolean indicating convergence. Default value is `false`.

## Returns
- If `test` is true, returns `true` if convergence is achieved, `false` otherwise.
- If `test` is false, returns the optimized potential parameters.

"""
function optim_parametrized_pot(my_params, pot, dim, ρ, targ_g2, targ_s; 
        large_r_grid = missing, n = 600, bin_size = 0.05, r_range = 10, k_range = 10, 
        g2_weight_range = 2, s_weight_range = 4,
        n_threads = 15, configs_per_box = 10, Ψ_tol = 0.005, show_pb = true, test = false)
    f_g2(b) = b.compute_g2()
    f_s(b) = struc_fac(b.particles, b.l, k_range, bin_size)
    threadarr = missing
    current_Ψ = 100
    round = 1
    max_round = ifelse(test, 10, 100)
    while current_Ψ > Ψ_tol && round <= max_round
        println("round " * string(round))
        #do the simulation
        boxarr, threadarr = simu_boxes(pot, my_params, dim, n, ρ, bin_size, r_range, large_r_grid, threadarr; 
            n_threads = n_threads, configs_per_box = configs_per_box, show_pb = show_pb)
        n_configs = length(boxarr)
        weights_uniform = ones(n_configs)/n_configs
        #compute direct space pair statistics
        g2_arr = ThreadsX.map(f_g2, boxarr)
        g2_old = reweighed_f(weights_uniform, g2_arr)
        r_vec = g2_old[:, 1]
        println("\ng2:")
        Base.print_matrix(stdout, g2_old[1:Int(min(length(r_vec), 100)), :])
        #compute fourier space pair statistics
        s_arr = ThreadsX.map(f_s, boxarr)
        s_old = reweighed_f(weights_uniform, s_arr)
        k_vec = s_old[:, 1]
        println("\nS:")
        Base.print_matrix(stdout, s_old)
        #create target
        targ_g2_data = hcat(r_vec, targ_g2.(r_vec))
        targ_s_data = hcat(k_vec, targ_s.(k_vec))
        #the objective function
        function Ψ(params_new)
            pot_old(r) = pot(r, my_params)
            pot_new(r) = pot(r, params_new)
            weights_new = reweigh(boxarr, pot_old, pot_new, g2_arr)
            g2_new = reweighed_f(weights_new, g2_arr)
            s_new = reweighed_f(weights_new, s_arr)
            return bin_size*
            (sum([(g2_new[i, 2] - targ_g2_data[i, 2])^2*exp(-(r_vec[i]/g2_weight_range)^2)*surface_area_sph(dim, r_vec[i]) for i in eachindex(r_vec)]*ρ) +
             sum([(s_new[i, 2] - targ_s_data[i, 2])^2*exp(-(k_vec[i]/s_weight_range)^2)*surface_area_sph(dim, k_vec[i]) for i in eachindex(k_vec)]/((2*pi)^dim*ρ)))
        end
        #optimize the potential parameters
        opt = Optim.optimize(Ψ, my_params; method = BFGS(), autodiff = :forward, show_trace = true, iterations = 60)
        my_params = Optim.minimizer(opt)
        println(my_params)
        current_Ψ = Optim.minimum(opt)
        println(current_Ψ)
        flush(stdout)
        #start next round
        round += 1
    end
    if test
        return current_Ψ < Ψ_tol
    else
        return my_params
    end
end

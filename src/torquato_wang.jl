using ForwardDiff
include("box.jl")

"""
    optim_parametrized_pot(my_params; <keyword arguments>)

Optimize the parameters of a parametrized potential, starting with the initial guesses my_params

# Arguments

- `pot`: the functional form of the potential
- `large_r_grid = missing`: ewald summation to treat very long-ranged interactions
- `dim`: dimension of the system
- `n`: number of particles
- `ρ`: number density
- `bin_size`: bin size for the pair correlation function g_2
- `r_range`: the range of r in direct space to compute the pair correlation function g_2(r)
- `k_range`: the range of k in fourier space to compute the structure factor S(k)
- `targ_g2`: target pair correlation function
- `g2_weight_range`: range parameter of the weight function for g_2: w_g2(r) = exp(-(r/g2_weight_range)^2)
- `targ_s`: target structure factor
- `s_weight_range`: range parametrer of the weight function for S: w_s(k) = exp(-(k/g2_weight_range)^2)
- `n_threads`: number of threads used for parallel computation of the simulation
- `configs_per_box`: number of configurations to generate in each thread
- `Ψ_tol`: stopping criterion for the error functional Ψ (distance between target and optimized pair statistics)
- `show_pb = true`: show progress bar
- `test = false`: whether we are testing the package
"""
function optim_parametrized_pot(my_params; 
        pot, large_r_grid = missing, dim, n, ρ, bin_size, r_range, k_range, targ_g2, g2_weight_range, targ_s, s_weight_range, 
        n_threads, configs_per_box, Ψ_tol, show_pb = true, test = false)
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
    return current_Ψ < Ψ_tol, my_params
end

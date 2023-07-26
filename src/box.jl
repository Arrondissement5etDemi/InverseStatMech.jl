using LinearAlgebra, Plots, IterTools, Distributed, SharedArrays, DelimitedFiles, ProgressMeter, ThreadsX

mutable struct Box
    ###fields below###
    particles 
    l #box side length
    bin_size
    range
    n_cells_per_side
    cell_list #cell list to accelerate MC simulations
    energies #all the particle energies
    ###functions below###
    dim
    n
    ρ
    ρInv
    neighbors_j
    rdf_j
    compute_g2
    energy_j
    compute_energies!
    move!
    attempt_move!
    visualize
    
    function Box(_particles, _l, _bin_size, _range)
        this = new()
        this.particles = _particles
        this.l = _l
        this.bin_size = _bin_size
        this.range = _range
        dim, n = size(this.particles)
        this.n_cells_per_side = floor(Int, n^(1/dim))
        this.cell_list = fill([], ntuple(i -> this.n_cells_per_side, dim)) #to store indices of particles in the cells
        for i in 1:size(this.particles, 2)
            cell_inds = Int.(floor.(this.particles[:, i]/this.l*this.n_cells_per_side) .+ 1)
            this.cell_list[cell_inds...] = vcat(this.cell_list[cell_inds...], [i])
        end
        this.energies = zeros(n)
        
        this.dim = function()
            return size(this.particles, 1)
        end

        this.n = function()
            return size(this.particles, 2)
        end

        this.ρ = function()
            return this.n()/(this.l)^this.dim()
        end

        this.ρInv = function()
            return (this.l)^this.dim()/this.n()   
        end

        this.neighbors_j = function(j)
            pj = this.particles[:, j]
            cell_length = this.l/this.n_cells_per_side
            central_inds = Int.(ceil.(pj/cell_length))
            surrounding_ind = Int(ceil(this.range/cell_length))
            result = []
            for i in IterTools.product(ntuple(i -> -surrounding_ind:surrounding_ind, this.dim())...)
                current_cell_ind = mod.(central_inds + collect(i) .- 1, this.n_cells_per_side) .+ 1
                current_neighbors = this.cell_list[current_cell_ind...]
                result = vcat(result, current_neighbors)
            end
            return unique(result)
        end
        
        this.rdf_j = function(j) 
            pj = this.particles[:, j]
            r_vec = this.bin_size/2:this.bin_size:this.range
            g2_vec = zeros(length(r_vec))
            all_neighbors = 1:this.n()#this.neighbors_j(j)
            ```now run the computing of dist_pbc in parallel```
            for neighbor in all_neighbors
                if neighbor ≠ j
                    dist = dist_pbc(pj, this.particles[:, neighbor], this.l)
                    bin = ceil(Int, dist/this.bin_size)
                    if bin ≤ length(g2_vec)
                        r = r_vec[bin]
                        g2_vec[bin] += 1/(this.n()*this.ρ()*(volume_sph(dim, r + this.bin_size/2) - volume_sph(dim, r - this.bin_size/2)))
                    end
                end
            end
            return hcat(r_vec, g2_vec)
        end

        this.compute_g2 = function()
            result = this.rdf_j(1)
            for j = 2:this.n()
                result[:, 2] += this.rdf_j(j)[:, 2]
            end
            return result
        end

        this.energy_j = function(j, pair_pot, use_cl = true, iso = true)
            pj = this.particles[:, j]
            if use_cl
                all_neighbors = this.neighbors_j(j)
            else
                all_neighbors = 1:this.n()
            end
            result = 0
            for neighbor in all_neighbors
                if neighbor ≠ j 
                    if iso #isotropic potential
                        dist = dist_pbc(pj, this.particles[:, neighbor], this.l)
                        result += pair_pot(dist)
                    else #anisotropic potential
                        rij_vec = min_symm.(abs.(pj - this.particles[:, neighbor]), this.l)
                        result += pair_pot(rij_vec)
                    end
                end
            end
            return result
        end

        this.compute_energies! = function(pair_pot, use_cl = true, iso = true)
            for j = 1:this.n()
                this.energies[j] = this.energy_j(j, pair_pot, use_cl, iso)
            end
        end

        this.move! = function(j, displace, use_cl = true, displace_vec = missing)
            if ismissing(displace_vec)
                displace_vec = (rand(this.dim())*2 .- 1)*displace
            end
            cell_inds_old = Int.(floor.(this.particles[:, j]/this.l*this.n_cells_per_side) .+ 1)
            this.particles[:, j] = mod.(this.particles[:, j] + displace_vec, this.l) #move it
            #update cell list
            cell_inds_new = Int.(floor.(this.particles[:, j]/this.l*this.n_cells_per_side) .+ 1)
            if cell_inds_old ≠ cell_inds_new
                filter!(x -> x ≠ j, this.cell_list[cell_inds_old...])
                this.cell_list[cell_inds_new...] = vcat(this.cell_list[cell_inds_new...], [j])
            end
            return displace_vec
        end

        ```proposes to move the \$j-th particle```
        this.attempt_move! = function(j, displace, temperature, pair_pot, use_cl = true, iso = true)
            old_e = this.energies[j]
            displace_vec = this.move!(j, displace, use_cl)
            new_e = this.energy_j(j, pair_pot, use_cl, iso)
            threshold = min(1, exp(-(new_e - old_e)/temperature))
            if rand() > threshold
                this.move!(j, displace, use_cl, -displace_vec)
            else
                this.energies[j] = new_e
            end
        end

        this.visualize = function()
            x = vec(this.particles[1, :])
            y = vec(this.particles[2, :])
            if this.dim() == 2
                return Plots.scatter(x, y, markersize = 0.5)
            elseif this.dim() == 3
                z = vec(this.particles[3, :])
                return Plots.scatter(x, y, z, markersize = 0.5)
            end
        end
        return this
    end
end

function random_box(dim, n, ρ, bin_size, range)
    l = (n/ρ)^(1/dim)
    particles = rand(dim, n)*l
    b = Box(particles, l, bin_size, range)
    return b
end

function box_from_file(dim, n, ρ, filename, bin_size, range)
    particles = readdlm(filename, '\t', Float64, '\n')'
    l = (n/ρ)^(1/dim)
    b = Box(particles, l, bin_size, range)
    return b
end

function equilibrate!(b::Box, pair_pot, pot_params, use_cl = true, large_r_grid = missing, pair_pot_faster = missing; 
        temperature = 1, sweeps = 100, displace = 0.2, show_pb = false)
    iso = ismissing(large_r_grid) 
    if ismissing(pair_pot_faster) #bin the potential for faster evaluation
        r_vec = b.bin_size/2:b.bin_size:(b.l/2 + b.bin_size)
        if iso
            pot_grid = [pair_pot(r, pot_params) for r in r_vec]
        else
            small_r_grid = [pair_pot(norm(collect(r)), pot_params) for r in IterTools.product(ntuple(i -> r_vec, b.dim())...)]
            pot_grid = large_r_grid + small_r_grid
        end
        pair_pot_faster = make_faster(b.bin_size, pot_grid)
    end
    b.compute_energies!(pair_pot_faster, use_cl, iso)
    if show_pb
        @showprogress for i = 1:sweeps
            for j = 1:b.n()
                b.attempt_move!(j, displace, temperature, pair_pot_faster, use_cl, iso)
            end
        end
    else
        for i = 1:sweeps
            for j = 1:b.n()
                b.attempt_move!(j, displace, temperature, pair_pot_faster, use_cl, iso)
            end
        end
    end
    return pair_pot_faster
end

function g2_boxes(boxarr)
    b1 = boxarr[1]
    r_vec = b1.bin_size/2:b1.bin_size:b1.range
    result = zeros(length(r_vec))
    for box in boxarr
        result += box.compute_g2()[:, 2]
    end
    return hcat(r_vec, result/length(boxarr))
end

function reweigh(boxarr, pot_old, pot_new, g2_arr; temperature = 1)
    function total_energy(i, pot)
        b, g2 = boxarr[i], g2_arr[i]
        n, ρ, bin_size, dim = b.n(), b.ρ(), b.bin_size, b.dim()
        r_vec = g2[:, 1]
        result = 0
        for j in eachindex(r_vec)
            r = r_vec[j]
            factor = ρ*(v1(r + bin_size/2, dim) - v1(r - bin_size/2, dim))*n/2
            result += pot(r)*g2[j, 2]*factor
        end
        return result
    end
    n_boxes = length(boxarr)
    old_energies = total_energy.(1:n_boxes, pot_old)
    new_energies = total_energy.(1:n_boxes, pot_new)
    log_weights = -(new_energies - old_energies)./temperature
    log_weights_shifted = log_weights .- log_weights[1]
    weights = exp.(log_weights_shifted) 
    result = weights/sum(weights) #normalize
    return result
end

function reweighed_f(weights, f_data)
    return sum([f_data[i]*weights[i] for i in eachindex(f_data)])
end

function s_boxes(boxarr, k_range)
    l = boxarr[1].l
    result = struc_fac(boxarr[1].particles, l, k_range)
    for i in 2:length(boxarr)
        result[:, 2] += struc_fac(boxarr[i].particles, l, k_range)[:, 2]
    end
    result[:, 2] /= length(boxarr)
    return result
end

```simulate g2```
function simu_boxes(pair_pot, pot_params, dim, n, ρ, bin_size, range, large_r_grid, box_arr = missing; 
        n_threads = 20, configs_per_box = 5, show_pb)
    config_arr = Vector{Box}(undef, n_threads*configs_per_box)
    if ismissing(box_arr)
        box_arr = Vector{Box}(undef, n_threads)
        b = random_box(dim, n, ρ, bin_size, range)
        pair_pot_faster = equilibrate!(b, pair_pot, pot_params, false, large_r_grid; sweeps = 600, show_pb = show_pb) #initial equilibration
        for i in eachindex(box_arr)
            box_arr[i] = deepcopy(b)
        end
    else
        pair_pot_faster = equilibrate!(box_arr[1], pair_pot, pot_params, false, large_r_grid; sweeps = 1, show_pb = show_pb)
    end
    if show_pb
        p = Progress(n_threads*configs_per_box)
        @sync for i in 1:n_threads
            Threads.@spawn for j = 1:configs_per_box
                equilibrate!(box_arr[i], pair_pot, pot_params, false, large_r_grid, pair_pot_faster; sweeps = 50)
                config_arr[(i - 1)*configs_per_box + j] = deepcopy(box_arr[i])
                next!(p)
                flush(stdout)
            end
        end
        finish!(p)
    else
        @sync for i in 1:n_threads
            Threads.@spawn for j = 1:configs_per_box
                equilibrate!(box_arr[i], pair_pot, pot_params, false, large_r_grid, pair_pot_faster; sweeps = 50)
                config_arr[(i - 1)*configs_per_box + j] = deepcopy(box_arr[i])
                print(".")
                flush(stdout)
            end
        end
    end
    return config_arr, box_arr
end

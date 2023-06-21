using LinearAlgebra
using Optim
using BenchmarkTools
using SpecialFunctions

min_symm(x, l) = min(x, l - x)
dist_pbc(parti1, parti2, l) = norm(min_symm.(abs.(parti1 - parti2), l)) #assumes square box
surface_area_sph(dim, r) =  2*pi^(dim/2)/gamma(dim/2)*r^(dim - 1)
volume_sph(dim, r) = π^(dim/2)/gamma(dim/2 + 1)*r^dim

min_symm(x, l) = min(x, l - x)

gaussian(x) = exp(-x^2)

lj(r, params) = params[1]/r^12 - params[2]/r^6


triangle_func(x, x0, α) = max(0, 1 - abs(x - x0)/α)/α

dropcol(M::AbstractMatrix, j) = M[:, deleteat!(collect(axes(M, 2)), j)]

function grad_descent(func, params, a = 0.1, tol = 10^-4)
    val = func(params)
    val_new = val
    grad_cfg = ForwardDiff.GradientConfig(func, params)
    grad = ForwardDiff.gradient(func, params, grad_cfg)
    dropping = false
    #find the right step size
    while !dropping
        params_new = params - grad*a
        val_new = func(params_new)
        dropping = (val_new < val)
        if !dropping
            a /= 10
        end
    end
    val = val_new
    while norm(grad) > tol
        println(val)
        println(grad)
        println(params)
        flush(stdout)
        val = val_new
        params -= grad*a
        val_new = func(params)
        grad = ForwardDiff.gradient(func, params, grad_cfg)
        dropping = (val_new < val)
        if !dropping
            a /= 2
        end
    end
    return params
end

function make_faster(bin_size, grid)
    function faster_f(r)
        inds = min.(ceil.(Int, r/bin_size), size(grid, 1))
        return grid[inds...]
    end
    return faster_f
end

function pretty_print(array)
    show(IOContext(stdout, :compact=>false), "text/plain", array)
    println()
end

```computes the structure factor of a system in a cubic box```
function struc_fac(particles, l, range, bin_size = 0.05)
    k_min = 2*π/l
    dim, n = size(particles)
    n_range = ceil(Int, range/k_min)
    raw = zeros(0, 2)
    for i in product(ntuple(i -> -n_range:n_range, dim)...)
        n_tilde = 0
        k_vec = collect(i)*k_min
        if k_vec ≠ zeros(dim) && k_vec[1] ≥ 0 && norm(k_vec) ≤ range
            for r_vec in eachcol(particles)
                n_tilde += exp(-im*k_vec⋅r_vec)
            end
            s = norm(n_tilde)^2/n
            raw = vcat(raw, [norm(k_vec) s])
        end
    end
    k_col = (k_min + bin_size/2):bin_size:(range + bin_size)
    n_bins = length(k_col)
    collector = [[] for _ = 1:n_bins]
    result = zeros(n_bins, 2)
    result[:, 1] = k_col
    for row in eachrow(raw)
        bin = max(1, ceil(Int, (row[1] - k_min)/bin_size))
        push!(collector[bin], row[2])
    end
    for i = 1:n_bins
        if collector[i] ≠ []
            result[i, 2] = mean(collector[i])
        else
            result[i, 2] = result[i - 1, 2]
        end
    end
    return result
end

```computes the magnetic structure factor of an unpolarized system in a cubic box```
function magnetic_struc_fac(particles, l, range, bin_size = 0.05)
    k_min = 2*π/l
    dim, n = size(particles)
    n_range = ceil(Int, range/k_min)
    raw = zeros(0, 2)
    for i in product(ntuple(i -> -n_range:n_range, dim)...)
        n_tilde = 0
        k_vec = collect(i)*k_min
        if k_vec ≠ zeros(dim) && k_vec[1] ≥ 0 && norm(k_vec) ≤ range
            for j = 1:n
                r_vec = particles[:, j]
                if j > n/2
                    n_tilde += exp(-im*k_vec⋅r_vec)
                else
                    n_tilde += -exp(-im*k_vec⋅r_vec)
                end
            end
            s = norm(n_tilde)^2/n
            raw = vcat(raw, [norm(k_vec) s])
        end
    end
    k_col = (k_min + bin_size/2):bin_size:(range + bin_size)
    n_bins = length(k_col)
    collector = [[] for _ = 1:n_bins]
    result = zeros(n_bins, 2)
    result[:, 1] = k_col
    for row in eachrow(raw)
        bin = max(1, ceil(Int, (row[1] - k_min)/bin_size))
        push!(collector[bin], row[2])
    end
    for i = 1:n_bins
        if collector[i] ≠ []
            result[i, 2] = mean(collector[i])
        else
            result[i, 2] = result[i - 1, 2]
        end
    end
    return result
end

```computes the structure factor between same spins of an unpolarized system in a cubic box```
function struc_fac_same(particles, l, range, bin_size = 0.05)
    s = struc_fac(particles, l, range, bin_size)
    mag_s = magnetic_struc_fac(particles, l, range, bin_size)
    s_same_vec = (s[:, 2] + mag_s[:, 2])/2
    return hcat(s[:, 1], s_same_vec)
end

```computes the structure factor between opposite spins of an unpolarized system in a cubic box```
function struc_fac_oppo(particles, l, range, bin_size = 0.05)
    s = struc_fac(particles, l, range, bin_size)
    mag_s = magnetic_struc_fac(particles, l, range, bin_size)
    s_oppo_vec = (s[:, 2] - mag_s[:, 2])/2 .+ 1
    return hcat(s[:, 1], s_oppo_vec)
end

function parse_func(filename, range)
    f = open(filename, "r")
    line1 = reshape(parse.(Float64, split(readline(f))), 1, 2)
    line2 = reshape(parse.(Float64, split(readline(f))), 1, 2)
    close(f)
    bin_size = line2[1] - line1[1]
    nlines = floor(Int, range/bin_size)
    data = zeros(nlines, 2)
    f = open(filename, "r")
    for i = 1:nlines
        line = readline(f)
        data[i, :] = reshape(parse.(Float64, split(line)), 1, 2)
    end
    close(f)
    x_vec = data[:, 1]
    f_vec = data[:, 2]
    return x_vec, f_vec
end

function pretty_print(array)
    show(IOContext(stdout, :compact=>false), "text/plain", array)
    println()
end

function bfgs_2(f, params, delta, stepsize, other_args_for_f...)
    n_params = length(params)
    grad = zeros(n_params)
    grad_new = gradient(f, params, delta, other_args_for_f...) 
    h = I(n_params)
    gradnorm = norm(grad_new)
    round = 0
    params_old = deepcopy(params)
    params_new = zeros(n_params)
    while gradnorm > 0.001 && round <= 100
        round += 1
        grad = deepcopy(grad_new)
        direction = h*grad
        dx = zeros(n_params)
        line_search_count = 1
        e_old = f(params_old, other_args_for_f...)
        e_new = e_old
        alphaK = 0.0
        while line_search_count < 100
            params_new = params_old - stepsize*direction
            e_new = f(params_new, other_args_for_f...)
            if e_new < e_old 
                alphaK += stepsize
                e_old, params_old = e_new, params_new
                line_search_count += 1
            else
                params_new = params_old
                if line_search_count > 1
                    break
                elseif stepsize > 1E-8
                    stepsize /= 1.5
                else
                    h = 0.1I(n_params)
                    break
                end
            end
        end
        dx = -alphaK*direction
        if norm(dx) < 0.05
            h = I(n_params)
            continue
        end
        if line_search_count > 10
            stepsize *= 2
        end
        grad_new = gradient(f, params_new, delta, other_args_for_f...)
        gradnorm = norm(grad_new)
        if gradnorm <= 0.001
            break
        end
        y = grad_new - grad
        dx_y = dx⋅y
        y_h_y = y⋅(h*y)
        dxT_dx = dx*dx'
        second_term = (dx_y + y_h_y)/(dx_y*dx_y)*dxT_dx
        h_y_dxT = (h*y)*dx'
        third_term = (h_y_dxT + h_y_dxT')/dx_y
        h += second_term - third_term
    end
    return (params_new, f(params_new, other_args_for_f...))
end

function gradient(f, params, delta, other_args_for_f...) 
    n_params = length(params)
    result = zeros(n_params)
    up = zeros(n_params)
    down = zeros(n_params)
    for i = 1:n_params
        for j = 1:n_params
            if j != i 
                up[j] = params[j]
                down[j] = params[j]
            else 
                up[j] = params[j] + delta/2
                down[j] = params[j] - delta/2
            end
        end
        result[i] = (f(up, other_args_for_f...) -  f(down, other_args_for_f...))/delta;
    end
    return result
end

function test_opt()
    f(x, y) = x[1]^2*y
    @time bfgs_2(f, [100], 0.01, 0.2, 2)
    return bfgs_2(f, [100], 0.01, 0.2, 2)
end

function norm_sq(vec)
    return vec⋅vec
end

function boltzmann_accept(oldE::Real, newE::Real, temperature::Real)::Bool
	if newE < oldE 
		return true
	end
    if isinf(newE) 
        if isinf(oldE)
            return true
        end
		return false
	end
	rand_num = rand()
    prob = exp(-(newE - oldE)/temperature)
    if rand_num <= prob
        return true
    end 
	return false
end

function q6_2D(particles, l) 
    n = size(particles, 2)
    sum = 0
    for i = 1:n
        p = particles[:, i];
        for j = 1:(i - 1)
            q = particles[:, j]
            pq_vec = l/2 .- abs.(mod.(p - q, l) .- l/2)
            theta = atan(pq_vec[2], pqvec[1])
            sum += (cos(6*theta) + im*sin(6*theta))
        end
    end
    return norm(sum)/(n*(n - 1)/2)
end

function ϕ2ρ3D(ϕ, diam = 1)
    return 6*ϕ/(π*diam^3)
end

function ϕ2ρ2D(ϕ, diam = 1)
    return 4*ϕ/(π*diam^2)
end

function ϕ2ρ1D(ϕ, diam = 1)
    return ϕ/diam
end

```generates random point in ball centered at \$center with radius \$r in dimension \$dim```
function rand_pt_in_sphere(r, dim, center)
    pt = 2*ones(dim)*r #just a point that's initially out of the ball
    while norm(pt) > r
        pt = rand(dim)*2*r .- r
    end
    return pt + center
end

```deletes a column of a matrix```
dropcol(M::AbstractMatrix, j) = M[:, deleteat!(collect(axes(M, 2)), j)]

```volume of a \$d-dimensional sphere of radius \$r```
v1(r, d) = π^(d/2)/gamma(d/2 + 1)*r^d

function fourier(f, r_vec, dim, k_vec)
    function fourier_singlek(f, r_vec, dim::Int, k)
        result = 0
        bin_size = r_vec[2] - r_vec[1]
        for i in eachindex(r_vec)
            r = r_vec[i]
            result += r^(dim - 1)*f[i]*besselj(dim/2 - 1, k*r)/(k*r)^(dim/2 - 1)
        end
        return result*bin_size*(2*π)^(dim/2)
    end
    return [fourier_singlek(f, r_vec, dim, k) for k in k_vec]
end

function inv_fourier(f, k_vec, dim, r_vec)
    function inv_fourier_singler(f, k_vec, dim::Int, r)
        result = 0
        bin_size = k_vec[2] - k_vec[1]
        for i in eachindex(k_vec)
            k = k_vec[i]
            result += k^(dim - 1)*f[i]*besselj(dim/2 - 1, k*r)/(k*r)^(dim/2 - 1)
        end
        return result*bin_size/(2*π)^(dim/2)
    end
    return [inv_fourier_singler(f, k_vec, dim, r) for r in r_vec]
end


#test_opt()

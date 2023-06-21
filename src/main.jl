#find effective classical interactions for 2D gaussian pair statistics
include("../optimize_single_compo.jl") #use the path of whereever the file optimize_single_compo.jl is

pot(r, params) = -params[1]*exp(-(r/params[2])^2)*log(r) + (-2*log(r))*(- exp(-(r/params[3])^2)) #short ranged part

params_vec = [2.224938118380188, 1.1508032809268853, 1.0695396447428824]

dim = 2
n = 1600
ρ = 1
l = (n/ρ)^(1/dim)
bin_size = 0.05
r_vec = bin_size/2:bin_size:l/2 + bin_size

function large_r_ewald(r, ewald_range, l) #this takes care of the long-ranged part of the potential, -2*log(r)
    one_body = 2*ewald_range^2*(-6 + π + log(4) + 4*log(ewald_range*l))
    return one_body + sum([-2*log(norm(r + collect(ind)*l)) for ind in product(ntuple(i -> -ewald_range:ewald_range - 1, 2)...)])
end

large_r_grid = ThreadsX.collect(large_r_ewald(collect(r), 13, l) for r in product(ntuple(i -> r_vec, dim)...))

optimize(params_vec; 
    pot = pot,
    large_r_grid = large_r_grid,
    dim = dim,
    n = n,
    ρ = ρ,
    bin_size = bin_size,
    r_range = l/2,
    k_range = 10,
    targ_g2 = r -> 1 - exp(-π*r^2),
    targ_s = k -> 1 - exp(-k^2/(4*π)),
    g2_weight_range = 2,
    s_weight_range = 4,
    n_boxes = 15,
    configs_per_box = 5,
    Ψ_tol = 0.001,
    show_pb = false
)


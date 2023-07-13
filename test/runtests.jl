using Spec2Struc, SpecialFunctions
using Test

@testset "Spec2Struc.jl" begin
    kF = 2*√π
    function targ_s(k)
        if k ≤ 2kF
            return 2*(asin(k/(2kF)) + (k/(2kF))*√(1 - (k/(2kF))^2))/π
        else
            return 1
        end
    end
    targ_g2(r) = 1 - 4*besselj(1, kF*r)^2/(kF*r)^2
    pot(r, params) = -params[1]*exp(-(r/params[2])^2)*log(r) + √π/(2*r)*(1 - exp(-(r/params[3])^2))
    params_vec = [2.0, 1.0, 1.0] #[2.099144746573296, 0.666655413783426, 0.4739706702894683]
    @test Spec2Struc.optim_parametrized_pot(params_vec;
        pot = pot,
        dim = 2,
        n = 600,
        ρ = 1,
        bin_size = 0.05,
        r_range = sqrt(600)/2,
        k_range = 10,
        targ_g2 = targ_g2,
        targ_s = targ_s,
        g2_weight_range = 2,
        s_weight_range = 4,
        n_threads = 15,
        configs_per_box = 5,
        Ψ_tol = 0.005,
        show_pb = true,
        test = true
       )[1]
end

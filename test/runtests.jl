using InverseStatMech, SpecialFunctions
using Test

@testset "ornstein zernike" begin
    kF = 2*√π
    targ_g2(r) = 1 - 4*besselj(1, kF*r)^2/(kF*r)^2
    function targ_s(k)
        if k ≤ 2kF
            return 2*(asin(k/(2kF)) + (k/(2kF))*√(1 - (k/(2kF))^2))/π
        else
            return 1
        end
    end
    @test InverseStatMech.ornstein_zernike_v(2, 1, targ_g2, targ_s, "PMF")(0.5) > 0
    @test InverseStatMech.ornstein_zernike_v(2, 1, targ_g2, targ_s, "MFA")(0.5) > 0
    @test InverseStatMech.ornstein_zernike_v(2, 1, targ_g2, targ_s, "HNC")(0.5) > 0
    @test InverseStatMech.ornstein_zernike_v(2, 1, targ_g2, targ_s, "PY")(0.5) > 0
end

@testset "reverse_monte_carlo" begin
    kF = 2*√π
    targ_g2(r) = 1 - 4*besselj(1, kF*r)^2/(kF*r)^2
    @test InverseStatMech.reverse_mc(2, 500, 1, targ_g2; test = true)
end


@testset "torquato-wang algorithm" begin
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
    @test InverseStatMech.optim_parametrized_pot(params_vec, pot, 2, 1.0, targ_g2, targ_s; Ψ_tol = 0.01, test = true)
end

@testset "iterative boltzmann inversion" begin
    @test InverseStatMech.iterative_boltzmann(r -> 0, 2, 1, r -> 1 - exp(-π*r^2); Ψ_tol = 0.4, test = true)
end

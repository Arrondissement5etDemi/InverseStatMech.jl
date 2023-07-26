using Interpolations

"""
    function ornstein_zernike_v(dim::Int, ρ::Real, g2::Function, s::Function, closure::String, r_vec::AbstractVector = 0.025:0.05:10,
    k_vec::AbstractVector = 0.025:0.05:20) :: Interpolations.GriddedInterpolation

Compute the Ornstein-Zernike potential (``βv(r)``) using the Ornstein-Zernike equation.

## Arguments

- `dim::Int`: The dimensionality of the system.
- `ρ::Real`: The number density (particles per unit volume) of the system.
- `g2::Function`: The pair correlation function ``g_2(r)`` as a function of ``r``.
- `s::Function`: The structure factor ``S(k)`` as a function of ``k``.
- `closure::String`: The closure relation to be used in the Ornstein-Zernike equation. Possible values are:
    - "PMF": Potential of mean force, basically ``-\\ln(g_2(r))``.
    - "MFA":Mean-Field Approximation (MFA).
    - "HNC": Hypernetted Chain closure.
    - "PY": Percus-Yevick closure.

## Optional Arguments

- `r_vec::AbstractVector`: A vector of r values at which the g_2(r) function is valid. Default is 0.025:0.05:10.
- `k_vec::AbstractVector`: A vector of k values at which the S(k) function is valid. Default is 0.025:0.05:20.

## Returns

- An instance of `Interpolations.GriddedInterpolation` representing the OZ potential ``βv(r)``, which can be used as a regular function.

Note: The `g2` and `s` functions should be provided as valid functions that return the pair correlation function and structure factor, respectively, as a function of r and k.

## Example:
    function pair_correlation_function(r)
        # Define the pair correlation function here

        # ...
    end
    function structure_factor(k)
        # Define the structure factor here

        # ...
    end
    closure_type = "PY"
    oz_v = ornstein_zernike_v(3, 0.5, pair_correlation_function, structure_factor, closure_type)
"""
function ornstein_zernike_v(dim, ρ, g2, s, closure, r_vec = 0.025:0.05:10, k_vec = 0.025:0.05:20)
    good_r_vec = zeros(0)
    ln_g2_vec = zeros(0)
    for r in r_vec
        if g2(r) > 0
            push!(good_r_vec, r)
            push!(ln_g2_vec, log(g2(r)))
        end
    end
    g2_vec = g2.(good_r_vec)
    if closure == "PMF"
        oz_data = -ln_g2_vec
    else
        c_tilde(k) = (s(k) - 1)/(ρ*s(k))
        c_tilde_vec = c_tilde.(k_vec)
        c_vec = inv_fourier(c_tilde_vec, k_vec, dim, good_r_vec)
        if closure == "MFA"
            oz_data = -c_vec
        elseif closure == "HNC"
            oz_data = g2_vec .- 1 - c_vec - ln_g2_vec
        elseif closure == "PY"
            oz_data = log.([1 - c_vec[i]/g2_vec[i] for i in eachindex(g2_vec)])
        end
    end
    return Interpolations.linear_interpolation(good_r_vec, oz_data, extrapolation_bc = Line())
end

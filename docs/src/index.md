# InverseStatMech.jl

*Efficient inverse statistical mechanical algorithms to generate effective potentials or configurations from pair correlation functions or structure factors.*

## Package Features

- Ornstein-Zernike equations to find approximate potentials corresponding to given target ``g_2(r)`` and ``S(k)``, including the following common closures:
    - Potential of mean force
    - Mean-field approximation
    - Hypernetted chain
    - Percus-Yevick
- Torquato-Wang algorithm (Torquato and Wang, PRE 2020) to find optimized parametrized potentials given target ``g_2(r)`` and ``S(k)``
- Iterative Boltzmann Inversion (Soper, Chem. Phys. 1996) to find optimimzed binned potentials given target ``g_2(r)``
- Reverse Monte-Carlo algorithm (McGreevy, J. Phys. Cond. Matter 2001) to find configuration that match target ``g_2(r)``
- More to come!

## Function Documentation

```@docs
ornstein_zernike_v
optim_parametrized_pot
iterative_boltzmann
reverse_mc
```

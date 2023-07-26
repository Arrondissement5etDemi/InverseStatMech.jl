# InverseStatMech.jl

*Efficient inverse statistical mechanical algorithms to generate effective potentials or configurations from pair correlation functions or structure factors.*

## Package Features
- Solves Ornstein-Zernike equations to find approximate potentials for given target ``g_2(r)`` and ``S(k)``, supports the following closures:
    - Potential of mean force
    - Mean field approximation
    - Hypernetted chain
    - Percus-Yevick
- Torquato-Wang algorithm (Torquato and Wang, PRE 2020) to find optimized parametrized potentials given target ``g_2(r)`` and ``S(k)``
- Iterative Boltzmann Inversion (Soper, Chem. Phys. 1996) to find optimimzed binned potentials given target ``g_2(r)``
- Reverse Monte-Carlo algorithm (McGreevy, J. Phys. Cond. Matter 2001) to find configuration that match target ``g_2(r)``
- More to come!

# Documentation
Please click [here](https://arrondissement5etdemi.github.io/InverseStatMech.jl/dev/).


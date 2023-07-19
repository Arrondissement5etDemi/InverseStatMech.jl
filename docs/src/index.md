# InverseStatMech.jl

*Efficient inverse statistical mechanical algorithms to generate effective potentials or configurations from pair correlation functions or structure factors.*

## Package Features

- Torquato-Wang algorithm (Torquato and Wang, PRE 2020) to find optimized parametrized potentials given target ``g_2(r)`` and ``S(k)``
- Iterative Boltzmann Inversion (Soper, Chem. Phys. 1996) to find optimimzed binned potentials given target ``g_2(r)``
- Reverse Monte-Carlo algorithm (McGreevy, J. Phys. Cond. Matter 2001) to find configuration that match target ``g_2(r)``
- More to come!

## Function Documentation

```@docs
optim_parametrized_pot
iterative_boltzmann
reverse_mc
```

import Pkg; Pkg.add("Documenter")
push!(LOAD_PATH,"../src/")
using InverseStatMech
using Documenter

makedocs(
         sitename = "InverseStatMech.jl",
         modules  = [InverseStatMech],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/Arrondissement5etDemi/InverseStatMech.jl",
   )

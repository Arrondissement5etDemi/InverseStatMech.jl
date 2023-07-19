import Pkg; Pkg.add("Documenter")
push!(LOAD_PATH,"../src/")
using InvStatMech
using Documenter

makedocs(
         sitename = "InverseStatMech.jl",
         modules  = [InvStatMech],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/Arrondissement5etDemi/InverseStatMech.jl",
   )

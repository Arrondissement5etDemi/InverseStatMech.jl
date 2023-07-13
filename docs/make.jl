import Pkg; Pkg.add("Documenter")
push!(LOAD_PATH,"../src/")
using Spec2Struc
using Documenter

makedocs(
         sitename = "Spec2Struc.jl",
         modules  = [Spec2Struc],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/Arrondissement5etDemi/Spec2Struc.jl"
)

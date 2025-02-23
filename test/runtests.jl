

using Pkg
current_path = @__DIR__
Pkg.activate(current_path * "/../.");
using HarmonicBalance
using Test

files = [
    "powers.jl",
    "harmonics.jl", 
    "fourier.jl",
    "parametron.jl",
    "load.jl"
    ]

for file in files
    include(file)
    printstyled(file * ":    OK\n"; color = :green)
end

printstyled("\nALL TESTS PASSED!\n"; color = :green)

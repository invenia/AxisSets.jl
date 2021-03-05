using AxisSets
using Documenter
using Test

using AxisSets: Pattern

@testset "AxisSets.jl" begin
    include("patterns.jl")
    doctest(AxisSets)
end

using AxisKeys
using AxisSets
using Dates
using Documenter
using Test

using AxisSets: Pattern, flatten

@testset "AxisSets.jl" begin
    include("flatten.jl")
    include("patterns.jl")
    doctest(AxisSets)
end

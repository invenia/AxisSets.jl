using AxisKeys
using AxisSets
using Dates
using Documenter
using OrderedCollections
using Test

using AxisSets: Dataset, Pattern, constraintmap, flatten, validate

@testset "AxisSets.jl" begin
    include("flatten.jl")
    include("patterns.jl")
    include("dataset.jl")
    doctest(AxisSets)
end

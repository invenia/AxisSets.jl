using AxisKeys
using AxisSets
using Dates
using Documenter
using Missings
using OrderedCollections
using ReadOnlyArrays
using Test

using AxisSets: Pattern, constraintmap, dimpaths, flatten, validate

@testset "AxisSets.jl" begin
    include("flatten.jl")
    include("patterns.jl")
    include("dataset.jl")
    include("indexing.jl")

    # Repl output changes across julia versions and architectures
    Sys.WORD_SIZE == 64 && VERSION >= v"1.5" && doctest(AxisSets)
end

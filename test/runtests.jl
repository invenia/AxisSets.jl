using AxisKeys
using AxisSets
using Dates
using Documenter
using Impute
using Missings
using OrderedCollections
using ReadOnlyArrays
using Statistics
using Test
using TimeZones

using AxisSets: Pattern, constraintmap, dimpaths, flatten, validate
using Impute: ThresholdError

@testset "AxisSets.jl" begin
    include("flatten.jl")
    include("patterns.jl")
    include("dataset.jl")
    include("indexing.jl")
    include("functions.jl")
    include("impute.jl")

    # Repl output changes across julia versions and architectures
    Sys.WORD_SIZE == 64 && VERSION >= v"1.5" && doctest(AxisSets)
end

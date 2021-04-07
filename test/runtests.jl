using AxisKeys
using AxisSets
using Dates
using Documenter
using FeatureTransforms
using Impute
using Missings
using OrderedCollections
using ReadOnlyArrays
using Statistics
using Test
using TimeZones

using AxisSets:
    Pattern,
    KeyAlignmentError,
    KeyedDataset,
    constraintmap,
    dimpaths,
    flatten,
    validate
using Impute: ThresholdError

@testset "AxisSets.jl" begin
    include("flatten.jl")
    include("patterns.jl")
    include("dataset.jl")
    include("indexing.jl")
    include("functions.jl")
    include("impute.jl")
    include("featuretransforms.jl")

    # The doctests fail on x86, so only run them on 64-bit hardware & Julia 1.6
    Sys.WORD_SIZE == 64 && v"1.6" <= VERSION < v"1.7" && doctest(AxisSets)
end

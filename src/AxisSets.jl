module AxisSets

using AutoHashEquals
using AxisKeys
using FeatureTransforms
using Impute
using NamedDims
using OrderedCollections
using ReadOnlyArrays

using Impute: DeclareMissings, Filter, Imputor, Validator

export KeyedDataset

# There's a few places calling `only` is convenient, even for older Julia releases
if VERSION < v"1.4"
    function _only(x)
        if isempty(x)
            throw(ArgumentError("Collection is empty, must contain exactly 1 element"))
        elseif length(x) > 1
            throw(ArgumentError("Collection has multiple elements, must contain exactly 1 element"))
        else
            first(x)
        end
    end
else
    _only(x) = only(x)
end

include("flatten.jl")
include("patterns.jl")
include("dataset.jl")
include("indexing.jl")
include("functions.jl")
include("impute.jl")
include("featuretransforms.jl")
include("utils.jl")

end

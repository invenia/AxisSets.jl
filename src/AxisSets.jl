module AxisSets

using AxisKeys
using Impute
using IterTools
using NamedDims
using OrderedCollections
using ReadOnlyArrays
using Tables

using Impute:
    Filter,
    Imputor,
    Validator

# Short hand type for complicated union of nested Keyed or NamedDims arrays
XArray{L, T, N} = Union{NamedDimsArray{L,T,N,<:KeyedArray}, KeyedArray{T,N,<:NamedDimsArray}}

include("flatten.jl")
include("datasets.jl")

end

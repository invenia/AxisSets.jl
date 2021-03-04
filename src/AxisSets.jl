module AxisSets

using AxisKeys
using NamedDims

# We'll try and use a relatively distinctive delimeters that are less likely to be confused
# with common key characters
# https://github.com/mcabbott/NamedPlus.jl/blob/master/src/reshape.jl#L165
const DEFAULT_FLATTEN_DELIM = :⁻    # Tuple flattening
const DEFAULT_PROD_DELIM = :ᵡ       # ndim array flatten / reshape using product of keys

# Short hand type for complicated union of nested Keyed or NamedDims arrays
const XArray{L, T, N} = Union{NamedDimsArray{L,T,N,<:KeyedArray}, KeyedArray{T,N,<:NamedDimsArray}}

include("flatten.jl")
include("patterns.jl")

end

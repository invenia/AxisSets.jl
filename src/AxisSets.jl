module AxisSets

using AxisKeys
using NamedDims

"""
    DEFAULT_FLATTEN_DELIM

`:⁻` (or `\^-`)

Separates the parent symbols of a nested `NamedTuple` that has been flattened.
A less common symbol was used to avoid collisions with `:_` in the parent symbols.
"""
const DEFAULT_FLATTEN_DELIM = :⁻

"""
    DEFAULT_PROD_DELIM

`:ᵡ` (or `\^x`)

Separates the parent symbols from an n-dimensional array that was flattened / reshaped.
A less common symbol was used to avoid collisions with `:_` in the parent symbols.
"""
const DEFAULT_PROD_DELIM = :ᵡ

# Short hand type for complicated union of nested Keyed or NamedDims arrays
const XArray{L, T, N} = Union{NamedDimsArray{L,T,N,<:KeyedArray}, KeyedArray{T,N,<:NamedDimsArray}}

include("flatten.jl")
include("patterns.jl")

end

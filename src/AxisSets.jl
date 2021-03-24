module AxisSets

using AutoHashEquals
using AxisKeys
using Impute
using NamedDims
using OrderedCollections
using ReadOnlyArrays

using Impute: DeclareMissings, Filter, Imputor, Validator

export KeyedDataset

"""
    DEFAULT_FLATTEN_DELIM

`:⁻` (or `\\^-`)

Separates the parent symbols of a nested `NamedTuple` that has been flattened.
A less common symbol was used to avoid collisions with `:_` in the parent symbols.

# Example
```jldoctest
julia> using AxisSets: flatten

julia> data = (
           val1 = (a1 = 1, a2 = 2),
           val2 = (b1 = 11, b2 = 22),
           val3 = [111, 222],
           val4 = 4.3,
       );

julia> flatten(data)
(val1⁻a1 = 1, val1⁻a2 = 2, val2⁻b1 = 11, val2⁻b2 = 22, val3 = [111, 222], val4 = 4.3)
```
"""
const DEFAULT_FLATTEN_DELIM = :⁻

"""
    DEFAULT_PROD_DELIM

`:ᵡ` (or `\\^x`)

Separates the parent symbols from an n-dimensional array that was flattened / reshaped.
A less common symbol was used to avoid collisions with `:_` in the parent symbols.

# Example
```jldoctest
julia> using AxisKeys, Dates; using AxisSets: flatten

julia> A = KeyedArray(
           reshape(1:24, (4, 3, 2));
           time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
           obj=[:a, :b, :c],
           loc=[1, 2],
       );

julia> axiskeys(flatten(A, (:obj, :loc)), :objᵡloc)
6-element Array{Symbol,1}:
 :aᵡ1
 :bᵡ1
 :cᵡ1
 :aᵡ2
 :bᵡ2
 :cᵡ2
```
"""
const DEFAULT_PROD_DELIM = :ᵡ

# Short hand type for complicated union of nested Keyed or NamedDims arrays
const XArray{L, T, N} = Union{NamedDimsArray{L,T,N,<:KeyedArray}, KeyedArray{T,N,<:NamedDimsArray}}

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

end

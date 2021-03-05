module AxisSets

using AxisKeys
using NamedDims

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
           loc=1:3,
           obj=[:a, :b],
       );

julia> flatten(A, (:loc, :obj))
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   time ∈ 4-element StepRange{Dates.DateTime,...}
→   locᵡobj ∈ 6-element Vector{Symbol}
And data, 4×6 reshape(::UnitRange{Int64}, 4, 6) with eltype Int64:
                                     …   Symbol("2ᵡb")    Symbol("3ᵡb")
   DateTime("2021-01-01T11:00:00")      17               21
   DateTime("2021-01-01T12:00:00")      18               22
   DateTime("2021-01-01T13:00:00")      19               23
   DateTime("2021-01-01T14:00:00")      20               24
```
"""
const DEFAULT_PROD_DELIM = :ᵡ

# Short hand type for complicated union of nested Keyed or NamedDims arrays
const XArray{L, T, N} = Union{NamedDimsArray{L,T,N,<:KeyedArray}, KeyedArray{T,N,<:NamedDimsArray}}

include("flatten.jl")
include("patterns.jl")

end

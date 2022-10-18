
isassociative(x::NamedTuple) = true
isassociative(x::AbstractDict) = true
isassociative(x::Pair) = true
isassociative(x::Iterators.Pairs) = true
isassociative(x::Vector{<:Pair}) = true
isassociative(x) = false

"""
    flatten(collection, [delim])

Flatten a collection of nested associative types into a flat collection of pairs.

# Example
```jldoctest
julia> using AxisSets: flatten

julia> data = (
           val1 = (a1 = 1, a2 = 2),
           val2 = (b1 = 11, b2 = 22),
           val3 = [111, 222],
           val4 = 4.3,
       );

julia> flatten(data, :_)
(val1_a1 = 1, val1_a2 = 2, val2_b1 = 11, val2_b2 = 22, val3 = [111, 222], val4 = 4.3)
```

    flatten(A, dims, [delim])

Flatten a `KeyedArray` along the specified consecutive dimensions.
The `dims` argument can either be a `Tuple` of symbols or a `Pair{Tuple, Symbol}` if
you'd like to specify the desired flattened dimension name.

# Example
```jldoctest
julia> using AxisKeys, Dates, NamedDims; using AxisSets: flatten

julia> A = KeyedArray(
           reshape(1:24, (4, 3, 2));
           time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
           obj=[:a, :b, :c],
           loc=[1, 2],
       );

julia> dimnames(flatten(A, (:obj, :loc), :_))
(:time, :obj_loc)
```
"""
function flatten end

# Single arg version defaults to `nothing` delim
# All other methods dispatch on the two arg form to be explicit
flatten(x) = flatten(x, nothing)

# For simplicity we just convert singleton `Pair` arguments to `Pair[x]`, even though it's
# less efficient.
flatten(x::Pair, delim::Nothing) = flatten([x], delim)

# Primary flatten algorithm. Other methods should call this and adjust keys as necessary.
function flatten(x::Union{Vector{<:Pair}, Iterators.Pairs}, delim::Nothing)
    result = Channel{Pair}() do chnl
        for (k, v) in x
            if isassociative(v)
                children = flatten(v)
                iter = isa(children, NamedTuple) ? pairs(children) : children
                for (_k, _v) in iter
                    new_key = isa(_k, Tuple) ? (k, _k...) : (k, _k)
                    put!(chnl, new_key => _v)
                end
            else
                put!(chnl, (k isa Tuple ? k : (k,)) => v)
            end
        end
    end

    return collect(result)
end

# NOTE: NamedTuples only support symbol names, so we fallback to a pairs iterator if a delimiter isn't provided
flatten(x::NamedTuple, delim::Nothing) = flatten(pairs(x), delim)
function flatten(x::NamedTuple, delim::Symbol)::NamedTuple
    kwargs = map(flatten(pairs(x))) do (k, v)
        _k = isa(k, Tuple) ? join(k, delim) : k
        return Symbol(_k) => v
    end
    return (; kwargs...)
end

# Since the key type could be changed as part of flattening we can only ensure the that
# result is a `Dict` (vs the same input dict).
flatten(x::AbstractDict, delim) = Dict(flatten(collect(x), delim)...)

# Utility flatten to handle joining tuples of symbols or strings into a single string/symbol
function flatten(x::Vector{<:Pair}, delim::AbstractString)
    return [join(k, delim) => v for (k, v) in flatten(x)]
end

function flatten(x::Vector{<:Pair}, delim::Symbol)
    return [Symbol(join(k, delim)) => v for (k, v) in flatten(x)]
end

function flatten(A::KeyedArray, dims::Tuple, delim::Symbol)
    new_name = Symbol(join(dims, delim))
    flatten(A, dims => new_name, delim)
end

function flatten(A::KeyedArray, dims::Pair{<:Tuple, Symbol}, delim=nothing)
    # Lookup our unnamed dimensions to flatten
    # We sort the result to ensure that the dimensions to flatten are consecutive
    fd = sort!(collect(NamedDims.dim(A, first(dims))))

    # The max difference between the dimensions to flatten should be 1, otherwise we have
    # non-consecutive dimensions and should throw an error
    d = diff(fd)
    isempty(d) && throw(ArgumentError("Flatten dimensions must be consecutive"))
    maximum(d) == 1 || throw(ArgumentError("Flatten dimensions must be consecutive"))

    # The offset is equal to the number of dimensions that are being dropped
    # (ie: ndims(origin) - ndims(flattened))
    offset = length(first(dims)) - 1

    # Our desired number of dimensions after flattening
    n = ndims(A) - offset

    # Extract the original dimension sizes, names and keys for easy reference
    _size = collect(size(A))
    _names = collect(NamedDims.dimnames(A))
    _keys = collect(axiskeys(A))

    # Generic function for constructing new dimension sizes, names and keys, since the
    # pattern is the largely the same. As we iterate from 1 to n we check the relative
    # index dimension `d` relative to our dimensions to flatten.
    #
    # - d < first(fd): Return original values since nothing has changed
    # - d == first(d): Apply our flatten values
    # - d > first(d): Return original values using an index offset
    function newdims(src::Vector, val)
        return ntuple(n) do d
            if d < first(fd)
                src[d]
            elseif d == first(fd)
                val
            else
                src[d + offset]
            end
        end
    end

    # Flattened size values is just the flattened dimension sizes multiplied together
    sz = newdims(_size, *(_size[fd]...))
    # Flattened name is just the second value of the input dims pair.
    nm = newdims(_names, last(dims))
    # Flattened key is the product of the keys to be flattened, either as a tuple or symbol.
    keys = newdims(
        _keys,
        [
            delim === nothing ? t : Symbol(join(t, delim))
            for t in Iterators.product(_keys[fd]...)
        ][:]
    )

    # Finally construct our new `KeyedArray` by reshaping the parent array and providing
    # our new axis names/keys. We call parent twice to avoid calling `reshape` on the
    # `NamedDimsArray`
    return KeyedArray(reshape(parent(parent(A)), sz); NamedTuple{nm}(keys)...)
end

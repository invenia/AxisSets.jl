
isassociative(x::NamedTuple) = true
isassociative(x::AbstractDict) = true
isassociative(x::Iterators.Pairs) = true
isassociative(x::Vector{<:Pair}) = true
isassociative(x) = false

# We'll try and use a relatively distinctive delimeter that's less likely to be confused
# with common key characters
# https://github.com/mcabbott/NamedPlus.jl/blob/master/src/reshape.jl#L165
const DEFAULT_DELIM = :ᵡ

# NOTE: NamedTuples only support symbol names, so we use a delimiter to combine them.
function flatten(x::NamedTuple; delim=DEFAULT_DELIM)::NamedTuple
    kwargs = map(flatten(pairs(x))) do (k, v)
        _k = isa(k, Tuple) ? join(k, delim) : k
        return Symbol(_k) => v
    end
    return (; kwargs...)
end

# Since the key type could be changed as part of flattening we can only ensure the that
# result is a `Dict` (vs the same input dict).
function flatten(x::AbstractDict; delim=nothing)
    return Dict(flatten(collect(x); delim=delim)...)
end

function flatten(x::Union{Vector{<:Pair}, Iterators.Pairs}; delim=nothing)
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
                put!(chnl, (k,) => v)
            end
        end
    end

    return collect(result)
end

# Utility flatten to handle joining tuples of symbols or strings into a single string/symbol
function flatten(x::Vector{<:Pair{AbstractString}}, delim::AbstractString)
    return [join(k, delim) => v for (k, v) in flatten(x)]
end

function flatten(x::Vector{<:Pair{Symbol}}, delim::Symbol)
    return [Symbol(join(k, delim)) => v for (k, v) in flatten(x)]
end

# Fallback if delim is nothing is just the single argument form
flatten(x::Vector{<:Pair}, delim::Nothing) = flatten(x)

function flatten(A::XArray, dims::Tuple, delim=DEFAULT_DELIM)
    new_name = Symbol(join(dims, delim))
    flatten(A, dims => new_name, delim)
end

function flatten(A::XArray, dims::Pair{<:Tuple, Symbol}, delim=nothing)
    fd = sort!(collect(NamedDims.dim(A, first(dims))))
    maximum(diff(fd)) == 1 || throw(ArgumentError("Flatten dimensions must be consecutive"))

    offset = length(first(dims)) - 1
    n = ndims(A) - offset
    _size = collect(size(A))
    _names = collect(NamedDims.dimnames(A))
    _keys = collect(axiskeys(A))

    sz = ntuple(n) do d
        if d < first(fd)
            _size[d]
        elseif d == first(fd)
            *(_size[fd]...)
        else
            _size[d + offset]
        end
    end

    nm = ntuple(n) do d
        if d < first(fd)
            _names[d]
        elseif d == first(fd)
            last(dims)
        else
            _names[d + offset]
        end
    end

    keys = ntuple(n) do d
        if d < first(fd)
            _keys[d]
        elseif d == first(fd)
            tuples = Iterators.product(_keys[fd]...)
            [delim === nothing ? t : Symbol(join(t, delim)) for t in tuples][:]
        else
            _keys[d + offset]
        end
    end

    return KeyedArray(reshape(parent(parent(A)), sz); NamedTuple{nm}(keys)...)
end


isassociative(x::NamedTuple) = true
isassociative(x::AbstractDict) = true
isassociative(x::Iterators.Pairs) = true
isassociative(x::Vector{<:Pair}) = true
isassociative(x) = false

# NOTE: NamedTuples only support symbol names, so we use a delimiter to combine them.
function flatten(x::NamedTuple; delim=:_)::NamedTuple
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

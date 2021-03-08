"""
    Dataset{T}

A `Dataset` describes an associative collection of component `KeyedArray`s with constraints
on their shared dimensions.

# Fields
- `constraints::OrderedSet{Pattern}` - Constraint [`Pattern`](@ref)s on shared dimensions.
- `data::LittleDict{Tuple{Vararg{Symbol}}, T}` - Flattened key paths as tuples of symbols
  to each component array of type `T`.
"""
struct Dataset{T<:XArray}
    # Our constraints are a collection of pseudo path tuples typically with 1 or
    # more `:_` wildcard components
    constraints::OrderedSet{Pattern}
    # Data lookup can be by any type, but typically it'll either be symbol or tuple.
    data::LittleDict{Tuple{Vararg{Symbol}}, T}
end

function Dataset(pairs::Pair{<:Tuple}...; constraints=Pattern[])
    data = LittleDict{Tuple{Vararg{Symbol}}, XArray}(pairs...)

    # If no constraints have been specified then we default to (:__, dimname)
    constraint_set = if isempty(constraints)
        OrderedSet{Pattern}(
            Pattern(:__, d) for d in Iterators.flatten(dimnames.(values(data)))
        )
    else
        OrderedSet{Pattern}(constraints)
    end

    result = Dataset(constraint_set, data)
    validate(result)
    return result
end

# Taking pairs is the most general constructor as it doesn't make assumptions about the
# data key type.
function Dataset(pairs::Pair{Symbol}...; constraints=Pattern[])
    data = (
        Tuple(Symbol.(split(string(k), string(DEFAULT_FLATTEN_DELIM)))) => v
        for (k, v) in pairs
    )

    return Dataset(data...; constraints=constraints)
end

# Utility kwargs constructor.
Dataset(; constraints=Pattern[], kwargs...) = Dataset(kwargs...; constraints=constraints)

function Base.show(io::IO, ds::Dataset{T}) where T
    n = length(ds.data)
    print(io, "Dataset{$T} with $n entries:")
    for c in ds.constraints
        print(io, "\n  ", c)
    end
    for (k, v) in ds.data
        printstyled(io, "\n  ", k, " => "; color=:cyan)
        printstyled(io, replace(sprint(Base.summary, v), "\n" => "\n    "); color=:cyan)
        print(io, "\n    ", replace(sprint(Base.print_array, v), "\n" => "\n    "))
    end
end

#################
# Dict iterators
#################
Base.keys(ds::Dataset) = keys(ds.data)
Base.values(ds::Dataset) = values(ds.data)
Base.pairs(ds::Dataset) = pairs(ds.data)


"""
    dimpaths(ds, [pattern]) -> Vector{<:Tuple{Vararg{Symbol}}}

Return a list of all dimension paths in the [`Dataset`](@ref).
Optionally, you can filter the results using a [`Pattern`](@ref).
"""
dimpaths(ds::Dataset, pattern::Pattern) = filter(in(pattern), dimpaths(ds))
function dimpaths(ds::Dataset)
    paths = Iterators.flatten(((k..., d) for d in dimnames(v)) for (k, v) in ds.data)
    return collect(paths)
end

"""
    constraintmap(ds)

Returns a mapping of constraint patterns to specific dimension paths.
The returned dictionary has keys of type [`Pattern`](@ref) and the values are sets of
`Tuple{Vararg{Symbol}}`.
"""
function constraintmap(ds::Dataset)
    items = dimpaths(ds)
    return LittleDict{Pattern, Set{Tuple{Vararg{Symbol}}}}(
        c => Set(filter(in(c), items)) for c in ds.constraints
    )
end

"""
    dimnames(ds)

Returns a list of the unique dimension names within the [`Dataset`](@ref).
"""
function NamedDims.dimnames(ds::Dataset)
    return unique(Iterators.flatten(dimnames(a) for a in values(ds)))
end

"""
    axiskeys(ds)
    axiskeys(ds, dimname)
    axiskeys(ds, pattern)
    axiskeys(ds, dimpath)

Returns a list of unique axis keys within the [`Dataset`](@ref).
A `Tuple` will always be returned unless you explicitly specify the `dimpath` you want.
"""
function AxisKeys.axiskeys(ds::Dataset)
    return Tuple(unique(Iterators.flatten(axiskeys(a) for a in values(ds))))
end

function AxisKeys.axiskeys(ds::Dataset, dimpath::Tuple{Vararg{Symbol}})
    key, dim = dimpath[1:end-1], dimpath[end]
    component = ds.data[key]
    return axiskeys(component, dim)
end

function AxisKeys.axiskeys(ds::Dataset, pattern::Pattern)
    return Tuple(unique(axiskeys(ds, p) for p in dimpaths(ds, pattern)))
end

AxisKeys.axiskeys(ds::Dataset, dim::Symbol) = axiskeys(ds, Pattern(:__, dim))

"""
    validate(ds, [constraint])

Validate that all constrained dimension paths within a [`Dataset`](@ref) have matching key values.
Optionally, you can test an explicit constraint [`Pattern`](@ref).

# Returns
- `true` if an error isn't thrown

# Throws
- `ArgumentError`: If the constraints are not respected
"""
function validate(ds::Dataset)
    for (k, v) in constraintmap(ds::Dataset)
        validate(ds, k, v)
    end
    return true
end

function validate(ds::Dataset, constraint::Pattern)
    paths = filter(in(constraint), dimpaths(ds))
    validate(ds, constraint, paths)
    return true
end

function validate(ds::Dataset, constraint::Pattern, paths)
    if isempty(paths)
        @debug("No dimensions match the constraint $constraint")
    else
        f, r = Iterators.peel(axiskeys(ds, p) for p in paths)
        all(==(f), r) || throw(ArgumentError("Shared dimensions don't have matching keys"))
    end
    return true
end

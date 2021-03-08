"""
    Dataset{T}

A `Dataset` describes an associative collection of component `KeyedArray`s with constraints
on their shared dimensions.

# Fields
- `constraints::OrderedSet{Pattern}` - Constraint [`Pattern`s](@ref) on shared dimensions.
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

#####################################
# Dimension paths, names and keys
# - dimpaths
# - dimnames
# - axiskeys
#####################################
dimpaths(ds::Dataset, pattern::Pattern) = filter(in(pattern), dimpaths(ds))
function dimpaths(ds::Dataset)
    paths = Iterators.flatten(((k..., d) for d in dimnames(v)) for (k, v) in ds.data)
    return collect(paths)
end

function constraintmap(ds::Dataset)
    items = dimpaths(ds)
    return LittleDict{Pattern, Set{Tuple{Vararg{Symbol}}}}(
        c => Set(filter(in(c), items)) for c in ds.constraints
    )
end

# dimnames on a dataset returns the unique dimnames
function NamedDims.dimnames(ds::Dataset)
    return unique(Iterators.flatten(dimnames(a) for a in values(ds)))
end

# axiskeys on a dataset returns the unique key values
function AxisKeys.axiskeys(ds::Dataset)
    return Tuple(unique(Iterators.flatten(axiskeys(a) for a in values(ds))))
end

function AxisKeys.axiskeys(ds::Dataset, dimpath::Tuple{Vararg{Symbol}})
    key, dim = dimpath[1:end-1], dimpath[end]
    component = ds.data[key]
    return axiskeys(component, dim)
end

function AxisKeys.axiskeys(ds::Dataset, pattern::Pattern)
    return Tuple(axiskeys(ds, p) for p in dimpaths(ds, pattern))
end

AxisKeys.axiskeys(ds::Dataset, dim::Symbol) = axiskeys(ds, Pattern(:__, dim))

#############
# Validation
#############
function validate(ds::Dataset)
    for (k, v) in constraintmap(ds::Dataset)
        validate(ds, k, v)
    end
end

function validate(ds::Dataset, constraint::Pattern)
    paths = filter(in(constraint), dimpaths(ds))
    validate(ds, constraint, paths)
end

function validate(ds::Dataset, constraint::Pattern, paths::Set{Tuple{Vararg{Symbol}}})
    if isempty(paths)
        @debug("No dimensions match the constraint $constraint")
    else
        f, r = Iterators.peel(axiskeys(ds, p) for p in paths)
        all(==(f), r) || throw(ArgumentError("Shared dimensions don't have matching keys"))
    end
end

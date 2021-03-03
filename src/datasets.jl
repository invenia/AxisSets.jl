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
        f, r = firstrest(axiskeys(ds, p) for p in paths)
        all(==(f), r) || throw(ArgumentError("Shared dimensions don't have matching keys"))
    end
end

########
# Merge
########
function Base.merge(ds::Dataset, others::Dataset...)
    result = Dataset(
        union(ds.constraints, getfield.(others, :constraints)...),
        merge(ds.data, getfield.(others, :data)...),
    )

    validate(result)
    return result
end

######################
# Indexing and Lookup
######################
function Base.getproperty(ds::Dataset, sym::Symbol)
    symkey = (sym,)
    if sym in fieldnames(Dataset)
        # Fields take priority
        getfield(ds, sym)
    elseif sym in dimnames(ds)
        # If we're looking up key then return a ReadOnlyArray of it.
        # If folks want to mutate it then they're going to need to access it through
        # the nested interface.
        ReadOnlyArray(first(axiskeys(ds, sym)))
    elseif haskey(ds.data, symkey)
        # If the symkey is in the data dict then return that
        ds[symkey]
    else
        throw(ErrorException("type Dataset has no field $sym"))
    end
end

# getindex always returns a KeyedArray with shared keys wrapped in a ReadOnlyArray.
Base.getindex(ds::Dataset, key::Symbol) = getindex(ds, (key,))
function Base.getindex(ds::Dataset, key::Tuple)
    A = ds.data[key]
    names = dimnames(A)
    keys = map(zip(names, axiskeys(A))) do (n, k)
        p = (key..., n)
        any(c -> p in c, ds.constraints) ? ReadOnlyArray(k) : k
    end

    kw = NamedTuple{dimnames(A)}(keys)
    return KeyedArray(parent(A); kw...)
end

# Callable syntax is reserved for any kind of filtering of components
# where a Dataset is returned.
# NOTE: Maybe we need a SubDataset type to ensure that these selection also
# can't violate our constraints?
(ds::Dataset)(args...) = filterset(ds, args...)
filterset(ds::Dataset, key...) = filterset(ds, key)
filterset(ds::Dataset, key::Tuple) = filterset(ds, Pattern(key))
filterset(ds::Dataset, key::Pattern) = filterset(ds, in(key))
function filterset(ds::Dataset, f::Function)
    data = filter(p -> f(first(p)), pairs(ds))
    paths = collect(Iterators.flatten(((k..., d) for d in dimnames(v)) for (k, v) in data))
    constraints = filter(c -> any(in(c), paths), ds.constraints)
    result = Dataset(constraints, data)
    validate(result)
    return result
end

# Generally, we want to apply batch operations to some subset of the dataset
# We define our own mapset! function which will perform the operations and then validate the
# resulting state to ensure that we have violated any of our constraints.
mapset!(f::Function, ds::Dataset) = mapset!(f, nothing, ds)
function mapset!(op::Function, f::Union{Function, Nothing}, ds::Dataset)
    for (k, v) in ds.data
        if f === nothing || f(k)
            ds.data[k] = op(v)
        end
    end
    validate(ds)
    return ds
end

rekey!(f::Function, ds::Dataset, name::Symbol) = rekey!(f, ds, Pattern(:__, name))
function rekey!(f::Function, ds::Dataset, pattern::Pattern)
    for p in filter(in(pattern), dimpaths(ds))
        k, d = p[1:end-1], p[end]
        a = ds.data[k]

        names = dimnames(a)
        keys = map(zip(names, axiskeys(a))) do (n, vals)
            n === d ? f(vals) : vals
        end

        kw = NamedTuple{names}(keys)
        ds.data[k] = KeyedArray(parent(a); kw...)
    end
    validate(ds)
    return ds
end

# Extend Impute.jl behaviours to work across multiple components at once
function Impute.validate(ds::Dataset, validator::Validator; dims=:)
    mapset!(dims === Colon() ? in(Pattern(:__, dims)) : nothing, ds) do v
        Impute.validate(v, validator; dims=dims)
    end
end

# mapset! won't work here because we need to use a multi-pass alg.
function Impute.apply!(ds::Dataset, f::Filter; dims)
    # Limit our constraint map to paths containing the supplied dim
    cmap = filter(p -> last(first(p).segments) === dims, constraintmap(ds))

    # Apply our shared filter mask for each set of constrained paths
    # NOTE: We're assuming that the constrained paths are mutually exclusive, but in theory
    # the same component could be processed twice
    for (constraint, paths) in cmap
        @debug "$constraint => $paths"
        # We're assuming this dataset has already been validated so all dimpaths are
        # already equal
        mask = trues(length(axiskeys(ds, first(paths))))

        # Pre-extract our component keys and values
        selection = [p[1:end-1] => ds.data[p[1:end-1]] for p in paths]

        # First pass to determine our shared key mask
        for (k, v) in selection
            for (i, s) in enumerate(eachslice(v; dims=dims))
                mask[i] &= f.func(s)
            end
        end

        # Second pass to use selectdim on each component with our mask
        for (k, v) in selection
            # copy is so we don't change the data element type to a view
            ds.data[k] = copy(selectdim(v, NamedDims.dim(dimnames(v), dims), mask))
        end
    end

    return ds
end

Impute.apply(ds::Dataset, f::Filter; dims) = Impute.apply!(deepcopy(ds), f; dims=dims)

function Impute.impute!(ds::Dataset, imp::Imputor; dims, kwargs...)
    selection = filter(p -> dims in dimnames(last(p)), ds.data)

    for (k, v) in selection
        Impute.impute!(v, imp; dims=dims, kwargs...)
    end
end

###########
# flatten!
###########
function flatten!(ds::Dataset, dims::Tuple, delim=DEFAULT_PROD_DELIM)
    new_name = Symbol(join(dims, delim))
    flatten!(ds, dims => new_name, delim)
end

function flatten!(ds::Dataset, dims::Pair{<:Tuple, Symbol}, delim=DEFAULT_PROD_DELIM)
    for (k, v) in ds.data
        if first(dims) ⊆ dimnames(v)
            ds.data[k] = flatten(v, dims, delim)
            # TODO: Add algorithm to generate a new constraint on the flattened components.
        end
    end
    return ds
end

# TODO
# Wrap a bunch of the functions from AxisKeys (e.g., sortkeys, mapreduce, eachslice) and have it work across multiple data components?

# Mutating key:
# 1. Lowest level involves manipulating `ds.data[component]` directly [available]
# 2. Next lowest would require a `getindex` assignment with the updated dim axis
# if `getindex` is being performed with a dims name then we know to propagate that
# change to all components in the data. (e.g., changing timezones on target axis)
# 3. Extend `Impute.filter` or for filter along a dimension (any failure along the dimension will remove it from all components)
# 4. Extend map/mapreduce for applying impute and stats operations to multiple components with that dimension.

struct Dataset{K, T<:XArray}
    # Dims are restricted to symbols because that's a NamedDims requirement
    dims::OrderedSet{Symbol}
    # Data lookup can be by any type, but typically it'll either be symbol or tuple.
    data::LittleDict{K, T}
end

# Taking pairs is the most general constructor as it doesn't make assumptions about the
# data key type.
function Dataset(pairs::Pair{K}...; dims=()) where K
    data = LittleDict{K, XArray}(pairs...)
    dimset = OrderedSet(isempty(dims) ? Iterators.flatten(dimnames.(values(data))) : dims)

    result = Dataset(dimset, data)
    _isvalid(result) || throw(ArgumentError("Shared axes don't match."))
    return result
end

# dataset(table, dims::Symbol...) -> Dataset(dims, arrays for each remaining value columns)
function Dataset(table; dims=())
    data = LittleDict{Symbol, XArray}()
    # We need to extrac the columns to so we can isolate all values columns
    cols = Tables.columns(table)
    valcols = setdiff(Tables.columnnames(cols), dims)

    # It'd be faster to just make a copy after the first value column since the axes
    # shouldn't be changing
    for c in valcols
        data[c] = wrapdims(table, c, dims...; default=missing, sort=true, force=true)
    end

    dimset = OrderedSet(isempty(dims) ? Iterators.flatten(dimnames.(values(data))) : dims)
    result = Dataset(dimset, data)

    # This shouldn't be necessary because we're using the same axes for all components.
    _isvalid(result) || throw(ArgumentError("Shared axes don't match."))
    return result
end

# Utility kwargs constructor.
Dataset(; dims=(), kwargs...) = Dataset(kwargs...; dims=dims)

function Base.show(io::IO, ds::Dataset{K, V}) where {K, V}
    dims = tuple(ds.dims...)
    n = length(ds.data)
    print(io, "Dataset{$K}(; dims=$dims) with $n entries:")
    for (k, v) in ds.data
        printstyled(io, "\n  ", k, " => "; color=:cyan)
        printstyled(io, replace(sprint(Base.summary, v), "\n" => "\n    "); color=:cyan)
        print(io, "\n    ", replace(sprint(Base.print_array, v), "\n" => "\n    "))
    end
end

# Some utility methods for ensure axis alignments
_isvalid(ds::Dataset) = all([_isvalid(ds, name) for name in ds.dims])

function _isvalid(ds::Dataset, name::Symbol)
    ax = _getaxes(ds, name)
    # @show ax
    isempty(ax) && return false
    f, r = firstrest(ax)
    return all(x -> x == f, r)
end

function _getaxes(ds::Dataset, name::Symbol)
    return [getproperty(a, name) for a in values(ds.data) if name in dimnames(a)]
end

Base.keys(ds::Dataset) = keys(ds.data)
Base.values(ds::Dataset) = values(ds.data)
Base.pairs(ds::Dataset) = pairs(ds.data)

# Merging dataset contents in a potentially destructive way
function Base.merge(ds::Dataset, others::Dataset...)
    result = Dataset(
        union(ds.dims, getfield.(others, :dims)...),
        merge(ds.data, getfield.(others, :data)...),
    )
    _isvalid(result) || throw(ArgumentError("Shared axes don't match."))
    return result
end

# NOTE: I think this can probably be deleted as we can probably just call flatten before
# calling a constructor.
# function flatten(pairs::Pair{T, <:Dataset{T}}...; delim=nothing) where T<:Union{Symbol, AbstractString}
#     dims = union(getfield.(last.(pairs), :dims)...)
#     data = LittleDict{T, XArray}(flatten([k => v.data for (k, v) in pairs], delim))
#     result = Dataset(dims, data)
#     _isvalid(result) || throw(ArgumentError("Shared axes don't match."))
#     return result
# end

# function flatten(pairs::Pair{T, <:Dataset{K}}...) where {T, K}
#     dims = union(getfield.(last.(pairs), :dims)...)
#     data = LittleDict{Tuple, XArray}(flatten([k => v.data for (k, v) in pairs]))
#     result = Dataset(dims, data)
#     _isvalid(result) || throw(ArgumentError("Shared axes don't match."))
#     return result
# end

# Should be able to merge subcomponents with a `cat` like function along dimensions.
function Base.getproperty(ds::Dataset, sym::Symbol)
    # Early exit for the actual fields
    sym in fieldnames(Dataset) && return getfield(ds, sym)

    # If we try to access the component keyed arrays then we need to wrap the shared axes
    # in a readonly array to avoid invalid mutations
    haskey(ds.data, sym) && return ds[sym]

    # If we're trying to grab a shared axis then find the first example of it in the data
    sym in ds.dims || throw(ErrorException("type Dataset has no field $sym"))

    for a in values(ds.data)
        if sym in dimnames(a)
            return ReadOnlyArray(getproperty(a, sym))
        end
    end
    return nothing
end

# The internal method is needed to avoid ambiguities.
function _getindex(ds::Dataset, key)
    A = ds.data[key]
    names = dimnames(A)
    keys = map(zip(names, axiskeys(A))) do (n, k)
        n in ds.dims ? ReadOnlyArray(k) : k
    end

    kw = NamedTuple{dimnames(A)}(keys)
    return KeyedArray(parent(A); kw...)
end

Base.getindex(ds::Dataset{T}, key::T) where {T} = _getindex(ds, key)
Base.getindex(ds::Dataset{T}, key::T) where {T<:Tuple} = _getindex(ds, key)
# I'm not sure this is the best idea, but it would be convenient
function Base.getindex(ds::Dataset{T}, inds...) where T<:Tuple
    selected = filter(k -> inds ⊆ k, keys(ds.data))

    # We call `ds[k]` for each selected key to ensure that shared axis are made ReadOnly,
    # avoiding accidental axis key mutations.
    data = [k => ds[k] for k in selected]
    dims = intersect(ds.dims, dimnames.(last.(data))...)
    return Dataset(data...; dims=dims)
end

#=
NOTE: Not sure if these should mutate the underlying fields or just create a new Datset.
The later would be more functional and possibly less error prone?
=#

# We apply a function with `rekey!` to avoid inadvertently reordering or resizing the key.
function remapkey!(f::Function, ds::Dataset, name::Symbol)
    # Only consider components with that named dimension
    selection = filter(p -> name in dimnames(last(p)), ds.data)
    arrays = values(selection)

    # We assume that all keys for the named dims match already
    key = getproperty(first(arrays), name)
    new_key = map(f, key)

    for (k, v) in selection
        names = dimnames(v)
        keys = map(zip(names, axiskeys(v))) do (n, k)
            n === name ? new_key : k
        end

        kw = NamedTuple{dimnames(v)}(keys)
        ds.data[k] = KeyedArray(parent(v); kw...)
    end

    return ds
end

addkey!(ds::Dataset, name::Symbol) = push!(ds.dims, name)
rmkey!(ds::Dataset, name::Symbol) = delete!(ds.dims, name)

function permutekey!(ds::Dataset, name::Symbol, v)
    # Maybe we could handle multiple dims at the same time and take kwargs...?
    kw = NamedTuple{(name,)}((v,))
    for (k, v) in filter(p -> name in dimnames(last(p)), ds.data)
        ds.data[k] = getindex(v; kw...)
    end
    return ds
end

# Extend Impute.jl behaviours to work across multiple components at once
function Impute.validate(ds::Dataset, validator::Validator; dims=:)
    selection = dims === Colon() ? ds.data : filter(p -> dims in dimnames(last(p)), ds.data)

    for (k, v) in selection
        Impute.validate(v, validator; dims=dims)
    end
end

function Impute.apply!(ds::Dataset, f::Filter; dims)
    # Only consider components with that named dimension
    selection = filter(p -> dims in dimnames(last(p)), ds.data)
    mask = trues(length(getproperty(last(first(selection)), dims)))

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

    return ds
end

Impute.apply(ds::Dataset, f::Filter; dims) = Impute.apply!(deepcopy(ds), f; dims=dims)

function Impute.impute!(ds::Dataset, imp::Imputor; dims, kwargs...)
    selection = filter(p -> dims in dimnames(last(p)), ds.data)

    for (k, v) in selection
        Impute.impute!(v, imp; dims=dims, kwargs...)
    end
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

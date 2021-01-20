module AxisSets

using AxisKeys
using Impute
using IterTools
using NamedDims
using OrderedCollections
using ReadOnlyArrays
using Tables

using Impute:
    Filter,
    Imputor,
    Validator

# Short hand type for complicated union of nested Keyed or NamedDims arrays
XArray{L, T, N} = Union{NamedDimsArray{L,T,N,<:KeyedArray}, KeyedArray{T,N,<:NamedDimsArray}}

struct Dataset{T<:XArray}
    dims::OrderedSet{Symbol}
    data::LittleDict{Symbol, T}
end

function Dataset(dims::Symbol...; kwargs...)
    # If we didn't get any kwargs then just create an empty Dataset
    isempty(kwargs) && return Dataset(OrderedSet(dims), LittleDict{Symbol, XArray}())

    # Checking the eltype doesn't always work if the `NamedDimsArray` covers different dimensions
    if all(v -> isa(v, XArray), values(kwargs))
        # If we were passed a valid XArray then just construct our LittleDict
        data = LittleDict{Symbol, XArray}(kwargs...)
    else
        # Otherwise we need to assume that we're operating over a collection of tables
        data = LittleDict{Symbol, XArray}()

        # Might be a simpler way to do this if we have a `merge` function or something
        for (name, table) in kwargs
            @show typeof(table)
            # Reuse the table constructor and extra the data
            tmp = Dataset(table, dims...).data

            if length(tmp) == 1
                data[name] = first(values(tmp))
            elseif length(tmp) > 1
                for (k, v) in tmp
                    data[Symbol(name, "_", k)] = v
                end
            end
        end
    end

    dimset = OrderedSet(isempty(dims) ? Iterators.flatten(dimnames.(values(data))) : dims)
    result = Dataset(dimset, data)
    _isvalid(result) || throw(ArgumentError("Shared axes don't match."))
    return result
end

# dataset(table, dims::Symbol...) -> Dataset(dims, arrays for each remaining value columns)
function Dataset(table, dims::Symbol...)
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

# TODO: Nice show method
# 1. Print shared axes with colours
# 2. Call show on data with AxisKeys.jl view

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

function Base.getproperty(ds::Dataset, sym::Symbol)
    # Early exit for the actual fields
    sym in fieldnames(Dataset) && return getfield(ds, sym)

    # If we try to access the component keyed arrays then we need to wrap the shared axes
    # in a readonly array to avoid invalid mutations
    if haskey(ds.data, sym)
        A = ds.data[sym]
        names = dimnames(A)
        keys = map(zip(names, axiskeys(A))) do (n, k)
            n in ds.dims ? ReadOnlyArray(k) : k
        end

        kw = NamedTuple{dimnames(A)}(keys)
        return KeyedArray(parent(A); kw...)
    # If we're trying to grab a shared axis then find the first example of it in the data
    elseif sym in ds.dims
        for a in values(ds.data)
            if sym in dimnames(a)
                return ReadOnlyArray(getproperty(a, sym))
            end
        end
        return nothing
    else
        return throw(ErrorException("type Dataset has no field $sym"))
    end
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

end

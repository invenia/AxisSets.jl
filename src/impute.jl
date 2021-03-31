# Extend Impute.jl methods to work across multiple components at once
# Since Impute.jl only supports single dimension operations and handling subselection
# filtering would be tricky we aren't going to handle merging `dims` with other
# pattern keys for now.

"""
    Impute.validate(ds::KeyedDataset, validator::Validator; dims=:)

Apply the validator to components in the [`KeyedDataset`](@ref) with the specified `dims`.
"""
function Impute.validate(ds::KeyedDataset, validator::Validator; dims=:)
    patterns = dims === Colon() ? Pattern[(:__, )] : Pattern[(:__, dims)]
    return map(a -> Impute.validate(a, validator; dims=dims), ds, patterns...)
end

"""
    Impute.apply(ds::KeyedDataset, imp::DeclareMissings)

Declare missing values across all components in the [`KeyedDataset`](@ref).
"""
Impute.apply(ds::KeyedDataset, imp::DeclareMissings) = Impute.apply!(deepcopy(ds), imp)
function Impute.apply!(ds::KeyedDataset, imp::DeclareMissings)
    return map!(a -> Impute.apply!(a, imp), ds, ds)
end

"""
    Impute.impute(ds, imp; dims)

Apply the imputation algorithm `imp` along the `dims` for all components of the
[`KeyedDataset`](@ref) with that dimension.

# Example
```jldoctest
julia> using AxisKeys, Impute; using AxisSets: KeyedDataset, flatten;

julia> ds = KeyedDataset(
           flatten([
               :train => [
                   :temp => KeyedArray([1.0 1.1; missing 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                   :load => KeyedArray([7.0 7.7; 8.0 missing; 9.0 9.9]; time=1:3, loc=[:x, :y]),
                ],
                :predict => [
                   :temp => KeyedArray([1.0 missing; 2.0 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                   :load => KeyedArray([7.0 7.7; 8.1 missing; 9.0 9.9]; time=1:3, loc=[:x, :y]),
                ]
            ])...
       );

julia> [k => parent(parent(v)) for (k, v) in Impute.substitute(ds; dims=:time).data]  # KeyedArray printing isn't consistent in jldoctests
4-element Vector{Pair{Tuple{Symbol, Symbol}, Matrix{Union{Missing, Float64}}}}:
   (:train, :temp) => [1.0 1.1; 2.2 2.2; 3.0 3.3]
   (:train, :load) => [7.0 7.7; 8.0 8.0; 9.0 9.9]
 (:predict, :temp) => [1.0 1.0; 2.0 2.2; 3.0 3.3]
 (:predict, :load) => [7.0 7.7; 8.1 8.1; 9.0 9.9]

julia> [k => parent(parent(v)) for (k, v) in Impute.substitute(ds; dims=:loc).data]
4-element Vector{Pair{Tuple{Symbol, Symbol}, Matrix{Union{Missing, Float64}}}}:
   (:train, :temp) => [1.0 1.1; missing 2.2; 3.0 3.3]
   (:train, :load) => [7.0 7.7; 8.0 8.8; 9.0 9.9]
 (:predict, :temp) => [1.0 missing; 2.0 2.2; 3.0 3.3]
 (:predict, :load) => [7.0 7.7; 8.1 8.8; 9.0 9.9]
```
"""
function Impute.impute(ds::KeyedDataset, imp::Imputor; dims, kwargs...)
    return Impute.impute!(deepcopy(ds), imp; dims=dims, kwargs...)
end

function Impute.impute!(ds::KeyedDataset, imp::Imputor; dims, kwargs...)
    # NOTE: We don't use `map!` here because we know we can mutate the underlying data,
    # so we don't need to assign the value back into the `ds.data`
    selection = filter(p -> dims in dimnames(last(p)), ds.data)

    for (k, v) in selection
        Impute.impute!(v, imp; dims=dims, kwargs...)
    end

    # Just to be safe, we check that our imputation method didn't break our constraints.
    validate(ds)
    return ds
end

"""
    Impute.apply(ds, filter; dims)

Filter out missing data along the `dims` for each component in the [`KeyedDataset`](@ref)
with that dimension.

# Example
```jldoctest
julia> using AxisKeys, Impute; using AxisSets: KeyedDataset, Pattern, flatten;

julia> ds = KeyedDataset(
           flatten([
               :train => [
                   :temp => KeyedArray([1.0 1.1; missing 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                   :load => KeyedArray([7.0 7.7; 8.0 missing; 9.0 9.9]; time=1:3, loc=[:x, :y]),
                ],
                :predict => [
                   :temp => KeyedArray([1.0 missing; 2.0 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                   :load => KeyedArray([7.0 7.7; 8.1 missing; 9.0 9.9]; time=1:3, loc=[:x, :y]),
                ]
            ])...
       );

julia> [k => parent(parent(v)) for (k, v) in Impute.filter(ds; dims=:time).data]  # KeyedArray printing isn't consistent in jldoctests
4-element Vector{Pair{Tuple{Symbol, Symbol}, Matrix{Union{Missing, Float64}}}}:
   (:train, :temp) => [3.0 3.3]
   (:train, :load) => [9.0 9.9]
 (:predict, :temp) => [3.0 3.3]
 (:predict, :load) => [9.0 9.9]

julia> [k => parent(parent(v)) for (k, v) in Impute.filter(ds; dims=Pattern(:train, :__, :time)).data]
4-element Vector{Pair{Tuple{Symbol, Symbol}, Matrix{Union{Missing, Float64}}}}:
   (:train, :temp) => [1.0 1.1; 3.0 3.3]
   (:train, :load) => [7.0 7.7; 9.0 9.9]
 (:predict, :temp) => [1.0 missing; 3.0 3.3]
 (:predict, :load) => [7.0 7.7; 9.0 9.9]

julia> [k => parent(parent(v)) for (k, v) in Impute.filter(ds; dims=:loc).data]
4-element Vector{Pair{Tuple{Symbol, Symbol}, Matrix{Union{Missing, Float64}}}}:
   (:train, :temp) => [1.0 1.1; missing 2.2; 3.0 3.3]
   (:train, :load) => [7.0; 8.0; 9.0]
 (:predict, :temp) => [1.0 missing; 2.0 2.2; 3.0 3.3]
 (:predict, :load) => [7.0; 8.1; 9.0]
```
"""
Impute.apply(ds::KeyedDataset, f::Filter; dims) = Impute.apply!(deepcopy(ds), f; dims=dims)

_pattern(dims::Pattern) = dims
_pattern(dims::Tuple) = Pattern(dims)
_pattern(dims) = Pattern(:__, dims)

function Impute.apply!(ds::KeyedDataset, f::Filter; dims)
    pattern = _pattern(dims)
    dim = pattern.segments[end]

    dim in (:_, :__) && throw(ArgumentError(
        "$pattern points to an ambiguous dimension in the dataset. " *
        "Kwarg `dims` must end with a dimname."
    ))
    checkpaths = dimpaths(ds, pattern)

    # Limit our constraint map to paths containing the supplied dim
    cmap = filter(constraintmap(ds)) do (constraint, paths)
        # Because dimpath constraints only really make sense if they end with a shared
        # dimname we can just filter out constraints that don't match our desired paths.
        any(p -> p in checkpaths, paths)
    end
    # Extract component paths to be checked
    checkpaths = [p[1:end-1] for p in checkpaths]

    # Apply our shared filter mask for each set of constrained paths
    for (constraint, paths) in cmap
        @debug "$constraint => $paths"
        # We're assuming this dataset has already been validated so all dimpaths are
        # already equal
        mask = trues(length(axiskeys(ds, first(paths))))

        # Pre-extract our component keys and values
        selection = [p[1:end-1] => ds.data[p[1:end-1]] for p in paths]

        # First pass to determine our shared key mask
        for (k, v) in selection
            if k in checkpaths
                for (i, s) in enumerate(eachslice(v; dims=dim))
                    mask[i] &= f.func(s)
                end
            end
        end

        # Second pass to use selectdim on each component with our mask
        for (k, v) in selection
            # copy is so we don't change the data element type to a view
            ds.data[k] = copy(selectdim(v, NamedDims.dim(dimnames(v), dim), mask))
        end
    end

    return ds
end

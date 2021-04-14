FeatureTransforms.is_transformable(::KeyedDataset) = true

"""
    _apply_paths(ds::KeyedDataset, dims)

Based on the pattern specified by `dims`, returns a `Tuple` of
1. paths to components of `ds` that a `FeatureTransforms.Transform` should apply to,
2. the dimension of the components to apply along.
"""
function _apply_paths(ds::KeyedDataset, dims)
    pattern = _pattern(dims)

    # Get paths to components
    apply_paths = dimpaths(ds, pattern)
    apply_paths = [p[1:end-1] for p in apply_paths]

    dim = pattern.segments[end]
    if dim in (:_, :__)
        # Corresponds to element-wise apply in FeatureTransforms
        dim = Colon()
        apply_paths = unique(apply_paths)
    end

    return apply_paths, dim
end

"""
    FeatureTransforms.apply(ds::KeyedDataset, t::Transform; dims, kwargs...)

Apply the `Transform` along the `dims` for each component in the [`KeyedDataset`](@ref)
with that dimension, and return a new [`KeyedDataset`](@ref) of the transformed components.

If `dims` is a path (`Pattern` or `Tuple`), transform the components that match the path.
Otherwise, transform every component in the `KeyedDataset` that has a `dims` dimension.

Keyword arguments are passed to the equivalent `FeatureTransforms` method.

# Example
```jldoctest
julia> using AxisKeys, FeatureTransforms; using AxisSets: KeyedDataset, Pattern, flatten;

julia> ds = KeyedDataset(
           flatten([
               :train => [
                   :load => KeyedArray([7.0 7.7; 8.0 8.2; 9.0 9.9]; time=1:3, loc=[:x, :y]),
                   :price => KeyedArray([-2.0 4.0; 3.0 2.0; -1.0 -1.0]; time=1:3, id=[:a, :b]),
               ],
               :predict => [
                   :load => KeyedArray([7.0 7.7; 8.1 7.9; 9.0 9.9]; time=1:3, loc=[:x, :y]),
                   :price => KeyedArray([0.5 -1.0; -5.0 -2.0; 0.0 1.0]; time=1:3, id=[:a, :b]),
               ]
           ])...
       );

julia> p = Power(2);

julia> r = FeatureTransforms.apply(ds, p; dims=(:_, :price, :_));

julia> [k => parent(parent(v)) for (k, v) in r.data]
2-element Vector{Pair{Tuple{Symbol, Symbol}, Matrix{Float64}}}:
   (:train, :price) => [4.0 16.0; 9.0 4.0; 1.0 1.0]
 (:predict, :price) => [0.25 1.0; 25.0 4.0; 0.0 1.0]
```
"""
function FeatureTransforms.apply(ds::KeyedDataset, t::Transform; dims, kwargs...)
    apply_paths, dim = _apply_paths(ds, dims)

    pairs = map(apply_paths) do path
        path => FeatureTransforms.apply(ds.data[path], t; dims=dim, kwargs...)
    end

    return KeyedDataset(pairs...)
end

"""
    FeatureTransforms.apply!(ds::KeyedDataset, t::Transform; dims, kwargs...)

Apply the `Transform` along the `dims` for each component in the [`KeyedDataset`](@ref)
with that dimension, and return the mutated [`KeyedDataset`](@ref).

If `dims` is a path (`Pattern` or `Tuple`), transform the components that match the path.
Otherwise, transform every component in the `KeyedDataset` that has a `dims` dimension.

Keyword arguments are passed to the equivalent `FeatureTransforms` method.
"""
function FeatureTransforms.apply!(ds::KeyedDataset, t::Transform; dims, kwargs...)
    apply_paths, dim = _apply_paths(ds, dims)

    for path in apply_paths
        FeatureTransforms.apply!(ds.data[path], t; dims=dim, kwargs...)
    end

    return ds
end

"""
    FeatureTransforms.apply_append(ds::KeyedDataset, t::Transform; dims, kwargs...)

Apply the `Transform` along the `dims` for each component in the [`KeyedDataset`](@ref)
with that dimension, and return a new [`KeyedDataset`](@ref) with the result of each
transform appended to the original component.

If `dims` is a path (`Pattern` or `Tuple`), transform the components that match the path.
Otherwise, transform every component in the `KeyedDataset` that has a `dims` dimension.

Keyword arguments are passed to the equivalent `FeatureTransforms` method.
"""
function FeatureTransforms.apply_append(ds::KeyedDataset, t::Transform; dims, kwargs...)
    apply_paths, dim = _apply_paths(ds, dims)

    pairs = map(apply_paths) do path
        path => FeatureTransforms.apply_append(ds.data[path], t; dims=dim, kwargs...)
    end

    return KeyedDataset(pairs...)
end

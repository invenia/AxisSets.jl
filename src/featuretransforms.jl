FeatureTransforms.is_transformable(::KeyedDataset) = true

"""
    FeatureTransforms.apply(ds::KeyedDataset, t::Transform; dims=:, kwargs...)

Apply the `Transform` along the `dims` for each component in the [`KeyedDataset`](@ref)
with that dimension.

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

julia> [k => parent(parent(v)) for (k, v) in FeatureTransforms.apply(ds, p; dims=(:_, :price, :_)).data]
4-element Vector{Pair{Tuple{Symbol, Symbol}, Matrix{Float64}}}:
    (:train, :load) => [7.0 7.7; 8.0 8.2; 9.0 9.9]
   (:train, :price) => [4.0 16.0; 9.0 4.0; 1.0 1.0]
  (:predict, :load) => [7.0 7.7; 8.1 7.9; 9.0 9.9]
 (:predict, :price) => [0.25 1.0; 25.0 4.0; 0.0 1.0]
```
"""
function FeatureTransforms.apply(ds::KeyedDataset, t::Transform; dims, kwargs...)
    pattern = _pattern(dims)
    dim = pattern.segments[end]

    # Get paths to components
    apply_paths = dimpaths(ds, pattern)
    apply_paths = [p[1:end-1] for p in apply_paths]

    if dim in (:_, :__)
        # Corresponds to element-wise apply in FeatureTransforms
        dim = Colon()
        apply_paths = unique(apply_paths)
    end

    for path in apply_paths
        component = ds.data[path]
        ds.data[path] = FeatureTransforms.apply(component, t; dims=dim, kwargs...)
    end

    return ds
end

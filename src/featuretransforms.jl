"""
    FeatureTransforms.apply(ds::KeyedDataset, t::Transform, [key]; dims=:, kwargs...)

Apply the `Transform` to each component of the [`KeyedDataset`](@ref).
Returns a new dataset with the same constraints, but transformed components.

The transform can be applied to a subselection of components via a [`Pattern`](@ref) `key`.
Otherwise, components are selected by the desired `dims`.

Keyword arguments including `dims` are passed to the appropriate `FeatureTransforms` method
for a component.

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

julia> r = FeatureTransforms.apply(ds, p, (:_, :price, :_));

julia> [k => parent(parent(v)) for (k, v) in r.data]
4-element Vector{Pair{Tuple{Symbol, Symbol}, Matrix{Float64}}}:
    (:train, :load) => [7.0 7.7; 8.0 8.2; 9.0 9.9]
   (:train, :price) => [4.0 16.0; 9.0 4.0; 1.0 1.0]
  (:predict, :load) => [7.0 7.7; 8.1 7.9; 9.0 9.9]
 (:predict, :price) => [0.25 1.0; 25.0 4.0; 0.0 1.0]
```
"""
function FeatureTransforms.apply(ds::KeyedDataset, t::Transform, keys...; dims=:, kwargs...)
    return map(ds, _transform_pattern(keys, dims)...) do a
        FeatureTransforms.apply(a, t; dims=dims, kwargs...)
    end
end

_transform_pattern(keys, dims) = isempty(keys) ? _transform_pattern(dims) : Pattern[keys...]
_transform_pattern(::Colon) = Pattern[(:__,)]
_transform_pattern(dims::Symbol) = Pattern[(:__, dims)]
_transform_pattern(dims) = Pattern[(:__, d) for d in dims]

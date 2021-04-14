FeatureTransforms.is_transformable(::KeyedDataset) = true

_pattern(::Colon) = Pattern[(:__,)]
_pattern(dims::Symbol) = Pattern[(:__, dims)]
_pattern(dims) = Pattern[(:__, d) for d in dims]

"""
    FeatureTransforms.apply(ds::KeyedDataset, t::Transform, [key]; dims=:, kwargs...)

Apply the `Transform` to components of the [`KeyedDataset`](@ref) along dimension `dims`.
The transform can be applied to a subselection of components via a [`Pattern`](@ref) `key`.

Keyword arguments are passed to the equivalent `FeatureTransforms` method for a component.

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
function FeatureTransforms.apply(ds::KeyedDataset, t::Transform, keys...; dims=:, kwargs...)
    patterns = isempty(keys) ? _pattern(dims) : Pattern[keys...]
    return map(a -> FeatureTransforms.apply(a, t; dims=dims, kwargs...), ds, patterns...)
end

"""
    FeatureTransforms.apply!(ds::KeyedDataset, t::Transform, [key]; dims=:, kwargs...)

Apply the `Transform` to components of the [`KeyedDataset`](@ref) along dimension `dims`,
mutating the components in-place.
The transform can be applied to a subselection of components via a [`Pattern`](@ref) `key`.

Keyword arguments are passed to the equivalent `FeatureTransforms` method for a component.
"""
function FeatureTransforms.apply!(
    ds::KeyedDataset, t::Transform, keys...;
    dims=:, kwargs...
)
    patterns = isempty(keys) ? _pattern(dims) : Pattern[keys...]
    return map(a -> FeatureTransforms.apply!(a, t; dims=dims, kwargs...), ds, patterns...)
end

"""
    FeatureTransforms.apply_append(ds::KeyedDataset, t::Transform, [key]; dims=:, kwargs...)

Apply the `Transform` to components of the [`KeyedDataset`](@ref) along dimension `dims`.
The transform can be applied to a subselection of components via a [`Pattern`](@ref) `key`.

Keyword arguments are passed to the equivalent `FeatureTransforms` method for a component.
"""
function FeatureTransforms.apply_append(
    ds::KeyedDataset, t::Transform, keys...;
    inner=false, dims=:, kwargs...
)
    patterns = isempty(keys) ? _pattern(dims) : Pattern[keys...]
    if inner
        return map(ds, patterns...) do a
            FeatureTransforms.apply_append(a, t; dims=dims, kwargs...)
        end
    end
end

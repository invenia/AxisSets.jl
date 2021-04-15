FeatureTransforms.is_transformable(::KeyedDataset) = true

_transform_pattern(keys, dims) = isempty(keys) ? _transform_pattern(dims) : Pattern[keys...]
_transform_pattern(::Colon) = Pattern[(:__,)]
_transform_pattern(dims::Symbol) = Pattern[(:__, dims)]
_transform_pattern(dims) = Pattern[(:__, d) for d in dims]

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
function FeatureTransforms.apply(
    ds::KeyedDataset, t::Transform, keys...;
    dims=:, kwargs...
)
    return map(ds, _transform_pattern(keys, dims)...) do a
        FeatureTransforms.apply(a, t; dims=dims, kwargs...)
    end
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
    return map(ds, _transform_pattern(keys, dims)...) do a
        FeatureTransforms.apply!(a, t; dims=dims, kwargs...)
    end
end

"""
    FeatureTransforms.apply_append(ds::KeyedDataset, t::Transform, [key]; dims=:, kwargs...)

Apply the `Transform` to components of the [`KeyedDataset`](@ref) along dimension `dims`.
The transform can be applied to a subselection of components via a [`Pattern`](@ref) `key`.

Keyword arguments are passed to the equivalent `FeatureTransforms` method for a component.
"""
function FeatureTransforms.apply_append(
    ds::KeyedDataset, t::Transform, keys...;
    inner=false, component_name=nothing, dims=:, kwargs...
)
    patterns = _transform_pattern(keys, dims)

    if inner  # batched apply_append on each component
        return map(ds, patterns...) do a
            FeatureTransforms.apply_append(a, t; dims=dims, kwargs...)
        end
    else  # merge transformed components as new components of dataset
        # select any components the keys match
        selected = unique(x[1:end-1] for x in dimpaths(ds) if any(p -> x in p, patterns))

        # construct keys of new transformed components
        new_keys = map(selected) do k
            component_name = isnothing(component_name) ? :component : component_name
            (k[1:end-1]..., component_name)
        end

        # pair new keys with transformed components
        pairs = map(new_keys, selected) do new_k, k
            new_k => FeatureTransforms.apply(ds.data[k], t; dims=dims, kwargs...)
        end

        return merge(ds, KeyedDataset(pairs...))
    end
end

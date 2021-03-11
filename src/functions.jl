"""
    map(f, ds, [key]; dims) -> KeyedDataset

Apply function `f` to each component of the [`KeyedDataset`](@ref).
Returns a new dataset with the same constraints, but new components.
The function can be applied to a subselection of components via a [`Pattern`](@ref) `key` and/or `dims`.

# Example
```jldoctest
julia> using AxisKeys, Statistics; using AxisSets: KeyedDataset, flatten;

julia> ds = KeyedDataset(
           flatten([
               :g1 => [
                   :a => KeyedArray(zeros(3); time=1:3),
                   :b => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :y]),
                ],
                :g2 => [
                    :a => KeyedArray(ones(3); time=1:3),
                    :b => KeyedArray(zeros(3, 2); time=1:3, loc=[:x, :y]),
                ]
            ])...
       );

julia> r = map(a -> a .+ 100, ds, (:__, :a));  # KeyedArray printing isn't consistent in jldoctests

julia> [k => mean(v) for (k, v) in r.data]
4-element Array{Pair{Tuple{Symbol,Symbol},Float64},1}:
 (:g1, :a) => 100.0
 (:g1, :b) => 1.0
 (:g2, :a) => 101.0
 (:g2, :b) => 0.0
```
"""
function Base.map(f::Function, ds::KeyedDataset, key::Tuple; kwargs...)
    return map(f, ds, Pattern(key); kwargs...)
end

function Base.map(f::Function, ds::KeyedDataset, key::Pattern=Pattern(:__); dims=:)
    # Select any components where the key and dims match
    # NOTE: If this is a common operation maybe we should working into the lookup syntax?
    selected = filter(collect(keys(ds.data))) do k
        names = dimnames(ds.data[k])
        dimsmatch = (
            dims === Colon() ||
            (isa(dims, Symbol) && dims in names) ||
            (!isa(dims, Symbol) && dims ⊆ names)
        )
        return dimsmatch && k in key
    end

    return map!(f, deepcopy(ds), ds(k -> k in selected))
end

function Base.map!(f::Function, dest::KeyedDataset, src::KeyedDataset)
    # Mutate the underlying data field to avoid calling `validate`
    # on each insertion
    data = dest.data
    for (k, v) in src.data
        data[k] = f(v)
    end

    validate(dest)
    return dest
end

"""
    mapslices(f, ds, [key]; dims) -> KeyedDataset

Apply the `mapslices` call to each of the desired components and returns a new [`KeyedDataset`](@ref).
The desired components can be selected via a [`Pattern`](@ref) `key` and/or `dims`.

# Example
```jldoctest
julia> using AxisKeys, Statistics; using AxisSets: KeyedDataset;

julia> ds = KeyedDataset(
           :val1 => KeyedArray(zeros(3, 2); time=1:3, obj=[:a, :b]),
           :val2 => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :y]),
       );

julia> r = mapslices(sum, ds; dims=:time);  # KeyedArray printing isn't consistent in jldoctests

julia> [k => parent(parent(v)) for (k, v) in r.data]
2-element Array{Pair{Tuple{Symbol},Array{Float64,2}},1}:
 (:val1,) => [0.0 0.0]
 (:val2,) => [3.0 3.0]
```
"""
function Base.mapslices(f::Function, ds::KeyedDataset, args...; dims)
    return map(a -> mapslices(f, a; dims=dims), ds, args...; dims=dims)
end

"""
    merge(ds::KeyedDataset, others::KeyedDataset...)

Combine the constraints and data from multiple [`KeyedDataset`](@ref)s.

# Example
```jldoctest
julia> using AxisKeys; using AxisSets: KeyedDataset;

julia> ds1 = KeyedDataset(
           :a => KeyedArray(zeros(3); time=1:3),
           :b => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :y]),
       );

julia> ds2 = KeyedDataset(
           :c => KeyedArray(ones(3); time=1:3),
           :d => KeyedArray(zeros(3, 2); time=1:3, loc=[:x, :y]),
       );

julia> collect(keys(merge(ds1, ds2).data))
4-element Array{Tuple{Vararg{Symbol,N} where N},1}:
 (:a,)
 (:b,)
 (:c,)
 (:d,)
```
"""
function Base.merge(ds::KeyedDataset, others::KeyedDataset...)
    result = KeyedDataset(
        union(ds.constraints, getfield.(others, :constraints)...),
        merge(ds.data, getfield.(others, :data)...),
    )

    validate(result)
    return result
end

"""
    rekey(f, ds, dim)

Apply function `f` to key values of each matching `dim` in the [`KeyedDataset`](@ref).
`dim` can either by a `Symbol` or a [`Pattern`](@ref) for the dimension paths.

# Example
```jldoctest
julia> using AxisKeys; using AxisSets: KeyedDataset, rekey;

julia> ds = KeyedDataset(
           :a => KeyedArray(zeros(3); time=1:3),
           :b => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :y]),
       );

julia> r = rekey(k -> k .+ 1, ds, :time);

julia> r.time
3-element ReadOnlyArrays.ReadOnlyArray{Int64,1,UnitRange{Int64}}:
 2
 3
 4
```
"""
rekey(f::Function, ds::KeyedDataset, args...) = rekey!(f, deepcopy(ds), args...)
rekey!(f::Function, ds::KeyedDataset, dim::Symbol) = rekey!(f, ds, Pattern(:__, dim))
rekey!(f::Function, ds::KeyedDataset, dimpath::Tuple) = rekey!(f, ds, Pattern(dim))
function rekey!(f::Function, ds::KeyedDataset, pattern::Pattern)
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

# TODO: Implement cat, hcat and vcat

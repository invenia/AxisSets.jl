# Code for accessing/indexing KeyedDataset components
"""
    getproperty(ds::KeyedDataset, sym::Symbol)

Extract [`KeyedDataset`](@ref) fields, dimension keys or components in that order.
Shared axis keys are wrapped in a `ReadOnlyArray` for safety.

# Example
```jldoctest
julia> using AxisKeys; using AxisSets: KeyedDataset;

julia> ds = KeyedDataset(
           :val1 => KeyedArray(zeros(3, 2); time=1:3, obj=[:a, :b]),
           :val2 => KeyedArray(ones(3, 2) .+ 1.0; time=1:3, loc=[:x, :y]),
       );

julia> collect(keys(ds.data))
2-element Array{Tuple{Vararg{Symbol,N} where N},1}:
 (:val1,)
 (:val2,)

julia> ds.time
3-element ReadOnlyArrays.ReadOnlyArray{Int64,1,UnitRange{Int64}}:
 1
 2
 3

julia> dimnames(ds.val1)
(:time, :obj)
```
"""
function Base.getproperty(ds::KeyedDataset, sym::Symbol)
    symkey = (sym,)
    if sym in fieldnames(KeyedDataset)
        return getfield(ds, sym)
    elseif sym in dimnames(ds)
        _keys = axiskeys(ds, sym)
        if length(_keys) == 1
            return ReadOnlyArray(first(_keys))
        else
            throw(ArgumentError(
                "$sym is an ambiguous dimension in the dataset. " *
                "More than one set of values exist for this dimension: $_keys"
            ))
        end
    elseif haskey(ds.data, symkey)
        return ds[symkey]
    else
        throw(ArgumentError("type KeyedDataset has no field $sym"))
    end
end

"""
    getindex(ds::KeyedDataset, key)

Lookup [`KeyedDataset`](@ref) component by its `Tuple` key, or `Symbol` for keys of depth 1.
Shared axis keys for the returned `KeyedArray` are wrapped in a `ReadOnlyArray` for safety.

# Example
```jldoctest
julia> using AxisKeys; using AxisSets: KeyedDataset;

julia> ds = KeyedDataset(
           :val1 => KeyedArray(zeros(3, 2); time=1:3, obj=[:a, :b]),
           :val2 => KeyedArray(ones(3, 2) .+ 1.0; time=1:3, loc=[:x, :y]),
       );

julia> ds[:val1]
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
 ↓   time ∈ 3-element ReadOnlyArrays.ReadOnlyArray{Int64,...}
  →   obj ∈ 2-element ReadOnlyArrays.ReadOnlyArray{Symbol,...}
  And data, 3×2 Array{Float64,2}:
       (:a)  (:b)
  (1)   0.0   0.0
  (2)   0.0   0.0
  (3)   0.0   0.0
"""
Base.getindex(ds::KeyedDataset, key::Symbol) = getindex(ds, (key,))
function Base.getindex(ds::KeyedDataset, key::Tuple)
    A = ds.data[key]
    names = dimnames(A)
    keys = map(zip(names, axiskeys(A))) do (n, k)
        p = (key..., n)
        any(c -> p in c, ds.constraints) ? ReadOnlyArray(k) : k
    end

    kw = NamedTuple{dimnames(A)}(keys)
    return KeyedArray(parent(A); kw...)
end

"""
    setindex!(ds::KeyedDataset{T}, val, key) -> T

Store the new `val` in the [`KeyedDataset`](@ref). If any new dimension names don't any
existing constraints then `Pattern(:__, <dimname>)` is used by default.
If the axis values of the new `val` doesn't meet the existing constraints in the dataset
then an error will be throw.

# Example
```jldoctest
julia> using AxisKeys; using AxisSets: KeyedDataset, constraintmap;

julia> ds = KeyedDataset(:a => KeyedArray(zeros(3); time=1:3));

julia> ds[:b] = KeyedArray(ones(3, 2); time=1:3, lag=[-1, -2]);

julia> collect(constraintmap(ds))
2-element Array{Pair{AxisSets.Pattern,Set{Tuple{Vararg{Symbol,N} where N}}},1}:
 AxisSets.Pattern((:__, :time)) => Set([(:b, :time), (:a, :time)])
  AxisSets.Pattern((:__, :lag)) => Set([(:b, :lag)])

julia> ds[:c] = KeyedArray(ones(3, 2); time=2:4, lag=[-1, -2])
ERROR: ArgumentError: Shared dimensions don't have matching keys
```
"""
Base.setindex!(ds::KeyedDataset, val, key::Symbol) = setindex!(ds, val, (key,))
function Base.setindex!(ds::KeyedDataset, val, key::Tuple)
    ds.data[key] = val

    for d in dimnames(val)
        dimpath = (key..., d)

        # Similar to construction, if our dimpath isn't present in any existing constraints
        # then we introduce a new one that's `Pattern(:__, dim)`
        if all(c -> !in(dimpath, c), ds.constraints)
            push!(ds.constraints, Pattern(:__, d))
        end
    end

    validate(ds)
end

"""
    (ds::KeyedDataset)(key) -> KeyedDataset

A collable syntax for selecting of filtering a subset of a [`KeyedDataset`](@ref).

# Example
```jldoctest
julia> using AxisKeys; using AxisSets: KeyedDataset, flatten;

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

julia> collect(keys(ds(:__, :a).data))
2-element Array{Tuple{Vararg{Symbol,N} where N},1}:
 (:g1, :a)
 (:g2, :a)

julia> collect(keys(ds(:g1, :__).data))
2-element Array{Tuple{Vararg{Symbol,N} where N},1}:
 (:g1, :a)
 (:g1, :b)
```
"""
(ds::KeyedDataset)(args...) = _filterset(ds, args...)
_filterset(ds::KeyedDataset, key...) = _filterset(ds, key)
_filterset(ds::KeyedDataset, key::Symbol) = ds[key]
_filterset(ds::KeyedDataset, key::Tuple) = _filterset(ds, Pattern(key))
_filterset(ds::KeyedDataset, key::Pattern) = _filterset(ds, in(key))
function _filterset(ds::KeyedDataset, f::Function)
    data = filter(p -> f(first(p)), pairs(ds.data))
    paths = collect(Iterators.flatten(((k..., d) for d in dimnames(v)) for (k, v) in data))
    constraints = filter(c -> any(in(c), paths), ds.constraints)
    result = KeyedDataset(constraints, data)
    validate(result)
    return result
end

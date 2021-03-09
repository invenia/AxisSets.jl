"""
    KeyedDataset{T}

A `KeyedDataset` describes an associative collection of component `KeyedArray`s with constraints
on their shared dimensions.

# Fields
- `constraints::OrderedSet{Pattern}` - Constraint [`Pattern`](@ref)s on shared dimensions.
- `data::LittleDict{Tuple{Vararg{Symbol}}, T}` - Flattened key paths as tuples of symbols
  to each component array of type `T`.
"""
struct KeyedDataset{T<:XArray}
    # Our constraints are a collection of pseudo path tuples typically with 1 or
    # more `:_` wildcard components
    constraints::OrderedSet{Pattern}
    # Data lookup can be by any type, but typically it'll either be symbol or tuple.
    data::LittleDict{Tuple{Vararg{Symbol}}, T}
end

function KeyedDataset(pairs::Pair{<:Tuple}...; constraints=Pattern[])
    data = LittleDict{Tuple{Vararg{Symbol}}, XArray}(pairs...)

    # If no constraints have been specified then we default to (:__, dimname)
    constraint_set = if isempty(constraints)
        OrderedSet{Pattern}(
            Pattern(:__, d) for d in Iterators.flatten(dimnames.(values(data)))
        )
    else
        OrderedSet{Pattern}(constraints)
    end

    result = KeyedDataset(constraint_set, data)
    validate(result)
    return result
end

# Taking pairs is the most general constructor as it doesn't make assumptions about the
# data key type.
function KeyedDataset(pairs::Pair{Symbol}...; constraints=Pattern[])
    data = (
        Tuple(Symbol.(split(string(k), string(DEFAULT_FLATTEN_DELIM)))) => v
        for (k, v) in pairs
    )

    return KeyedDataset(data...; constraints=constraints)
end

# Utility kwargs constructor.
KeyedDataset(; constraints=Pattern[], kwargs...) = KeyedDataset(kwargs...; constraints=constraints)

function Base.show(io::IO, ds::KeyedDataset{T}) where T
    n = length(ds.data)
    print(io, "KeyedDataset{$T} with $n entries:")
    for c in ds.constraints
        print(io, "\n  ", c)
    end
    for (k, v) in ds.data
        printstyled(io, "\n  ", k, " => "; color=:cyan)
        printstyled(io, replace(sprint(Base.summary, v), "\n" => "\n    "); color=:cyan)
        print(io, "\n    ", replace(sprint(Base.print_array, v), "\n" => "\n    "))
    end
end

#################
# Dict iterators
#################
Base.keys(ds::KeyedDataset) = keys(ds.data)
Base.values(ds::KeyedDataset) = values(ds.data)
Base.pairs(ds::KeyedDataset) = pairs(ds.data)


"""
    dimpaths(ds, [pattern]) -> Vector{<:Tuple{Vararg{Symbol}}}

Return a list of all dimension paths in the [`KeyedDataset`](@ref).
Optionally, you can filter the results using a [`Pattern`](@ref).

# Example
```jldoctest
julia> using AxisKeys; using AxisSets: KeyedDataset, dimpaths;

julia> ds = KeyedDataset(
           :val1 => KeyedArray(rand(4, 3, 2); time=1:4, loc=-1:-1:-3, obj=[:a, :b]),
           :val2 => KeyedArray(rand(4, 3, 2) .+ 1.0; time=1:4, loc=-1:-1:-3, obj=[:a, :b]),
       );

julia> dimpaths(ds)
6-element Array{Tuple{Symbol,Symbol},1}:
 (:val1, :time)
 (:val1, :loc)
 (:val1, :obj)
 (:val2, :time)
 (:val2, :loc)
 (:val2, :obj)
```
"""
dimpaths(ds::KeyedDataset, pattern::Pattern) = filter(in(pattern), dimpaths(ds))
function dimpaths(ds::KeyedDataset)
    paths = Iterators.flatten(((k..., d) for d in dimnames(v)) for (k, v) in ds.data)
    return collect(paths)
end

"""
    constraintmap(ds)

Returns a mapping of constraint patterns to specific dimension paths.
The returned dictionary has keys of type [`Pattern`](@ref) and the values are sets of
`Tuple{Vararg{Symbol}}`.

# Example
```jldoctest
julia> using AxisKeys; using AxisSets: KeyedDataset, constraintmap;

julia> ds = KeyedDataset(
           :val1 => KeyedArray(rand(4, 3, 2); time=1:4, loc=-1:-1:-3, obj=[:a, :b]),
           :val2 => KeyedArray(rand(4, 3, 2) .+ 1.0; time=1:4, loc=-1:-1:-3, obj=[:a, :b]),
       );

julia> cmap = constraintmap(ds);

julia> keys(cmap)
Base.KeySet for a OrderedCollections.LittleDict{AxisSets.Pattern,Set{Tuple{Vararg{Symbol,N} where N}},Array{AxisSets.Pattern,1},Array{Set{Tuple{Vararg{Symbol,N} where N}},1}} with 3 entries. Keys:
  AxisSets.Pattern((:__, :time))
  AxisSets.Pattern((:__, :loc))
  AxisSets.Pattern((:__, :obj))

julia> values(cmap)
Base.ValueIterator for a OrderedCollections.LittleDict{AxisSets.Pattern,Set{Tuple{Vararg{Symbol,N} where N}},Array{AxisSets.Pattern,1},Array{Set{Tuple{Vararg{Symbol,N} where N}},1}} with 3 entries. Values:
  Set(Tuple{Vararg{Symbol,N} where N}[(:val2, :time), (:val1, :time)])
  Set(Tuple{Vararg{Symbol,N} where N}[(:val1, :loc), (:val2, :loc)])
  Set(Tuple{Vararg{Symbol,N} where N}[(:val2, :obj), (:val1, :obj)])
```
"""
function constraintmap(ds::KeyedDataset)
    items = dimpaths(ds)
    return LittleDict{Pattern, Set{Tuple{Vararg{Symbol}}}}(
        c => Set(filter(in(c), items)) for c in ds.constraints
    )
end

"""
    dimnames(ds)

Returns a list of the unique dimension names within the [`KeyedDataset`](@ref).

# Example
```jldoctest
julia> using AxisKeys; using NamedDims; using AxisSets: KeyedDataset;

julia> ds = KeyedDataset(
           :val1 => KeyedArray(rand(4, 3, 2); time=1:4, loc=-1:-1:-3, obj=[:a, :b]),
           :val2 => KeyedArray(rand(4, 3, 2) .+ 1.0; time=1:4, loc=-1:-1:-3, obj=[:a, :b]),
       );

julia> dimnames(ds)
3-element Array{Symbol,1}:
 :time
 :loc
 :obj
```
"""
function NamedDims.dimnames(ds::KeyedDataset)
    return unique(Iterators.flatten(dimnames(a) for a in values(ds)))
end

"""
    axiskeys(ds)
    axiskeys(ds, dimname)
    axiskeys(ds, pattern)
    axiskeys(ds, dimpath)

Returns a list of unique axis keys within the [`KeyedDataset`](@ref).
A `Tuple` will always be returned unless you explicitly specify the `dimpath` you want.

# Example
```jldoctest
julia> using AxisKeys; using AxisSets: KeyedDataset;

julia> ds = KeyedDataset(
           :val1 => KeyedArray(rand(4, 3, 2); time=1:4, loc=-1:-1:-3, obj=[:a, :b]),
           :val2 => KeyedArray(rand(4, 3, 2) .+ 1.0; time=1:4, loc=-1:-1:-3, obj=[:a, :b]),
       );

julia> axiskeys(ds)
(1:4, -1:-1:-3, [:a, :b])

julia> axiskeys(ds, :time)
(1:4,)

julia> axiskeys(ds, (:val1, :time))
1:4
```
"""
function AxisKeys.axiskeys(ds::KeyedDataset)
    return Tuple(unique(Iterators.flatten(axiskeys(a) for a in values(ds))))
end

function AxisKeys.axiskeys(ds::KeyedDataset, dimpath::Tuple{Vararg{Symbol}})
    key, dim = dimpath[1:end-1], dimpath[end]
    component = ds.data[key]
    return axiskeys(component, dim)
end

function AxisKeys.axiskeys(ds::KeyedDataset, pattern::Pattern)
    return Tuple(unique(axiskeys(ds, p) for p in dimpaths(ds, pattern)))
end

AxisKeys.axiskeys(ds::KeyedDataset, dim::Symbol) = axiskeys(ds, Pattern(:__, dim))

"""
    validate(ds, [constraint])

Validate that all constrained dimension paths within a [`KeyedDataset`](@ref) have matching key values.
Optionally, you can test an explicit constraint [`Pattern`](@ref).

# Returns
- `true` if an error isn't thrown

# Throws
- `ArgumentError`: If the constraints are not respected
"""
function validate(ds::KeyedDataset)
    for (k, v) in constraintmap(ds::KeyedDataset)
        validate(ds, k, v)
    end
    return true
end

function validate(ds::KeyedDataset, constraint::Pattern)
    paths = filter(in(constraint), dimpaths(ds))
    validate(ds, constraint, paths)
    return true
end

function validate(ds::KeyedDataset, constraint::Pattern, paths)
    if isempty(paths)
        @debug("No dimensions match the constraint $constraint")
    else
        f, r = Iterators.peel(axiskeys(ds, p) for p in paths)
        all(==(f), r) || throw(ArgumentError("Shared dimensions don't have matching keys"))
    end
    return true
end
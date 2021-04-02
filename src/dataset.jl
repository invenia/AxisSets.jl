"""
    KeyedDataset

A `KeyedDataset` describes an associative collection of component `KeyedArray`s with constraints
on their shared dimensions.

# Fields
- `constraints::OrderedSet{Pattern}` - Constraint [`Pattern`](@ref)s on shared dimensions.
- `data::LittleDict{Tuple, KeyedArray}` - Flattened key paths as tuples component keyed arrays.
"""
@auto_hash_equals struct KeyedDataset
    # Our constraints are a collection of pseudo path tuples typically with 1 or
    # more `:_` wildcard components
    constraints::OrderedSet{Pattern}
    # Data lookup can be by any type, but typically it'll either be symbol or tuple.
    data::LittleDict{Tuple, KeyedArray}

    function KeyedDataset(
        constraints::OrderedSet{Pattern},
        data::LittleDict,
        check=true
    )
        ds = new(constraints, data)
        check && validate(ds)
        return ds
    end
end

function KeyedDataset(pairs::Pair...; constraints=Pattern[])
    # Convert any non-tuple keys to tuples
    tupled_pairs = map(pairs) do (k, v)
        k isa Tuple && return k => v

        if k isa Symbol
            Tuple(Symbol.(split(string(k), string(DEFAULT_FLATTEN_DELIM)))) => v
        else
            (k,) => v
        end
    end

    data = LittleDict(tupled_pairs)

    # If no constraints have been specified then we default to (:__, dimname)
    constraint_set = if isempty(constraints)
        OrderedSet{Pattern}(
            Pattern(:__, d) for d in Iterators.flatten(dimnames.(values(data)))
        )
    else
        OrderedSet{Pattern}(constraints)
    end

    result = KeyedDataset(constraint_set, data)
    return result
end

# Utility kwargs and empty constructor.
function KeyedDataset(; constraints=Pattern[], kwargs...)
    if isempty(kwargs)
        return KeyedDataset(OrderedSet{Pattern}(constraints), LittleDict{Tuple, KeyedArray}())
    else
        return KeyedDataset(kwargs...; constraints=constraints)
    end
end

function Base.show(io::IO, ds::KeyedDataset)
    n = length(ds.data)
    m = length(ds.constraints)

    # Extract the constraints as a vector for indexing
    constraints = collect(ds.constraints)

    lines = String["KeyedDataset with:", "  $n components"]
    for (k, v) in ds.data
        # Identify shared dimensions where appropriate
        dimensions = map(dimnames(v)) do dimname
            cidx = findall(c -> (k..., dimname) in c, constraints)
            isempty(cidx) ? string(dimname) : string(dimname, "[", _only(cidx), "]")
        end

        s = string(
            "    $k => ",
            join(size(v), "x"),
            " $(nameof(typeof(v))){$(eltype(v))}",
            " with dimension ",
            join(dimensions, ", ")
        )

        push!(lines, s)
    end

    push!(lines, "  $m constraints")

    for (i, c) in enumerate(constraints)
        _axiskeys = axiskeys(ds, c)
        key_summary = isempty(_axiskeys) ? "NA" : sprint(summary, _only(_axiskeys))
        push!(lines, "    [$i] $(c.segments) ∈ $key_summary")
    end

    print(io, join(lines, "\n"))
end


"""
    dimpaths(ds, [pattern]) -> Vector{<:Tuple}

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
6-element Vector{Tuple{Symbol, Symbol}}:
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
`Tuple`.

# Example
```jldoctest
julia> using AxisKeys; using AxisSets: KeyedDataset, constraintmap;

julia> ds = KeyedDataset(
           :val1 => KeyedArray(rand(4, 3, 2); time=1:4, loc=-1:-1:-3, obj=[:a, :b]),
           :val2 => KeyedArray(rand(4, 3, 2) .+ 1.0; time=1:4, loc=-1:-1:-3, obj=[:a, :b]),
       );

julia> collect(constraintmap(ds))
3-element Vector{Pair{AxisSets.Pattern, Set{Tuple}}}:
 Pattern((:__, :time)) => Set([(:val2, :time), (:val1, :time)])
  Pattern((:__, :loc)) => Set([(:val1, :loc), (:val2, :loc)])
  Pattern((:__, :obj)) => Set([(:val2, :obj), (:val1, :obj)])
```
"""
function constraintmap(ds::KeyedDataset)
    items = dimpaths(ds)
    return LittleDict{Pattern, Set{Tuple}}(
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
3-element Vector{Symbol}:
 :time
 :loc
 :obj
```
"""
function NamedDims.dimnames(ds::KeyedDataset)
    return unique(Iterators.flatten(dimnames(a) for a in values(ds.data)))
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
    return Tuple(unique(Iterators.flatten(axiskeys(a) for a in values(ds.data))))
end

function AxisKeys.axiskeys(ds::KeyedDataset, dimpath::Tuple)
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

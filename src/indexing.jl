# Code for accessing/indexing KeyedDataset components
function Base.getproperty(ds::KeyedDataset, sym::Symbol)
    symkey = (sym,)
    if sym in fieldnames(KeyedDataset)
        # Fields take priority
        getfield(ds, sym)
    elseif sym in dimnames(ds)
        # If we're looking up key then return a ReadOnlyArray of it.
        # If folks want to mutate it then they're going to need to access it through
        # the nested interface.
        ReadOnlyArray(first(axiskeys(ds, sym)))
    elseif haskey(ds.data, symkey)
        # If the symkey is in the data dict then return that
        ds[symkey]
    else
        throw(ErrorException("type KeyedDataset has no field $sym"))
    end
end

# getindex always returns a KeyedArray with shared keys wrapped in a ReadOnlyArray.
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

# Callable syntax is reserved for any kind of filtering of components
# where a KeyedDataset is returned.
# NOTE: Maybe we need a SubDataset type to ensure that these selection also
# can't violate our constraints?
(ds::KeyedDataset)(args...) = filterset(ds, args...)
filterset(ds::KeyedDataset, key...) = filterset(ds, key)
filterset(ds::KeyedDataset, key::Symbol) = ds[key]
filterset(ds::KeyedDataset, key::Tuple) = filterset(ds, Pattern(key))
filterset(ds::KeyedDataset, key::Pattern) = filterset(ds, in(key))
function filterset(ds::KeyedDataset, f::Function)
    data = filter(p -> f(first(p)), pairs(ds.data))
    paths = collect(Iterators.flatten(((k..., d) for d in dimnames(v)) for (k, v) in data))
    constraints = filter(c -> any(in(c), paths), ds.constraints)
    result = KeyedDataset(constraints, data)
    validate(result)
    return result
end

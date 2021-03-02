# AxisSets

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/AxisSets.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://invenia.github.io/AxisSets.jl/dev)
[![Build Status](https://github.com/invenia/AxisSets.jl/workflows/CI/badge.svg)](https://github.com/invenia/AxisSets.jl/actions)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

## Key Objectives

- Wrap multiple `KeyedArray{T,N,<:NamedDimsArray}`s with shared axes
- Support dimensional operations over multiple arrays while maintaining axis consistency

About the shared axes:
- Methods for our `Dataset` guarantee shared axes consistency
- Low level manipulation of the raw components are out-of-scope
  - We minimize this risk by returning ReadOnlyArrays for shared keys to protect users from themselves
- Shared dimensions don't require matching orientations, just key values (`M[time, obj]` and `N[obj, time]` is fine))
- Component `KeyedArray`s can have non-shared dimensions and do not need to have all of the shared dimensions.

## Implemented Operations

All operations provide two main benefits:

1. Avoids needing to manually find and manipulate each `KeyedArray` containing a given shared dimension
2. Ensures that the shared dimension remains consistent across all components

### Filter

```julia
julia> Impute.filter(ds; dims=:time)
```

Removes all slices along the `:time` dimension containing a missing in any internal component `KeyedArray`.

### Permute

```julia
julia> AxisSets.permutekey!(ds, :obj, [1, 3, 2])
```

If the `:obj` dimension values are `[:a, :b, :c]` then this would reorder them and their
corresponding array slices to `[:a, :c, :b]` for each internal component `KeyedArray`.

### Rekey

```julia
AxisSets.remapkey!(dt -> ZonedDateTime(dt, tz"UTC"), ds, :time)
```

Applies the mutation function (`ZonedDateTime` conversion) to all `:time` axis key values.

### Flatten

```julia
AxisSets.flatten(ds, (:loc, :obj))
```
Combines the `:loc` and `:obj` dimensions into one `:locᵡobj` axis and merges the keys into
a Symbol of `:<loc_key>ᵡ<obj_key>)`.

Different delimiter
```julia
AxisSets.flatten(ds, (:loc, :obj), :_)
```

Get new key as tuples rather than symbols:
```julia
AxisSets.flatten(ds, (:loc, :obj) => :feature)
```

Similar to the other operations, these are done for all components that share these dimensions.

TODO: Support `unflatten`

### Tables

Constructing Datasets from a table with key columns and one or more value columns
```julia
# Table with time, loc, obj, val1, and val2 columns
# Reminder the table could be a dataframe, csv, or libpq results
ds = Dataset(table, :time, :loc, :obj)
```
You could also construct a `Dataset` by calling:
```julia
key = (:time, :loc, :obj)
ds = Dataset(
    key...;
    val1=AxisKeys.wrapdims(table, :val1, key...),
    val2=AxisKeys.wrapdims(table, :val2, key...),
)
```
The later is particulary useful if you want to combine datasets that should still have
matching (or partially matching) key values.

Likewise a Dataset implements the tables interface, meaning that you can call
`Tables.row` and `Tables.columns` functions on it can be used with packages like:

1. PrettyTables.jl
2. TableOperations.jl
3. StatsPlots.jl

NOTE: Needs some examples

# Example

In this example, we're going to step through a set of common operations we typically perform when converting a collection of individually fetched features into a simple set of training fetures (X, y) and predict/testing features (X̂, ŷ).

Lets start by loading some packages we'll need.

```@example full
using AxisKeys, AxisSets, DataFrames, Dates, Impute, Random, TimeZones
using AxisSets: Pattern, flatten, rekey
```

## Data

```@setup full
# Only gonna lookback 4 days on the same hour and our training data only covers 1 week
train_input_times = DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 7)
train_output_times = train_input_times .+ Day(1)

# Predict times are just the next day
predict_input_times = DateTime(2021, 1, 7):Hour(1):DateTime(2021, 1, 8)
predict_output_times = predict_input_times .+ Day(1)

# We're gonna say that price, load and temp nodes are different non-overlapping ids
node_ids = [:a, :b, :c, :d]
load_ids = [:p, :q]
temp_ids = [:x, :y, :z]

# Lookback lag for prices, we're only applying the lag to prices just to keep things interesting.
feature_lags = Day.(-1:-1:-4)
rng = MersenneTwister(1234)

# Some
misratios = Dict(
    :a => 0.1,
    :b => 0.2,
    :c => 0.3,
    :d => 0.4,
    :p => 0.1,
    :q => 0.2,
    :x => 0.1,
    :y => 0.2,
    :z => 0.3,
)

train_factor = 0.2
predict_factor = 0.1

# A modified `rand` which has some probability of producing `missing` values based on
# the id and some preset factor
function misrand(factor, id)::Union{Missing, Float64}
    ratio = factor * misratios[id]
    @assert ratio >= 0.0
    rand(rng) > ratio || return missing
    return rand(rng)
end



data = (
    train = (
        input = (
            prices = DataFrame(
                NamedTuple{(:time, :id, :lag, :price)}((t..., misrand(train_factor, t[2])))
                for t in Iterators.product(train_input_times, node_ids, feature_lags)
            ),
            load = DataFrame(
                NamedTuple{(:time, :id, :load)}((t..., misrand(train_factor, t[2])))
                for t in Iterators.product(train_input_times, load_ids)
            ),
            temp = DataFrame(
                NamedTuple{(:time, :id, :temperature)}((t..., misrand(train_factor, t[2])))
                for t in Iterators.product(train_input_times, temp_ids)
            ),
        ),
        output = (
            prices = DataFrame(
                NamedTuple{(:time, :id, :price)}((t..., misrand(train_factor, t[2])))
                for t in Iterators.product(train_output_times, node_ids)
            ),
        ),
    ),
    predict = (
        input = (
            prices = DataFrame(
                NamedTuple{(:time, :id, :lag, :price)}((t..., misrand(predict_factor, t[2])))
                for t in Iterators.product(predict_input_times, node_ids, feature_lags)
            ),
            load = DataFrame(
                NamedTuple{(:time, :id, :load)}((t..., misrand(predict_factor, t[2])))
                for t in Iterators.product(predict_input_times, load_ids)
            ),
            temp = DataFrame(
                NamedTuple{(:time, :id, :temperature)}((t..., misrand(predict_factor, t[2])))
                for t in Iterators.product(predict_input_times, temp_ids)
            ),
        ),
        output = (
            prices = DataFrame(
                NamedTuple{(:time, :id, :price)}((t..., misrand(predict_factor, t[2])))
                for t in Iterators.product(predict_output_times, node_ids)
            ),
        ),
    ),
)
```

To see how we can use AxisSets.jl to aid with data wrangling problems, we're going to assume our dataset is a nested `NamedTuple` of `DataFrame`s.
We're using DataFrames for simplicity, but we could also construct our `Dataset` from a `LibPQ.Result` via the Tables interface.
Let's start by taking a look at what our data looks like.
To make things easier we're gonna flatten our nested structure and display the column names for each dataframe.

```@example full
flattened = AxisSets.flatten(data)
Dict(k => names(v) for (k, v) in pairs(flattened))
```

Something you may notice about our data is that each component has a `:time` and `:id` column which uniquely identify each value.
Therefore we can more compactly represent our components as `AxisKeys.KeyedArray`s.

```@example full
components = (
    k => allowmissing(wrapdims(v, Tables.columnnames(v)[end], Tables.columnnames(v)[1:end-1]...))
    for (k, v) in pairs(flattened)
)
```

This representation avoids storing duplicate `:time` and `:id` column values and allows us to perform normal n-dimensional array operation over the dataset more efficiently.

If we look al little closer we'll also find that several of these "key" columns align across the dataframes, while others do not.

For example, the `:time` columns across `train⁻input` tables align.
Similarly the `:id` columns match for both `train⁻input⁻prices` and `train⁻output⁻prices`.

```@example full
@assert issetequal(flattened.train⁻input⁻temp.time, flattened.train⁻input⁻load.time)
@assert issetequal(flattened.train⁻input⁻prices.id, flattened.train⁻output⁻prices.id)
```

However, not all `time` or `id` columns need to align.

```@example full
@assert !issetequal(flattened.train⁻input⁻prices.time, flattened.train⁻output⁻prices.time)
@assert !issetequal(flattened.train⁻input⁻temp.id, flattened.train⁻input⁻load.id)
```

It turns out we can summarize these alignment "constraints" pretty concisely.

1. All `time` columns must align within each of the 4 `train`/`predict` x `input`/`output` combinations.
2. All `id` columns must align for each `prices`, `temp` and `load`.

With AxisSets.jl we can declaratively state these alignment assumptions using [`Pattern`](@ref AxisSets.Pattern)s.

Constraint patterns on `:time`

```@example full
time_constraints = Pattern[
    # All train input time keys should match
    (:train, :input, :_, :time),

    # All train output time keys should match
    (:train, :output, :_, :time),

    # All predict input time keys should match
    (:predict, :input, :_, :time),

    # All predict output time keys should match
    (:predict, :output, :_, :time),
]
```

Constraint patterns on `:id`

```@example full
id_constraints = Pattern[
    # All ids for each data type should align across
    (:__, :prices, :id),
    (:__, :temp, :id),
    (:__, :load, :id),
]
```

## KeyedDataset

Okay, so how can we make the constraint `Pattern`s and component `KeyedArray`s more useful to us?
Well, we can now combine our constraints and component `KeyedArray`s into a `KeyedDataset`.

```@example full
ds = KeyedDataset(components...; constraints=vcat(time_constraints, id_constraints))
```

The objective of this type is to address two primary issues:

1. Ensure that our data wrangling operations won't violate our constraints outlined above.
2. Provide batched operations to minimize verbose data wrangling operations.

Let's perform some common operations:

We often want to filter out `id`s being consider if they have too many missing values.
Let's define a rule for when we want to filter out an `id`.

```@example full
threshold = 0.1
function filter_rule(x)
    r = count(ismissing, x) / length(x)
    r < threshold
end
```

Okay, so lets try just applying this filtering rule to each component of our dataset

```@example full
unique(axiskeys(Impute.filter(filter_rule, v; dims=:id), :id) for (k, v) in ds.data)
```

We can see that doing this results in inconsistent `:id` keys across our components.
Now lets try applying a batched version of that filtering rule across the entire dataset.

```@example full
ds = Impute.filter(filter_rule, ds; dims=:id)
unique(axiskeys(ds, :id))
```

Notice how our returned `KeyedDataset` respects the `:id` constraints we provided above.
Another kind of filtering we often do is dropping hours with any missing data after this point.

```@example full
ds = Impute.filter(ds; dims=:time)
unique(axiskeys(ds, :time))
```

You'll notice that we may have up-to 4 unique `:time` keys among our 8 components.
This is because we only expect keys to align across each `:train`/`predict` and `input`/`output` combinations as described above.

Finally, we should be able to restrict the component `KeyedArrays` to `disallowmissing`.

```@example full
ds = map(disallowmissing, ds)
```

Another common operation is to mutate the key values in batches.
In this case, we'll say that we need to convert the `:time` keys to `ZonedDateTime`s.

```@example full
ds = rekey(k -> ZonedDateTime.(k, tz"UTC"), ds, :time)
```

Okay, so now that all of our data manipulation is complete we want to combine all our components into 4 simple 2-d matrices

```@example full
results = (
    X = hcat(
        flatten(ds[(:train, :input, :prices)], (:id, :lag) => :id),
        ds[(:train, :input, :temp)],
        ds[(:train, :input, :load)],
    ),
    y = ds[(:train, :output, :prices)],
    X̂ = hcat(
        flatten(ds[(:predict, :input, :prices)], (:id, :lag) => :id),
        ds[(:predict, :input, :temp)],
        ds[(:predict, :input, :load)],
    ),
    ŷ = ds[(:predict, :output, :prices)],
)
results.X
```

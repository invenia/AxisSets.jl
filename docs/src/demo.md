# Demo

In this demo, we're going to step through a set of common operations we typically perform when converting a collection of individually fetch features into a simple set of training fetures (X, y) and predict/testing features (X', y').

Lets start by loading some packages we'll need.

```@repl demo
using AxisKeys, AxisSets, DataFrames, Dates, Impute, Random, TimeZones
using AxisSets: Dataset, Pattern
```

Now we'll create some arbitary datasets.
We're using DataFrames for simplicity, but we could also construct our `Dataset` from a `LibPQ.Result` via the Tables interface.

```@repl demo
# Only gonna lookback 4 days on the same hour and our training data only covers 1 week
train_input_times = DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 7);
train_output_times = train_input_times .+ Day(1);

# Predict times are just the next day
predict_input_times = DateTime(2021, 1, 7):Hour(1):DateTime(2021, 1, 8);
predict_output_times = predict_input_times .+ Day(1);

# We're gonna say that price, load and temp nodes are different non-overlapping ids
node_ids = [:a, :b, :c, :d];
load_ids = [:p, :q];
temp_ids = [:x, :y, :z];

# Lookback lag for prices, we're only applying the lag to prices just to keep things interesting.
feature_lags = Day.(-1:-1:-4);

rng = MersenneTwister(1234);

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
);

# A modified `rand` which has some probability of producing `missing` values based on
# the id and some preset factor
function misrand(factor, id)::Union{Missing, Float64}
    ratio = factor * misratios[id]
    @assert ratio >= 0.0
    rand(rng) > ratio || return missing
    return rand(rng)
end;

train_factor = 0.2;
predict_factor = 0.1;

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

Okay, so we now have an awkward nested structure of training and predicting inputs and outputs.

What are some of the assumption or constraints that must hold throughout our data wrangling and analysis?

1. All the time related columns must maintain consistency with one another (e.g., `train_input_times`, `predict_input_times`)
2. All ids must maintain consistency between the train and predict inputs. (e.g., `node_id`, `load_id`, `temp_id`)

Lets see if we can perform some common cleanup and transformation on this.

```@repl demo
components = (
    k => allowmissing(wrapdims(v, Tables.columnnames(v)[end], Tables.columnnames(v)[1:end-1]...))
    for (k, v) in pairs(AxisSets.flatten(data))
);
constraints = Pattern[
    # All train input time keys should match
    (:train, :input, :_, :time),
    # All train output time keys should match
    (:train, :output, :_, :time),

    # All predict input time keys should match
    (:predict, :input, :_, :time),
    # All predict output time keys should match
    (:predict, :output, :_, :time),

    # All ids for each data type should align across
    (:__, :prices, :id),
    (:__, :temp, :id),
    (:__, :load, :id),
];
ds = Dataset(components...; constraints=constraints)
```

Let's perform some common operations:

We often want to filter out `id`s being consider if they have too many missing values.
Let's define a rule for when we want to filter out an `id`.
```@repl demo
threshold = 0.1;
function filter_rule(x)
    r = count(ismissing, x) / length(x)
    r < threshold
end;
```
Okay, so lets try just applying this filtering rule to each component of our dataset
```@repl demo
unique(axiskeys(Impute.filter(filter_rule, v; dims=:id), :id) for (k, v) in ds.data)
```

We can see that doing this results in inconsistent `:id` keys across our components.
Now lets try applying a batched version of that filtering rule across the entire dataset.

```@repl demo
ds = Impute.filter(filter_rule, ds; dims=:id);
unique(axiskeys(ds, :id))
```

Notice how our returned `Dataset` respects the `:id` constraints we provided above.
Another kind of filtering we often do is dropping hours with any missing data after this point.

```@repl demo
ds = Impute.filter(ds; dims=:time);
unique(axiskeys(ds, :time))
```
You'll notice that we may have up-to 4 unique `:time` keys among our 8 components.
This is because we only expect keys to align across each `:train`/`predict` and `input`/`output` combinations as described above.

Finally, we should be able to restrict the component `KeyedArrays` to `disallowmissing`.

```@repl demo
# NOTE: I don't love the name of this function, maybe we could just overload `map!`?
AxisSets.mapset!(disallowmissing, ds)
```

Another common workflow need is to mutate the key values in batches.
In this case, we'll say that we need to convert the `:time` keys to `ZonedDateTime`s.
```@repl demo
# Again, I'm open to another names
AxisSets.rekey!(k -> ZonedDateTime.(k, tz"UTC"), ds, :time)
```

Optionally, we could also choose to flatten our inputs/ouputs into 2-d matrices which is what many ML algorithms expect.
```@repl demo
AxisSets.flatten!(ds, (:id, :lag) => :id)
```

Okay, so now that all of our data manipulation is complete we might want to collapse this back to a named tuple of X and y values to fit the usual ML notation.

```@repl demo
results = (
    X = hcat(values(ds(:train, :input, :_))...),
    y = ds[(:train, :output, :prices)],
    X̂ = hcat(values(ds(:predict, :input, :_))...),
    ŷ = ds[(:predict, :output, :prices)],
)
```

Questions:
- Show the before or after?
- Should we construct a new dataset at the end rather than a `NamedTuple`?

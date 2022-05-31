"""
    Pattern

A pattern is just a wrapper around a `Tuple` which enables searching and
filtering for matching components and dimension paths in a [`KeyedDataset`](@ref).
Special symbols `:_` and `:__` are used as wildcards, similar to `*` and `**` in glob
pattern matching.

# Example
```jldoctest
julia> using AxisSets: Pattern;

julia> items = [
           (:train, :input, :load, :time),
           (:train, :input, :load, :id),
           (:train, :input, :temperature, :time),
           (:train, :input, :temperature, :id),
           (:train, :output, :load, :time),
           (:train, :output, :load, :id),
       ];

julia> filter(in(Pattern(:__, :time)), items)
3-element Vector{NTuple{4, Symbol}}:
 (:train, :input, :load, :time)
 (:train, :input, :temperature, :time)
 (:train, :output, :load, :time)

julia> filter(in(Pattern(:__, :load, :_)), items)
4-element Vector{NTuple{4, Symbol}}:
 (:train, :input, :load, :time)
 (:train, :input, :load, :id)
 (:train, :output, :load, :time)
 (:train, :output, :load, :id)
```
"""
@auto_hash_equals struct Pattern{T<:Tuple}
    segments::T

    # We reduce the segments on construction to remove extra wildcards
    # e.g., `(:_, :__, ...)` reduces to`(:__, ...`)
    # NOTE: Since these patterns shouldn't be very long it seemed alright to do in an
    # inner constructor
    function Pattern(segments::Tuple)
        n = length(segments)
        mask = trues(n)

        # Prev index used for lookup up segment and mask values as we iterate through
        j = 1

        # Iterating forwards and backwards seemed like the easiest way to
        # remove extra wildcards on either side of a `:__`
        for i in Iterators.flatten([2:n, n-1:-1:1])
            curr = segments[i]
            curr_mask = mask[i]
            prev = segments[j]
            prev_mask = mask[j]

            if curr_mask
                if curr === :__ && prev in (:_, :__) && prev_mask
                    mask[j] = false
                else
                    j = i
                end
            end
        end

        vals = segments[mask]
        return new{typeof(vals)}(vals)
    end
end

Pattern(segments...) = Pattern(segments)
Base.convert(::Type{Pattern}, x::Tuple) = Pattern(x)

Base.show(io::IO, pattern::Pattern) = print(io, "Pattern($(pattern.segments))")

function Base.in(item::Tuple, pattern::Pattern)
    # If our pattern has more segments than the item then it isn't going to match
    length(pattern.segments) <= length(item) || return false

    # Setup our item and pattern iterators
    item_iter = iterate(item)
    pat_iter = iterate(pattern.segments)

    # Loop as long as neither iterator is exhausted
    while item_iter !== nothing && pat_iter !== nothing
        # Extract values and states
        item_val, item_st = item_iter
        pat_val, pat_st = pat_iter

        # Iterate as normal if the pattern value matches, is a subtype or it's :_
        if (
            (item_val isa Type && pat_val isa Type && item_val <: pat_val) ||
            isequal(item_val, pat_val) ||
            pat_val === :_
        )
            pat_iter = iterate(pattern.segments, pat_st)
            item_iter = iterate(item, item_st)
        # Look ahead when we see a multi-value wildcard to see if the next value matches
        elseif pat_val === :__
            next_iter = iterate(pattern.segments, pat_st)

            # Early exit if we've reached the end of our pattern
            next_iter === nothing && return true
            next_val, next_st = next_iter

            # If the next value will match then bump the pattern iterator state
            # otherwise just bump the item iterate and we'll check again next time.
            if (
                (item_val isa Type && next_val isa Type && item_val <: next_val) ||
                isequal(item_val, next_val)
            )
                pat_iter = next_iter
            else
                item_iter = iterate(item, item_st)
            end
        else
            # Early exit if it isn't a match or a wildcard
            return false
        end
    end

    # Return `true` if we've exhausted both iterators
    return item_iter === nothing && pat_iter === nothing
end


"""
    Patter

A pattern is just a wrapper around a `Tuple{Vararg{Symbol}}` which enables searching for
matching components and dimension paths in a Dataset.
Special symbols `:_` and `:__` are used as wildcards, similar to `*` and `**` in glob
pattern matching.
"""
struct Pattern
    segments::Tuple{Vararg{Symbol}}
end

Pattern(segments::Symbol...) = Pattern(segments)
Base.convert(::Type{Pattern}, x::Tuple{Vararg{Symbol}}) = Pattern(x)

# Primary functionality is to support identify tuples that match the constraint
function Base.in(item::Tuple{Vararg{Symbol}}, pattern::Pattern)
    pat = pattern.segments
    i = 1   # item  index
    j = 1   # contraint index
    n = length(item)
    m = length(pat)

    # If our constrain has more segments than the item then it isn't going to match
    m <= n || return false

    result = true
    while result && i <= n && j <= m
        if pat[j] === item[i] || pat[j] === :_
            i += 1
            j += 1
        elseif pat[j] === :__
            # Lookahead if we aren't at the end
            if j < m
                _next = pat[j+1]

                # If the next value matches the current item then just jump to the indices after both
                if _next === item[i]
                    i += 1
                    j += 2
                # Error if have multiple multiple wildcards in a row
                elseif _next === :__ || _next === :_
                    throw(ArgumentError("Cannot have multiple :_ or :__ wildcards in a row"))
                # Otherwise we just bump the item index
                else
                    i += 1
                end
            else
                # If our constraint ends with :__ then we automatically match the rest and
                # we can bump the index
                j += 1
            end
        else
            # If our segments don't match and aren't wildcards then it doesn't match
            result = false
        end
    end

    return result && i > n && j > m
end

# items = [
#     (:train, :input, :prices, :time),
#     (:train, :input, :prices, :id),
#     (:train, :input, :prices, :lag),
#     (:train, :input, :load, :time),
#     (:train, :input, :load, :id),
#     (:train, :input, :temperature, :time),
#     (:train, :input, :temperature, :id),
#     (:train, :output, :prices, :time),
#     (:train, :output, :prices, :id),
#     (:predict, :input, :prices, :time),
#     (:predict, :input, :prices, :id),
#     (:predict, :input, :prices, :lag),
#     (:predict, :input, :load, :time),
#     (:predict, :input, :load, :id),
#     (:predict, :input, :temperature, :time),
#     (:predict, :input, :temperature, :id),
#     (:predict, :output, :prices, :time),
#     (:predict, :output, :prices, :id),
# ]

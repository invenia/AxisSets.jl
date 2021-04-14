# Convert a dims argument to a Pattern
_pattern(dims::Pattern) = dims
_pattern(dims::Tuple) = Pattern(dims)
_pattern(::Colon) = Pattern(:__)
_pattern(dims) = Pattern(:__, dims)

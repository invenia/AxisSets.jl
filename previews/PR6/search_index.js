var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = AxisSets","category":"page"},{"location":"#AxisSets","page":"Home","title":"AxisSets","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [AxisSets]","category":"page"},{"location":"#AxisSets.DEFAULT_FLATTEN_DELIM","page":"Home","title":"AxisSets.DEFAULT_FLATTEN_DELIM","text":"DEFAULT_FLATTEN_DELIM\n\n:⁻ (or \\^-)\n\nSeparates the parent symbols of a nested NamedTuple that has been flattened. A less common symbol was used to avoid collisions with :_ in the parent symbols.\n\nExample\n\njulia> using AxisSets: flatten\n\njulia> data = (\n           val1 = (a1 = 1, a2 = 2),\n           val2 = (b1 = 11, b2 = 22),\n           val3 = [111, 222],\n           val4 = 4.3,\n       );\n\njulia> flatten(data)\n(val1⁻a1 = 1, val1⁻a2 = 2, val2⁻b1 = 11, val2⁻b2 = 22, val3 = [111, 222], val4 = 4.3)\n\n\n\n\n\n","category":"constant"},{"location":"#AxisSets.DEFAULT_PROD_DELIM","page":"Home","title":"AxisSets.DEFAULT_PROD_DELIM","text":"DEFAULT_PROD_DELIM\n\n:ᵡ (or \\^x)\n\nSeparates the parent symbols from an n-dimensional array that was flattened / reshaped. A less common symbol was used to avoid collisions with :_ in the parent symbols.\n\nExample\n\njulia> using AxisKeys, Dates; using AxisSets: flatten\n\njulia> A = KeyedArray(\n           reshape(1:24, (4, 3, 2));\n           time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),\n           obj=[:a, :b, :c],\n           loc=[1, 2],\n       );\n\njulia> axiskeys(flatten(A, (:obj, :loc)), :objᵡloc)\n6-element Array{Symbol,1}:\n :aᵡ1\n :bᵡ1\n :cᵡ1\n :aᵡ2\n :bᵡ2\n :cᵡ2\n\n\n\n\n\n","category":"constant"},{"location":"#AxisSets.Dataset","page":"Home","title":"AxisSets.Dataset","text":"Dataset{T}\n\nA Dataset describes an associative collection of component KeyedArrays with constraints on their shared dimensions.\n\nFields\n\nconstraints::OrderedSet{Pattern} - Constraint Patterns on shared dimensions.\ndata::LittleDict{Tuple{Vararg{Symbol}}, T} - Flattened key paths as tuples of symbols to each component array of type T.\n\n\n\n\n\n","category":"type"},{"location":"#AxisSets.Pattern","page":"Home","title":"AxisSets.Pattern","text":"Pattern\n\nA pattern is just a wrapper around a Tuple{Vararg{Symbol}} which enables searching and filtering for matching components and dimension paths in a Dataset. Special symbols :_ and :__ are used as wildcards, similar to * and ** in glob pattern matching.\n\nExample\n\njulia> using AxisSets: Pattern;\n\njulia> items = [\n           (:train, :input, :load, :time),\n           (:train, :input, :load, :id),\n           (:train, :input, :temperature, :time),\n           (:train, :input, :temperature, :id),\n           (:train, :output, :load, :time),\n           (:train, :output, :load, :id),\n       ];\n\njulia> filter(in(Pattern(:__, :time)), items)\n3-element Array{NTuple{4,Symbol},1}:\n (:train, :input, :load, :time)\n (:train, :input, :temperature, :time)\n (:train, :output, :load, :time)\n\njulia> filter(in(Pattern(:__, :load, :_)), items)\n4-element Array{NTuple{4,Symbol},1}:\n (:train, :input, :load, :time)\n (:train, :input, :load, :id)\n (:train, :output, :load, :time)\n (:train, :output, :load, :id)\n\n\n\n\n\n","category":"type"},{"location":"#AxisKeys.axiskeys-Tuple{AxisSets.Dataset}","page":"Home","title":"AxisKeys.axiskeys","text":"axiskeys(ds)\naxiskeys(ds, dimname)\naxiskeys(ds, pattern)\naxiskeys(ds, dimpath)\n\nReturns a list of unique axis keys within the Dataset. A Tuple will always be returned unless you explicitly specify the dimpath you want.\n\n\n\n\n\n","category":"method"},{"location":"#AxisSets.constraintmap-Tuple{AxisSets.Dataset}","page":"Home","title":"AxisSets.constraintmap","text":"constraintmap(ds)\n\nReturns a mapping of constraint patterns to specific dimension paths. The returned dictionary has keys of type Pattern and the values are sets of Tuple{Vararg{Symbol}}.\n\n\n\n\n\n","category":"method"},{"location":"#AxisSets.dimpaths-Tuple{AxisSets.Dataset,AxisSets.Pattern}","page":"Home","title":"AxisSets.dimpaths","text":"dimpaths(ds, [pattern]) -> Vector{<:Tuple{Vararg{Symbol}}}\n\nReturn a list of all dimension paths in the Dataset. Optionally, you can filter the results using a Pattern.\n\n\n\n\n\n","category":"method"},{"location":"#AxisSets.flatten","page":"Home","title":"AxisSets.flatten","text":"flatten(collection, [delim])\n\nFlatten a collection of nested associative types into a flat collection of pairs. If the input keys are symbols (ie: NamedTuple) then the ⁻  will be used, otherwise Tuple keys will be returned.\n\nExample\n\njulia> using AxisSets: flatten\n\njulia> data = (\n           val1 = (a1 = 1, a2 = 2),\n           val2 = (b1 = 11, b2 = 22),\n           val3 = [111, 222],\n           val4 = 4.3,\n       );\n\njulia> flatten(data)\n(val1⁻a1 = 1, val1⁻a2 = 2, val2⁻b1 = 11, val2⁻b2 = 22, val3 = [111, 222], val4 = 4.3)\n\nflatten(A, dims, [delim])\n\nFlatten a KeyedArray along the specified consecutive dimensions. The dims argument can either be a Tuple of symbols or a Pair{Tuple, Symbol} if you'd like to specify the desired flattened dimension name. If the dims is just a Tuple with no output dimension specified then ᵡ will be used to generate the new dimension name.\n\nExample\n\njulia> using AxisKeys, Dates, NamedDims; using AxisSets: flatten\n\njulia> A = KeyedArray(\n           reshape(1:24, (4, 3, 2));\n           time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),\n           obj=[:a, :b, :c],\n           loc=[1, 2],\n       );\n\njulia> dimnames(flatten(A, (:obj, :loc)))\n(:time, :objᵡloc)\n\n\n\n\n\n","category":"function"},{"location":"#AxisSets.validate-Tuple{AxisSets.Dataset}","page":"Home","title":"AxisSets.validate","text":"validate(ds, [constraint])\n\nValidate that all constrained dimension paths within a Dataset have matching key values. Optionally, you can test an explicit constraint Pattern.\n\nReturns\n\ntrue if an error isn't thrown\n\nThrows\n\nArgumentError: If the constraints are not respected\n\n\n\n\n\n","category":"method"},{"location":"#NamedDims.dimnames-Tuple{AxisSets.Dataset}","page":"Home","title":"NamedDims.dimnames","text":"dimnames(ds)\n\nReturns a list of the unique dimension names within the Dataset.\n\n\n\n\n\n","category":"method"}]
}

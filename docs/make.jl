using AxisSets
using Documenter

makedocs(;
    modules=[AxisSets],
    authors="Invenia Technical Computing Corporation",
    repo="https://github.com/invenia/AxisSets.jl/blob/{commit}{path}#L{line}",
    sitename="AxisSets.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://invenia.github.io/AxisSets.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/invenia/AxisSets.jl",
)

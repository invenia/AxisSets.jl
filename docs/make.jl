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
        assets=String["assets/invenia.css"],
    ),
    pages=[
        "Home" => "index.md",
        "Example" => "example.md",
        "API" => "api.md",
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/invenia/AxisSets.jl",
    devbranch = "main",
    push_preview = true,
)

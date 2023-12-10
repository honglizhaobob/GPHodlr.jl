using GPHodlr
using Documenter

DocMeta.setdocmeta!(GPHodlr, :DocTestSetup, :(using GPHodlr); recursive=true)

makedocs(;
    modules=[GPHodlr],
    authors="Hongli Zhao",
    repo="https://github.com/honglizhaobob/GPHodlr.jl/blob/{commit}{path}#{line}",
    sitename="GPHodlr.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://honglizhaobob.github.io/GPHodlr.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/honglizhaobob/GPHodlr.jl",
    devbranch="main",
)

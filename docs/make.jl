using SecondOrderPerturbationTheory
using Documenter

DocMeta.setdocmeta!(SecondOrderPerturbationTheory, :DocTestSetup, :(using SecondOrderPerturbationTheory); recursive=true)


makedocs(;
    modules=[SecondOrderPerturbationTheory],
    authors="wwangnju <wwangnju@163.com>",
    repo="https://github.com/Quantum-Many-Body/SecondOrderPerturbationTheory.jl/blob/{commit}{path}#{line}",
    sitename="SecondOrderPerturbationTheory.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Quantum-Many-Body.github.io/SecondOrderPerturbationTheory.jl",
        edit_link="master",
        assets=String[],#["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "examples/Introduction.md",
            "examples/KitaevModel.md",
        ],
        "Manual" => "manual.md",
    ],
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/SecondOrderPerturbationTheory.jl",
    devbranch="master",
)
